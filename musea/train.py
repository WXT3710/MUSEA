import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .data import load_inputs, make_loaders
from .metrics import map_score
from .model import (Blip_Hash_NET, Center_Loss1, GraphContrastiveLoss,
                    LogSoftmaxContrastiveLoss_all_positive_samples,
                    SemanticAnchorLoss, SemanticDistillLoss)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def initialize(config, arrays):
    device = config['device']
    net = Blip_Hash_NET(config, config['bits']).to(device)
    optimizer = torch.optim.RMSprop(net.parameters(), lr=config['lr'],
                                   weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    align = LogSoftmaxContrastiveLoss_all_positive_samples(temperature=0.07)
    centers = Center_Loss1(config, config['bits'], range(config['n_class'])).to(device)
    optimizer.add_param_group({'params': centers.parameters(), 'lr': config['search_center_lr']})
    rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all())
    text = torch.from_numpy(arrays['text']).float().to(device)
    sd = SemanticDistillLoss(text, config['bits'], config['sd_global_weight'],
                            config['sd_local_weight'], config['sd_proto_weight'],
                            config['sd_temperature']).to(device)
    optimizer.add_param_group({'params': sd.parameters()})
    sa = SemanticAnchorLoss(text, config['bits']).to(device)
    optimizer.add_param_group({'params': sa.parameters()})
    graph = GraphContrastiveLoss(text, config['bits'], config['gc_knn'],
                                config['gc_temperature']).to(device)
    optimizer.add_param_group({'params': graph.parameters()})
    torch.set_rng_state(rng[0])
    torch.cuda.set_rng_state_all(rng[1])
    return net, optimizer, scheduler, align, centers, sd, sa, graph


@torch.no_grad()
def encode(loader, net, device):
    net.eval()
    codes, labels = [], []
    for images, y, _ in loader:
        _, z = net.encoder_q(images.to(device))
        codes.append(z.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(codes), np.concatenate(labels)


def evaluate(query, gallery, net, device, cutoff, output=None):
    q, ql = encode(query, net, device)
    g, gl = encode(gallery, net, device)
    result = {'binary_mAP': map_score(q, g, ql, gl, cutoff),
              'continuous_mAP': map_score(q, g, ql, gl, cutoff, binary=False)}
    if output is not None:
        np.savez_compressed(output, query=q, gallery=g, query_labels=ql, gallery_labels=gl,
                            query_ids=np.array([r['image_id'] for r in query.dataset.records]),
                            gallery_ids=np.array([r['image_id'] for r in gallery.dataset.records]))
    return result


def train(args, config):
    seed_everything(args.seed)
    datasets, arrays, protocol = load_inputs(config, args.images, args.assets,
                                            args.splits, args.check_data)
    if args.check_data:
        print(json.dumps({'counts': protocol['counts'], 'status': 'verified'}))
        return
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    public_config = dict(config, seed=args.seed, input_features_sha256=hashlib.sha256(
        (Path(args.assets) / 'features.npz').read_bytes()).hexdigest())
    (out / 'config.json').write_text(json.dumps(public_config, indent=2) + '\n')
    runtime = dict(config, device=torch.device(args.device), blip_pretrained_pth=args.blip,
                   blip_med_config=str(Path(__file__).parent / 'blip/med_config.json'), blip_vit_mode='base')
    loaders = make_loaders(datasets, config, args.seed, args.workers)
    net, optimizer, scheduler, align, centers, sd, sa, graph = initialize(runtime, arrays)
    digest = hashlib.sha256()
    for name, module in {'net': net, 'hash_criterion2': centers, 'sd_loss_fn': sd,
                         'sa_loss_fn': sa, 'gc_loss_fn': graph}.items():
        for key, value in module.state_dict().items():
            digest.update((name + '.' + key).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    (out / 'initialization.json').write_text(json.dumps({'sha256': digest.hexdigest(), 'seed': args.seed}))
    local = torch.from_numpy(arrays['local']).float()
    local = F.normalize(local - local.mean(0, keepdim=True), dim=-1)
    direct = torch.from_numpy(arrays['direct']).float()
    alignment = torch.from_numpy(arrays['alignment']).float()
    best, best_epoch = -1.0, None
    steps = {'main': 0, 'extra': 0}
    for epoch in range(config['epoch']):
        net.train()
        total = 0.0
        for images, y, indices in tqdm(loaders['train'], desc=f'Epoch {epoch + 1}/{config["epoch"]}'):
            images, y = images.to(args.device), y.to(args.device)
            onehot = F.one_hot(y, config['n_class']).float()
            optimizer.zero_grad()
            features, z, key = net(images)
            loss = align(features, alignment[indices].to(args.device), onehot)
            loss = loss + config['search_hash_weight'] * centers(z, key, onehot, indices, epoch)
            if len(z) > 1:
                teacher = F.normalize(direct[indices].to(args.device).float(), dim=-1)
                valid = teacher.norm(dim=-1) > 0.5
                if valid.sum() > 1:
                    t = teacher[valid]
                    h = F.normalize(z[valid], dim=-1)
                    loss = loss + config['direct_hash_weight'] * F.mse_loss(h @ h.T, t @ t.T)
            loss = loss + config['sd_batch_weight'] * sd.forward_global_batch(centers.centres, y)
            loss = loss + config['sa_weight'] * sa.forward_batch(centers.centres, y)
            loss = loss + config['gc_weight'] * graph.forward_batch(centers.centres, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), config['grad_clip'])
            optimizer.step()
            steps['main'] += 1
            total += loss.item()
        scheduler.step()
        for images, y, indices in loaders['extra']:
            optimizer.zero_grad()
            _, z, _ = net(images.to(args.device))
            loss = sd.forward_local(z, local[indices].to(args.device), y.to(args.device),
                                    mode=config['sd_local_mode'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), config['grad_clip'])
            optimizer.step()
            steps['extra'] += 1
        q, ql = encode(loaders['val_query'], net, args.device)
        g, gl = encode(loaders['val_gallery'], net, args.device)
        score = map_score(q, g, ql, gl, config['topK'])
        record = {'epoch': epoch + 1, 'validation_binary_mAP': score,
                  'mean_main_loss': total / len(loaders['train'])}
        with (out / 'validation.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
        if score > best:
            best, best_epoch = score, epoch + 1
            checkpoint = {'net': net.state_dict(), 'centers': centers.state_dict(),
                          'sd': sd.state_dict(), 'sa': sa.state_dict(), 'graph': graph.state_dict(),
                          'epoch': best_epoch, 'validation_binary_mAP': best, 'config': public_config}
            torch.save(checkpoint, out / 'selected.tmp.pt')
            (out / 'selected.tmp.pt').replace(out / 'selected.pt')
    net.load_state_dict(torch.load(out / 'selected.pt', map_location=args.device, weights_only=True)['net'])
    test = evaluate(loaders['test_query'], loaders['test_gallery'], net, args.device,
                    config['topK'], out / 'test_codes.npz')
    result = {'validation_binary_mAP': best, 'selected_epoch': best_epoch,
              'test_binary_mAP': test['binary_mAP'], 'test_continuous_mAP': test['continuous_mAP'],
              'cutoff': config['topK'], 'steps': steps, 'selection': config['selection']}
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
