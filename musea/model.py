import torch
from torch import nn
import torch.nn.functional as F
from .blip import blip_itm

class Blip_Hash(nn.Module):

    def __init__(self, config, bit):
        super().__init__()
        self.hash_bit = bit
        self.blip = blip_itm(pretrained=config['blip_pretrained_pth'], med_config=config['blip_med_config'], image_size=224, vit=config['blip_vit_mode'])
        self.visual_encoder = self.blip.visual_encoder
        self.vision_proj = self.blip.vision_proj
        self.hash_head = nn.Sequential(nn.Linear(768, 512), nn.ReLU(), nn.Linear(512, self.hash_bit))

    def forward(self, x):
        vision_embeds = self.visual_encoder(x)
        cls_feat = vision_embeds[:, 0, :]
        proj_feat = F.normalize(cls_feat, dim=-1)
        hash_feat = self.hash_head(cls_feat)
        hash_feat = torch.tanh(hash_feat)
        return (proj_feat, hash_feat)

class Blip_Hash_NET(nn.Module):

    def __init__(self, config, hash_bit):
        super(Blip_Hash_NET, self).__init__()
        self.m = config['mome']
        self.encoder_q = Blip_Hash(config, hash_bit)
        self.encoder_k = Blip_Hash(config, hash_bit)
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    def forward(self, x):
        blip_f1, encode_x = self.encoder_q(x)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            blip_f2, encode_x2 = self.encoder_k(x)
        return (blip_f1, encode_x, encode_x2)

class LogSoftmaxContrastiveLoss_all_positive_samples(nn.Module):

    def __init__(self, temperature=0.07, exclude_self=True):
        super().__init__()
        self.temperature = temperature
        self.exclude_self = exclude_self
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, pred_feats, target_feats, labels):
        pred_feats = F.normalize(pred_feats, dim=-1)
        target_feats = F.normalize(target_feats, dim=-1)
        logits = torch.matmul(pred_feats, target_feats.T) / self.temperature
        log_probs = self.log_softmax(logits)
        with torch.no_grad():
            pos_mask = labels @ labels.T
            pos_mask = (pos_mask > 0).float()
            if self.exclude_self:
                pos_mask.fill_diagonal_(0.0)
            pos_count = pos_mask.sum(dim=1, keepdim=True)
            pos_mask = pos_mask / (pos_count + 1e-06)
        loss = -(pos_mask * log_probs).sum(dim=1).mean()
        return loss

class Center_Loss1(nn.Module):

    def __init__(self, config, bit, class_list):
        super(Center_Loss1, self).__init__()
        self.bit = bit
        self.n_class = len(class_list)
        self.device = config['device']
        init_mode = config.get('centre_init', 'bernoulli')
        if init_mode == 'bernoulli':
            centres_init = torch.bernoulli(torch.full((self.n_class, bit), 0.5)) * 2 - 1
        else:
            centres_init = torch.randn(self.n_class, bit)
        self.centres = nn.Parameter(centres_init)

    def forward(self, pred_hash, pred_hash2, label_onehot, ind, epoch):
        if self.centres.device != pred_hash.device:
            self.centres.data = self.centres.data.to(pred_hash.device)
        labels = torch.argmax(label_onehot, dim=1)
        batch_centres = self.centres[labels]
        loss_centre = F.mse_loss(pred_hash, batch_centres)
        pred_q_norm = F.normalize(pred_hash, p=2, dim=1)
        pred_k_norm = F.normalize(pred_hash2, p=2, dim=1)
        loss_consistency = F.mse_loss(pred_q_norm, pred_k_norm.detach())
        return loss_centre + 0.1 * loss_consistency

class SemanticDistillLoss(nn.Module):

    def __init__(self, vlm_text_prototypes, bit, global_weight=1.0, local_weight=5.0, proto_weight=0.1, temperature=0.2):
        super().__init__()
        D_vlm = vlm_text_prototypes.size(-1)
        self.register_buffer('vlm_text_proto', vlm_text_prototypes)
        self.bit = bit
        self.global_weight = global_weight
        self.local_weight = local_weight
        self.proto_weight = proto_weight
        self.temperature = temperature
        self.vlm_proj = nn.Linear(D_vlm, bit, bias=False)
        nn.init.orthogonal_(self.vlm_proj.weight)

    def forward_local(self, hash_codes, vlm_img_emb, labels, mode='kl'):
        B = hash_codes.size(0)
        if B <= 1:
            return torch.tensor(0.0, device=hash_codes.device)
        vlm_sim = F.normalize(vlm_img_emb, dim=-1) @ F.normalize(vlm_img_emb, dim=-1).T
        hash_sim = F.normalize(hash_codes, dim=-1) @ F.normalize(hash_codes, dim=-1).T
        if mode == 'mse':
            loss_local = F.mse_loss(hash_sim, vlm_sim)
        else:
            P_vlm = F.softmax(vlm_sim / self.temperature, dim=-1)
            logP_hash = F.log_softmax(hash_sim / self.temperature, dim=-1)
            loss_local = F.kl_div(logP_hash, P_vlm, reduction='batchmean')
        proj_vlm = self.vlm_proj(self.vlm_text_proto)
        unique_lbl = labels.unique()
        loss_proto = torch.tensor(0.0, device=hash_codes.device)
        for c in unique_lbl:
            m = labels == c
            if m.sum() == 0:
                continue
            loss_proto = loss_proto + F.mse_loss(hash_codes[m].mean(dim=0), proj_vlm[c])
        loss_proto = loss_proto / max(len(unique_lbl), 1)
        return self.local_weight * loss_local + self.proto_weight * loss_proto

    def forward_global_batch(self, hash_centers, labels, topk=None):
        unique_cls = labels.unique()
        C_sub = unique_cls.size(0)
        if C_sub <= 1:
            return torch.tensor(0.0, device=hash_centers.device)
        c_sub = hash_centers[unique_cls]
        t_sub = self.vlm_text_proto[unique_cls]
        vlm_sim = F.normalize(t_sub, dim=-1) @ F.normalize(t_sub, dim=-1).T
        hash_sim = F.normalize(c_sub, dim=-1) @ F.normalize(c_sub, dim=-1).T
        if topk is None:
            topk = max(3, C_sub // 4)
        topk = min(topk, C_sub - 1)
        _, idx = vlm_sim.topk(topk + 1, dim=-1)
        mask = torch.zeros_like(vlm_sim)
        mask.scatter_(1, idx, 1.0)
        mask.fill_diagonal_(0.0)
        P_vlm = F.softmax(vlm_sim / self.temperature, dim=-1) * mask
        P_vlm = P_vlm / (P_vlm.sum(dim=-1, keepdim=True) + 1e-08)
        logP_hash = F.log_softmax(hash_sim / self.temperature, dim=-1)
        return self.global_weight * F.kl_div(logP_hash, P_vlm, reduction='batchmean')

class SemanticAnchorLoss(nn.Module):

    def __init__(self, vlm_text_prototypes, bit):
        super().__init__()
        D_vlm = vlm_text_prototypes.size(-1)
        self.register_buffer('vlm_text_proto', vlm_text_prototypes)
        hidden = max(256, bit * 2)
        self.proj = nn.Sequential(nn.Linear(D_vlm, hidden, bias=False), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Linear(hidden, bit, bias=False))

    def forward_batch(self, hash_centers, labels):
        unique_cls = labels.unique()
        if unique_cls.size(0) <= 1:
            return torch.tensor(0.0, device=hash_centers.device)
        c_sub = hash_centers[unique_cls]
        t_sub = self.vlm_text_proto[unique_cls]
        proj_vlm = self.proj(t_sub)
        return F.mse_loss(c_sub, proj_vlm)

class GraphContrastiveLoss(nn.Module):

    def __init__(self, vlm_text_prototypes, bit, k=5, temperature=0.07):
        super().__init__()
        n_class, D_vlm = vlm_text_prototypes.shape
        self.register_buffer('vlm_text_proto', vlm_text_prototypes)
        self.bit = bit
        self.k = k
        self.temperature = temperature
        adj = self._build_affinity(vlm_text_prototypes, k)
        self.register_buffer('adj', adj)
        self.gcn = nn.Linear(bit, bit, bias=False)

    def _build_affinity(self, proto, k):
        proto_n = F.normalize(proto, dim=-1)
        sim = proto_n @ proto_n.T
        n = sim.size(0)
        k = min(k, n - 1)
        _, idx = sim.topk(k + 1, dim=-1)
        mask = torch.zeros_like(sim)
        mask.scatter_(1, idx, 1.0)
        mask.fill_diagonal_(0.0)
        adj = sim * mask
        adj = adj.clamp(min=0)
        deg = adj.sum(dim=1).clamp(min=1e-08)
        deg_inv_sqrt = deg.pow(-0.5)
        return deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)

    def forward_batch(self, hash_centers, labels):
        unique_cls = labels.unique()
        C_sub = unique_cls.size(0)
        if C_sub <= 1:
            return torch.tensor(0.0, device=hash_centers.device)
        c_sub = hash_centers[unique_cls]
        adj_sub = self.adj[unique_cls][:, unique_cls]
        propagated = self.gcn(c_sub)
        propagated = adj_sub @ propagated
        propagated_n = F.normalize(propagated, dim=-1)
        sim = propagated_n @ propagated_n.T / self.temperature
        pos_mask = (adj_sub > 0).float()
        pos_mask.fill_diagonal_(0.0)
        exp_sim = torch.exp(sim)
        exp_sim = exp_sim - exp_sim.diagonal().diag_embed()
        pos_sum = (exp_sim * pos_mask).sum(dim=1)
        all_sum = exp_sim.sum(dim=1)
        has_nbr = pos_mask.sum(dim=1) > 0
        if has_nbr.sum() == 0:
            return torch.tensor(0.0, device=hash_centers.device)
        loss = -torch.log((pos_sum[has_nbr] + 1e-08) / (all_sum[has_nbr] + 1e-08))
        return loss.mean()
