import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class Images(Dataset):
    def __init__(self, root, records):
        self.root = Path(root)
        self.records = records
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(self.root / record['image_id']) as image:
            value = self.transform(image.convert('RGB'))
        return value, record['label'], index


def load_inputs(config, image_root, asset_root, split_root, check_images=False):
    rows = {}
    split_root = Path(split_root)
    for name in ('train', 'val_query', 'val_gallery', 'test_query', 'test_gallery'):
        with (split_root / (name + '.csv')).open() as stream:
            records = list(csv.DictReader(stream))
        for row in records:
            row['label'] = int(row['label'])
            assert 0 <= row['label'] < config['n_class']
            path = Path(row['image_id'])
            assert not path.is_absolute() and '..' not in path.parts
            if check_images:
                assert (Path(image_root) / path).is_file(), path
        assert len(records) == len({r['image_id'] for r in records})
        rows[name] = records
    protocol = json.loads((split_root / 'protocol.json').read_text())
    ids = {k: {r['image_id'] for r in v} for k, v in rows.items()}
    assert not ids['val_query'] & ids['val_gallery']
    assert not ids['test_query'] & ids['test_gallery']
    assert not ids['train'] & (ids['val_query'] | ids['val_gallery'] | ids['test_query'])
    held = set(protocol['heldout_classes'])
    assert held <= set(protocol['seen_classes'])
    assert {r['label'] for r in rows['val_query']} <= held
    assert {r['label'] for r in rows['val_gallery']} <= held
    assert not held & set(protocol['unseen_classes'])
    assert not held & {r['label'] for r in rows['train'] if r['kind'] == 'real'}
    if config['dataset'] == 'SUN':
        assert len(protocol['seen_classes']) == 645 and len(protocol['unseen_classes']) == 72
        assert all(r['kind'] == 'real' and r['label'] in protocol['seen_classes'] for r in rows['train'])
    path = Path(asset_root) / 'features.npz'
    with np.load(path, allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    assert arrays['image_ids'].tolist() == [r['image_id'] for r in rows['train']]
    assert arrays['labels'].tolist() == [r['label'] for r in rows['train']]
    for key in ('alignment', 'local', 'direct', 'text'):
        assert arrays[key].ndim == 2 and np.isfinite(arrays[key]).all(), key
    assert arrays['alignment'].shape == (len(rows['train']), 768)
    assert arrays['local'].shape[0] == arrays['direct'].shape[0] == len(rows['train'])
    assert arrays['text'].shape[0] == config['n_class']
    datasets = {k: Images(image_root, v) for k, v in rows.items()}
    return datasets, arrays, protocol


def batch_plan(size, epochs, extra_batches, seed, batch_size=32):
    rng = np.random.default_rng(seed)
    main = np.stack([rng.permutation(size) for _ in range(epochs)])
    extra = np.stack([rng.permutation(size)[:extra_batches * batch_size]
                     .reshape(extra_batches, batch_size) for _ in range(epochs)])
    return main, extra


class Batches:
    def __init__(self, indices, batch_size=None):
        self.indices = indices
        self.batch_size = batch_size
        self.epoch = 0

    def __iter__(self):
        indices = self.indices[self.epoch]
        self.epoch += 1
        if self.batch_size is None:
            yield from indices.tolist()
        else:
            yield from (indices[i:i + self.batch_size].tolist()
                        for i in range(0, len(indices), self.batch_size))

    def __len__(self):
        size = len(self.indices[0])
        return size if self.batch_size is None else (size + self.batch_size - 1) // self.batch_size


def make_loaders(datasets, config, seed, workers):
    main, extra = batch_plan(len(datasets['train']), config['epoch'],
                             config['sd_local_batches'], seed, config['batch_size'])
    loaders = {}
    for name, sampler, clock in [('train', Batches(main, 32), 4201),
                                  ('extra', Batches(extra), 4205)]:
        loaders[name] = DataLoader(datasets['train'], batch_sampler=sampler,
                                  num_workers=workers, pin_memory=True,
                                  generator=torch.Generator().manual_seed(clock))
    for name, clock in [('val_query', 4203), ('val_gallery', 4204),
                        ('test_query', 4208), ('test_gallery', 4208)]:
        loaders[name] = DataLoader(datasets[name], batch_size=32, shuffle=False,
                                  num_workers=workers,
                                  generator=torch.Generator().manual_seed(clock))
    return loaders
