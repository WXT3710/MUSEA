import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['AWA2', 'CUB', 'SUN'], required=True)
    parser.add_argument('--images', required=True)
    parser.add_argument('--assets', required=True)
    parser.add_argument('--blip')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--check-data', action='store_true')
    args = parser.parse_args()
    if not args.check_data and (not args.blip or not args.output):
        parser.error('--blip and --output are required for training')
    root = Path(__file__).resolve().parent
    config = json.loads((root / 'configs' / (args.dataset + '.json')).read_text())
    args.splits = root / 'splits' / args.dataset
    from musea.train import train
    train(args, config)


if __name__ == '__main__':
    main()
