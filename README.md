## Environment

Python 3.12, CUDA 12.1.

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Training

```bash
python MUSEA_main.py --dataset AWA2 --images data/AWA2 --assets assets/AWA2 --blip weights/BLIP_base.pth --seed 42 --output outputs/AWA2_s42
python MUSEA_main.py --dataset CUB --images data/CUB --assets assets/CUB --blip weights/BLIP_base.pth --seed 42 --output outputs/CUB_s42
python MUSEA_main.py --dataset SUN --images data/SUN --assets assets/SUN --blip weights/BLIP_base.pth --seed 42 --output outputs/SUN_s42
```
