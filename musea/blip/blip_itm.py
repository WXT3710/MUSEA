from .med import BertConfig, BertModel
import torch
from torch import nn
import torch.nn.functional as F
from .blip import create_vit, load_checkpoint

class BLIP_ITM(nn.Module):

    def __init__(self, med_config='configs/med_config.json', image_size=384, vit='base', vit_grad_ckpt=False, vit_ckpt_layer=0, embed_dim=256):
        super().__init__()
        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer)
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)
        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)
        self.itm_head = nn.Linear(text_width, 2)

def blip_itm(pretrained='', **kwargs):
    model = BLIP_ITM(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
        assert len(msg.missing_keys) == 0
    return model
