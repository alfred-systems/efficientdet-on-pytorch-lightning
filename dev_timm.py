import timm
import torch

name = "hf_hub:timm/vit_base_patch16_clip_384.laion2b_ft_in12k_in1k"
name = "convnext_base.clip_laion2b"
model = timm.create_model(name, pretrained=True, features_only=True)
X = torch.randn(2, 3, 384, 384) 
out = model(X)
breakpoint()
print(out)
"""
name = "convnext_base.clip_laion2b"
(Pdb) p [o.shape for o in out]
[torch.Size([2, 128, 96, 96]), torch.Size([2, 256, 48, 48]), torch.Size([2, 512, 24, 24]), torch.Size([2, 1024, 12, 12])]
"""