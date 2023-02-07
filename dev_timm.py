import timm
import torch

name = "hf_hub:timm/vit_base_patch16_clip_384.laion2b_ft_in12k_in1k"
name = "convnext_base.clip_laion2b"
name = "efficientnet_b0"
model = timm.create_model(name, pretrained=True, features_only=True)
X = torch.randn(2, 3, 512, 512) 
out = model(X)
print([o.shape for o in out])
breakpoint()
print(out)
"""
name = "convnext_base.clip_laion2b"
(Pdb) p [o.shape for o in out]
386x386
[torch.Size([2, 128, 96, 96]), torch.Size([2, 256, 48, 48]), torch.Size([2, 512, 24, 24]), torch.Size([2, 1024, 12, 12])]
512x512
[torch.Size([2, 128, 128, 128]), torch.Size([2, 256, 64, 64]), torch.Size([2, 512, 32, 32]), torch.Size([2, 1024, 16, 16])]

name = "efficientnet_b0"
512x512
[torch.Size([2, 16, 256, 256]), torch.Size([2, 24, 128, 128]), torch.Size([2, 40, 64, 64]), torch.Size([2, 112, 32, 32]), torch.Size([2, 320, 16, 16])]
"""