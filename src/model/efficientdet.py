import timm
from src.model.utils import *
from src.model.backbone import EfficientNet_Backbone, FeatureExtractor, FeaturePicker
from src.model.fpn import BiFPN
from src.model.head import EfficientDet_Head, ClipDet_Head
from src.model.anchor import Anchor_Maker



class RetinaNet_Frame(nn.Module):

    anchor_sizes = None
    anchor_scales = None
    anchor_ratios = None
    strides = None

    def __init__(self,
                 img_size: int,
                 background_class: bool = True,
                 freeze_backbone: bool=False):

        print(f'The model is for images sized in {img_size}x{img_size}, freeze_backbone: {freeze_backbone}')
        super().__init__()

        self.num_anchors = len(self.anchor_scales) * len(self.anchor_ratios)
        self.freeze_backbone = freeze_backbone

        self.backbone = None
        self.fpn = None
        self.head = None

        self.anchors = self.retinanet_anchors(img_size, self.anchor_sizes, self.anchor_scales, self.anchor_ratios, self.strides)


    def forward(self, input, detect: bool = False):
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.backbone(input)
        else:
            features = self.backbone(input)

        features = self.fpn(features)
        out = self.head(features)
        if detect:
            self.detect(out)
        return out, self.anchors


    def detect(self, out):
        self.anchors = self.anchors.to(out.device)
        out[..., :2] = self.anchors[..., :2] + (out[..., :2] * self.anchors[..., 2:])
        out[..., 2:4] = torch.exp(out[..., 2:4]) * self.anchors[..., 2:]


    def retinanet_anchors(self, img_size, anchor_sizes, anchor_scales, anchor_ratios, strides):
        anchor_priors = self.retinanet_anchor_priors(anchor_sizes, anchor_scales, anchor_ratios, strides)
        anchors = Anchor_Maker(anchor_priors, strides, True, False, False)(img_size)
        return anchors


    @classmethod
    def retinanet_anchor_priors(cls, anchor_sizes, anchor_scales, anchor_ratios, strides):
        anchor_priors = []

        for stride, size in zip(strides, anchor_sizes):
            stride_priors = [[(size / stride) * s * r[0], (size / stride) * s * r[1]]
                             for s in anchor_scales
                             for r in anchor_ratios]

            anchor_priors.append(torch.Tensor(stride_priors))

        return anchor_priors


class EfficientDet(RetinaNet_Frame):

    resolutions = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    survival_probs = [None, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]

    config = {'bifpn_depth': [3, 4, 5, 6, 7, 7, 8, 8, 8],
              'bifpn_width': [96, 96, 112, 160, 224, 288, 384, 384, 384],
              'head_depth':  [3, 3, 3, 4, 4, 4, 5, 5, 5],
              'head_width':  [96, 96, 112, 160, 224, 288, 384, 384, 384]}

    anchor_sizes = [32, 64, 128, 256, 512]
    anchor_scales = [1, 2 ** (1 / 3), 2 ** (2 / 3)]
    anchor_ratios = [[1, 1], [1.4, 0.7], [0.7, 1.4]]

    strides = [8, 16, 32, 64, 128]

    def __init__(self,
                 coeff: int,
                 num_classes: int = 80,
                 background_class: bool = True,
                 pretrained: bool = False,
                 pretrained_backbone: bool = False,
                 **kwargs):

        self.img_size = self.resolutions[coeff]

        if coeff == 7:
            self.anchor_sizes = [40, 80, 160, 320, 640]

        if coeff == 8:
            self.anchor_sizes = [32, 64, 128, 256, 512, 1024]
            self.strides = [8, 16, 32, 64, 128, 256]

        num_levels = len(self.strides)

        d_bifpn = self.config['bifpn_depth'][coeff]
        w_bifpn = self.config['bifpn_width'][coeff]
        d_head = self.config['head_depth'][coeff]
        w_head = self.config['head_width'][coeff]

        survival_prob = self.survival_probs[coeff]

        super().__init__(self.img_size, **kwargs)

        self.backbone = timm.create_model(f"efficientnet_b{coeff}", pretrained=True, features_only=True)
        self.backbone = FeaturePicker(self.backbone, [2, 3, 4])

        widths = [16, 24, 40, 112, 320]
        channels = widths[2:]
        
        # [(8, 40, 64, 64), (8, 112, 32, 32), (8, 320, 16, 16)] => 
        # BiFPN => 
        # [(8, 96, 64, 64), (8, 96, 32, 32), (8, 96, 16, 16), (8, 96, 8, 8), (8, 96, 4, 4)]
        self.fpn = BiFPN(num_levels, d_bifpn, channels, w_bifpn, Act=nn.ReLU())

        self.head = EfficientDet_Head(num_levels, d_head, w_head, self.num_anchors, num_classes, nn.ReLU(), background_class=background_class)

        if pretrained:
            load_pretrained(self, 'efficientdet_d' + str(coeff))

            
class ConvNeXtDet(RetinaNet_Frame):

    resolutions = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    survival_probs = [None, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]

    config = {'bifpn_depth': [3, 4, 5, 6, 7, 7, 8, 8, 8],
              'bifpn_width': [96, 96, 112, 160, 224, 288, 384, 384, 384],
              'head_depth':  [3, 3, 3, 4, 4, 4, 5, 5, 5],
              'head_width':  [96, 96, 112, 160, 224, 288, 384, 384, 384]}

    anchor_sizes = [32, 64, 128, 256, 512]
    anchor_scales = [1, 2 ** (1 / 3), 2 ** (2 / 3)]
    anchor_ratios = [[1, 1], [1.4, 0.7], [0.7, 1.4]]

    strides = [8, 16, 32, 64, 128]

    def __init__(self,
                 coeff: int,
                 num_classes: int = 80,
                 background_class: bool = True,
                 **kwargs):
        coeff = 0
        self.img_size = self.resolutions[coeff]

        num_levels = len(self.strides)

        d_bifpn = self.config['bifpn_depth'][coeff]
        w_bifpn = self.config['bifpn_width'][coeff]
        d_head = self.config['head_depth'][coeff]
        w_head = self.config['head_width'][coeff]

        super().__init__(self.img_size, **kwargs)

        """
        'convnext_base.clip_laion2b': _cfg(
            hf_hub_id='laion/CLIP-convnext_base_w-laion2B-s13B-b82K',
            hf_hub_filename='open_clip_pytorch_model.bin',
            mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
            input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=1.0, num_classes=640),
        """
        self.backbone = timm.create_model(f"convnext_base.clip_laion2b", pretrained=True, features_only=True)
        self.backbone = FeaturePicker(self.backbone, [1, 2, 3])

        widths = [128, 256, 512, 1024]
        channels = widths[1:]

        self.fpn = BiFPN(num_levels, d_bifpn, channels, w_bifpn, Act=nn.ReLU())

        self.head = EfficientDet_Head(num_levels, d_head, w_head, self.num_anchors, num_classes, nn.ReLU(), background_class=background_class)


class ClipDet(RetinaNet_Frame):

    resolutions = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    survival_probs = [None, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]

    config = {'bifpn_depth': [3, 4, 5, 6, 7, 7, 8, 8, 8],
              'bifpn_width': [256, 256, 112, 160, 224, 288, 384, 384, 384],
              'head_depth':  [2, 3, 3, 4, 4, 4, 5, 5, 5],
              'head_width':  [256, 256, 112, 160, 224, 288, 384, 384, 384]}

    anchor_sizes = [32, 64, 128, 256, 512]
    anchor_scales = [1, 2 ** (1 / 3), 2 ** (2 / 3)]
    anchor_ratios = [[1, 1], [1.4, 0.7], [0.7, 1.4]]

    strides = [8, 16, 32, 64, 128]

    def __init__(self,
                 coeff: int,
                 freeze_backbone: bool=False,
                 **kwargs):
        coeff = 0
        self.img_size = self.resolutions[coeff]
        num_levels = len(self.strides)

        d_bifpn = self.config['bifpn_depth'][coeff]
        w_bifpn = self.config['bifpn_width'][coeff]
        d_head = self.config['head_depth'][coeff]
        w_head = self.config['head_width'][coeff]

        super().__init__(self.img_size, freeze_backbone=freeze_backbone, **kwargs)
        self.freeze_backbone = freeze_backbone

        """
        'convnext_base.clip_laion2b': _cfg(
            hf_hub_id='laion/CLIP-convnext_base_w-laion2B-s13B-b82K',
            hf_hub_filename='open_clip_pytorch_model.bin',
            mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
            input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=1.0, num_classes=640),
        """
        self.backbone = timm.create_model(f"convnext_base.clip_laion2b", pretrained=True, features_only=True)
        self.backbone = FeaturePicker(self.backbone, [1, 2, 3])

        widths = [128, 256, 512, 1024]
        channels = widths[1:]

        self.fpn = BiFPN(num_levels, d_bifpn, channels, w_bifpn, Act=nn.ReLU())
        self.head = ClipDet_Head(num_levels, d_head, w_head, self.num_anchors, 640, nn.ReLU())
        # self.global_emb_projs = nn.ModuleList([torch.nn.Linear(w_head, 640) for _ in range(3)])
    
    def forward(self, input, detect: bool=False, global_feat: bool=False):
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.backbone(input)
        else:
            features = self.backbone(input)
        
        features = self.fpn(features)
        
        if global_feat:
            out, features = self.head(features, flatten_emb=False)
            global_avgs = [
                feat.mean(dim=[2, 3])
                for feat in features[2:]
            ]
            return out, self.anchors, global_avgs
        else:
            out = self.head(features)
            if detect:
                self.detect(out)
            return out, self.anchors



