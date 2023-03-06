from src.model.utils import *
from loguru import logger


class Classifier(nn.Module):

    def __init__(self,
                 num_levels: int,
                 depth: int,
                 width: int,
                 num_anchors: int,
                 num_classes: int,
                 Act: nn.Module = nn.SiLU(),
                 background_class: bool = True,
                 output_prob: bool = True,
                 ):
        self.use_background_class = background_class
        num_classes += int(background_class)
        self.output_prob = output_prob
        
        self.num_levels, self.num_anchors, self.num_classes \
            = num_levels, num_anchors, num_classes

        super().__init__()

        self.conv_layers = nn.ModuleList([
            nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1, bias=False)
            # Seperable_Conv2d(width, width, 3, 1, bias=True)
            for _ in range(depth)
        ])
        self.bn_layers = nn.ModuleList([
            nn.ModuleList([nn.BatchNorm2d(width) for _ in range(depth)])
            for _ in range(num_levels)
        ])
        self.act = Act

        self.conv_pred = nn.Conv2d(width, num_anchors * num_classes, kernel_size=3, stride=1, padding=1, bias=True)
        # self.conv_pred = Seperable_Conv2d(width, num_anchors * num_classes, bias=True)

    def forward(self, features, flatten=True):
        out = []
        for i in range(self.num_levels):
            f = features[i]

            for conv, bn in zip(self.conv_layers, self.bn_layers[i]):
                f = conv(f)
                f = bn(f)
                f = self.act(f)

            pred = self.conv_pred(f)

            if flatten:
                pred = pred.permute(0, 2, 3, 1)
                pred = pred.contiguous().view(pred.shape[0], pred.shape[1], pred.shape[2], self.num_anchors, self.num_classes)
                pred = pred.contiguous().view(pred.shape[0], -1, self.num_classes)
            if self.output_prob:
                if self.use_background_class:
                    pred = torch.nn.functional.softmax(pred, dim=-1)
                else:
                    pred = torch.nn.functional.sigmoid(pred)
            out.append(pred)
        
        if flatten:
            out = torch.cat(out, dim=1)
        return out

    
class Encoder(nn.Module):

    def __init__(self,
                 num_levels: int,
                 depth: int,
                 width: int,
                 num_anchors: int,
                 num_classes: int,
                 Act: nn.Module = nn.SiLU(),
                 ):
        self.num_levels, self.num_anchors, self.num_classes \
            = num_levels, num_anchors, num_classes

        super().__init__()

        self.conv_layers = nn.ModuleList([
            nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1, bias=False)
            # Seperable_Conv2d(width, width, 3, 1, bias=True)
            for _ in range(depth)
        ])
        self.bn_layers = nn.ModuleList([
            nn.ModuleList([nn.BatchNorm2d(width) for _ in range(depth)])
            for _ in range(num_levels)
        ])
        self.act = Act

        self.conv_pred = nn.Conv2d(width, num_anchors * num_classes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.logit_scale = nn.Parameter(torch.tensor([0.07]))

    def forward(self, features, flatten=True):
        out = []
        for i in range(self.num_levels):
            f = features[i]

            for conv, bn in zip(self.conv_layers, self.bn_layers[i]):
                f = conv(f)
                f = bn(f)
                f = self.act(f)

            pred = self.conv_pred(f)

            if flatten:
                pred = pred.permute(0, 2, 3, 1)
                pred = pred.contiguous().view(pred.shape[0], pred.shape[1], pred.shape[2], self.num_anchors, self.num_classes)
                pred = pred.contiguous().view(pred.shape[0], -1, self.num_classes)
            # pred = pred * torch.clamp(torch.exp(self.logit_scale), min=1e-4, max=100)
            out.append(pred)
        
        if flatten:
            out = torch.cat(out, dim=1)
        return out


class Regressor(nn.Module):

    def __init__(self,
                 num_levels: int,
                 depth: int,
                 width: int,
                 num_anchors: int,
                 Act: nn.Module = nn.SiLU()
                 ):

        self.num_levels = num_levels

        super().__init__()

        self.conv_layers = nn.ModuleList([
            Seperable_Conv2d(width, width, 3, 1, bias=True)
            for _ in range(depth)
        ])
        self.bn_layers = nn.ModuleList([
            nn.ModuleList([nn.BatchNorm2d(width) for _ in range(depth)])
            for _ in range(num_levels)
        ])
        self.act = Act
        self.conv_pred = Seperable_Conv2d(width, num_anchors * 4, bias=True)

    def forward(self, features):
        out = []
        for i in range(self.num_levels):
            f = features[i]

            for conv, bn in zip(self.conv_layers, self.bn_layers[i]):
                f = conv(f)
                f = bn(f)
                f = self.act(f)

            pred = self.conv_pred(f)

            pred = pred.permute(0, 2, 3, 1).contiguous().view(pred.shape[0], -1, 4)

            out.append(pred)
        out = torch.cat(out, dim=1)

        return out


class EfficientDet_Head(nn.Module):

    def __init__(self,
                 num_levels: int,
                 depth: int,
                 width: int,
                 num_anchors: int,
                 num_classes: int,
                 Act: nn.Module = nn.SiLU(),
                 background_class: bool = True,
                 ):

        super().__init__()
        
        self.classifier = Classifier(num_levels, depth, width, num_anchors, num_classes, Act, background_class=background_class)
        self.regressor = Regressor(num_levels, depth, width, num_anchors, Act)

    def forward(self, features):
        reg_out = self.regressor(features)
        cls_out = self.classifier(features)
        out = torch.cat((reg_out, cls_out), dim=2)
        return out


class ClipDet_Head(nn.Module):
    """
    overhere encoder will output a embeding of size "embed_size" per anchor box,
    instead of a classification prob distribution.
    and classifier are only for distigish object and background.
    """
    def __init__(self,
                 num_levels: int,
                 depth: int,
                 width: int,
                 num_anchors: int,
                 embed_size: int,
                 Act: nn.Module = nn.SiLU(),
                 background_class: bool = True,
                 ):
        super().__init__()

        self.num_anchors = num_anchors
        self.embed_size = embed_size

        if background_class:
            logger.warning("ClipDet_Head will alway using sigmoid activation, hence ignore background_class parameter.")
        
        self.classifier = Classifier(
            num_levels, depth, width, num_anchors, 1, 
            Act, background_class=False
        )
        self.encoder = Encoder(
            num_levels, depth, width, num_anchors, embed_size, Act,
        )
        self.regressor = Regressor(num_levels, depth, width, num_anchors, Act)

    def forward(self, features, flatten_emb=True):
        reg_out = self.regressor(features)
        emb_out = self.encoder(features, flatten=flatten_emb)
        cls_out = self.classifier(features)
        
        if flatten_emb:
            out = torch.cat((reg_out, cls_out, emb_out), dim=2)
            return out
        else:
            out = torch.cat((reg_out, cls_out), dim=2)
            mean_anchor_emb = []
            for emb in emb_out:
                b, n, h, w = emb.shape
                emb = emb.view([b, self.num_anchors, self.embed_size, h, w])
                mean_anchor_emb.append(emb.mean(dim=1))
            return out, mean_anchor_emb


class ClipFuseDet_Head(nn.Module):
    """
    overhere encoder will output a embeding of size "embed_size" per anchor box,
    instead of a classification prob distribution.
    and classifier are only for distigish object and background.
    """
    def __init__(self,
                 num_levels: int,
                 depth: int,
                 width: int,
                 num_anchors: int,
                 embed_size: int,
                 Act: nn.Module = nn.SiLU(),
                 background_class: bool = True,
                 ):
        super().__init__()

        self.num_anchors = num_anchors
        self.embed_size = embed_size

        if background_class:
            logger.warning("ClipDet_Head will alway using sigmoid activation, hence ignore background_class parameter.")
        
        self.vl_proj = torch.nn.Sequential(
            torch.nn.Conv2d(
                width + embed_size, 
                width + embed_size, 
                kernel_size=1, 
                padding=0,
                bias=True
            ),
            torch.nn.BatchNorm2d(width + embed_size),
            torch.nn.ReLU(),
            
            torch.nn.Conv2d(
                width + embed_size, 
                width + embed_size, 
                kernel_size=1, 
                padding=0,
                bias=True
            ),
            torch.nn.BatchNorm2d(width + embed_size),
            torch.nn.ReLU(),
            
            torch.nn.Conv2d(
                width + embed_size, 
                width, 
                kernel_size=1, 
                padding=0,
                bias=True
            ),
        )
        
        self.classifier = Classifier(
            num_levels, depth + 2, width, num_anchors, 1, 
            Act, background_class=False
        )
        self.regressor = Regressor(num_levels, depth, width, num_anchors, Act)

    def forward(self, features: List[Tensor], query_embed: Union[List[Tensor], Tensor]):
        """
        features: (b, c, h, w) x num of BiFPN layers
        query_embed: (b, clip_embed_size, ) or (num_regions, clip_embed_size, ) x batch_size
        """
        reg_out = self.regressor(features)
        cls_features = []
        for feat in features:
            b, c, h, w = feat.shape
            query_embed = F.normalize(query_embed, dim=-1)
            query_4d = query_embed.unsqueeze(-1).unsqueeze(-1)
            query_4d = query_4d.repeat(1, 1, h, w)
            feat = self.vl_proj(torch.cat([feat, query_4d], dim=1))
            cls_features.append(feat)
        cls_out = self.classifier(cls_features)
        
        out = torch.cat((reg_out, cls_out), dim=2)
        return out