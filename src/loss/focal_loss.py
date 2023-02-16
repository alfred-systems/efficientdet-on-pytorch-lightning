from src.__init__ import *
from src.model.anchor import Anchor_Assigner
from loguru import logger


def new_focal_loss(logits, targets, alpha: float, gamma: float, normalizer, label_smoothing: float = 0.01):
    """Compute the focal loss between `logits` and the golden `target` values.
    'New' is not the best descriptor, but this focal loss impl matches recent versions of
    the official Tensorflow impl of EfficientDet. It has support for label smoothing, however
    it is a bit slower, doesn't jit optimize well, and uses more memory.
    Focal loss = -(1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    Args:
        logits: A float32 tensor of size [batch, height_in, width_in, num_predictions].
        targets: A float32 tensor of size [batch, height_in, width_in, num_predictions].
        alpha: A float32 scalar multiplying alpha to the loss from positive examples
            and (1-alpha) to the loss from negative examples.
        gamma: A float32 scalar modulating loss from hard and easy examples.
        normalizer: Divide loss by this value.
        label_smoothing: Float in [0, 1]. If > `0` then smooth the labels.
    Returns:
        loss: A float32 scalar representing normalized total loss.
    """
    # compute focal loss multipliers before label smoothing, such that it will not blow up the loss.
    pred_prob = logits
    # pred_prob = logits.sigmoid()
    
    targets = targets.to(logits.dtype)
    onem_targets = 1. - targets
    p_t = (targets * pred_prob) + (onem_targets * (1. - pred_prob))
    alpha_factor = targets * alpha + onem_targets * (1. - alpha)
    modulating_factor = (1. - p_t) ** gamma

    # apply label smoothing for cross_entropy for each entry.
    if label_smoothing > 0.:
        targets = targets * (1. - label_smoothing) + .5 * label_smoothing
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

    # compute the final loss and return
    element_wise_loss = (1 / normalizer) * alpha_factor * modulating_factor * ce
    return element_wise_loss.sum()


def clip_loss(image_embeddings, text_embeddings, temperature=20.0):
    # image_embeddings = F.normalize(image_embeddings, dim=1)
    # text_embeddings = F.normalize(text_embeddings, dim=1)

    logits = (text_embeddings @ image_embeddings.T) / temperature
    images_similarity = image_embeddings @ image_embeddings.T
    texts_similarity = text_embeddings @ text_embeddings.T
    targets = F.softmax(
        (images_similarity + texts_similarity) / 2 * temperature, dim=-1
    )
    # breakpoint()
    images_loss = (-targets.T * F.log_softmax(logits.T, dim=-1)).sum(1)
    # texts_loss = (-targets * F.log_softmax(logits, dim=-1)).sum(1)
    # return (images_loss + texts_loss) / 2.0
    return images_loss


class FocalL1_Loss(nn.Module):

    def __init__(self,
                 fore_th: float,
                 back_th: float,
                 alpha: float = 0.25,
                 gamma: float = 1.5,
                 beta: float = 0.1,
                 fore_mean: bool = True,
                 reg_weight: Optional[float] = None,
                 average: bool = True,
                 bbox_format: str = 'cxcywh'
                 ):

        super().__init__()

        self.fore_th = fore_th
        self.back_th = back_th
        self.anchor_assigner = Anchor_Assigner(fore_th, back_th, False, False, bbox_format)

        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.fore_mean = fore_mean

        self.reg_weight = reg_weight if reg_weight else 1.0
        self.average = average
    
    @property
    def accept_softmax(self):
        if hasattr(self, '_accept_softmax'):
            return self._accept_softmax
        else:
            return False
    
    @accept_softmax.setter
    def accept_softmax(self, v: bool):
        if not hasattr(self, '_accept_softmax'):
            logger.warning(f"focal_loos.accept_softmax = {v}")
        self._accept_softmax = v

    @staticmethod
    def focal_loss(cls_pred, fore_idx, back_idx, fore_label_cls, alpha, gamma, mean):

        fore_pred = cls_pred[fore_idx]
        back_pred = cls_pred[back_idx]

        fore_pred_t = torch.where(fore_label_cls == 1, fore_pred, 1 - fore_pred)  # _t for denote target
        back_pred_t = 1 - back_pred

        fore_alpha_t = torch.where(fore_label_cls == 1, alpha, 1 - alpha)
        back_alpha_t = 1 - alpha

        fore_weight = -1 * fore_alpha_t * torch.pow(1 - fore_pred_t, gamma)
        back_weight = -1 * back_alpha_t * torch.pow(1 - back_pred_t, gamma)

        fore_loss = fore_weight * torch.log(fore_pred_t)
        back_loss = back_weight * torch.log(back_pred_t)

        floss = torch.sum(fore_loss)
        bloss = torch.sum(back_loss)
        # cap = torch.max(floss * 10, torch.ones_like(floss) * 100)
        # loss = floss + bloss.clamp(max=cap)
        loss = floss + bloss
        if mean:
            num = fore_idx.size(0)
            loss = loss / num if num > 0 else loss

        return loss
    
    @staticmethod
    def focal_smax_loss(cls_pred, fore_idx, back_idx, fore_label_cls, alpha, gamma, mean):

        fore_pred = cls_pred[fore_idx]
        back_pred = cls_pred[back_idx]
        # bg_label = torch.zeros_like(back_pred)
        # bg_label[..., -1] += 1

        fore_pred_t = torch.where(fore_label_cls == 1, fore_pred, 1 - fore_pred)
        back_pred_t = torch.cat([1 - back_pred[..., :-1], back_pred[..., -1:]], dim=-1)

        fore_alpha_t = torch.where(fore_label_cls == 1, alpha, 1 - alpha)
        back_alpha_t = torch.cat([
            torch.ones_like(back_pred[..., :-1]) - alpha,
            torch.zeros_like(back_pred[..., -1:]) + alpha,
        ], dim=-1)

        fore_weight = -1 * fore_alpha_t * torch.pow(1 - fore_pred_t, gamma)
        back_weight = -1 * back_alpha_t * torch.pow(1 - back_pred_t, gamma)

        fore_loss = fore_weight * torch.log(fore_pred_t)
        back_loss = back_weight * torch.log(back_pred_t)

        loss = torch.sum(fore_loss) + torch.sum(back_loss)
        if mean:
            num = fore_idx.size(0)
            loss = loss / num if num > 0 else loss

        return loss


    @staticmethod
    def smooothL1_loss(reg_pred, anchors, fore_idx, fore_label_bbox, beta, mean):
        fore_pred = reg_pred[fore_idx]
        fore_anchor = anchors.squeeze()[fore_idx]

        reg_label = torch.zeros_like(fore_label_bbox)

        reg_label[..., 0] = (fore_label_bbox[..., 0] - fore_anchor[..., 0]) / fore_anchor[..., 2]
        reg_label[..., 1] = (fore_label_bbox[..., 1] - fore_anchor[..., 1]) / fore_anchor[..., 3]
        reg_label[..., 2] = torch.log(fore_label_bbox[..., 2].clamp(min=1) / fore_anchor[..., 2])
        reg_label[..., 3] = torch.log(fore_label_bbox[..., 3].clamp(min=1) / fore_anchor[..., 3])

        mae = torch.abs(reg_label - fore_pred)

        loss = torch.where(torch.le(mae, beta), 0.5 * (mae ** 2) / beta, mae - 0.5 * beta)
        loss = torch.sum(loss)
        if mean:
            num = 4 * fore_idx.size(0)
            loss = loss / num if num > 0 else loss

        return loss


    def forward(self,
                preds: Tensor,
                anchors: Tensor,
                labels: Tensor
                ) -> Tuple[Tensor, Tensor, Tensor]:

        if len(preds.shape) != 3:
            raise ValueError("preds should be given in 3d tensor")

        if len(anchors.shape) != 3:
            raise ValueError("anchors should be given in 3d tensor")

        if len(labels.shape) != 3:
            raise ValueError("labels should be given in 3d tensor")

        reg_preds = preds[..., :4]
        cls_preds = preds[..., 4:]
        redu_prob = cls_preds.sum(dim=-1)
        softmax = torch.allclose(redu_prob, torch.ones_like(redu_prob))
        
        cls_preds = cls_preds.clamp(1e-6, 1.0 - 1e-6)
        self.accept_softmax = softmax

        target_assigns = self.anchor_assigner(labels, anchors)
        cls_losses, reg_losses = [], []

        for i, assign in enumerate(target_assigns):
            fore_idx = assign['foreground'][0]
            back_idx = assign['background'][0]

            fore_label_cls = assign['foreground'][1][..., 4:]
            fore_label_bbox = assign['foreground'][1][..., :4]

            if softmax:
                cls_losses.append(
                    self.focal_smax_loss(
                        cls_preds[i], fore_idx, back_idx, fore_label_cls, 
                        self.alpha, self.gamma, self.fore_mean))
            else:
                cls_losses.append(
                    self.focal_loss(
                        cls_preds[i], fore_idx, back_idx, fore_label_cls, 
                        self.alpha, self.gamma, self.fore_mean))
            reg_losses.append(self.smooothL1_loss(reg_preds[i], anchors, fore_idx, fore_label_bbox, self.beta, self.fore_mean))

        cls_loss = sum(cls_losses)
        reg_loss = sum(reg_losses)
        total_loss = cls_loss + self.reg_weight * reg_loss

        if self.average:
            batch = len(target_assigns)
            total_loss /= batch
            cls_loss /= batch
            reg_loss /= batch

        return total_loss, cls_loss, reg_loss


class ContrastiveL1_Loss(FocalL1_Loss):
    """
    CLIP Contrastive loss for training classification head embedding,
    L1 loss for boudning box regression.
    """
    def __init__(self,
                 fore_th: float,
                 back_th: float,
                 alpha: float = 0.25,
                 gamma: float = 1.5,
                 beta: float = 0.1,
                 fore_mean: bool = True,
                 reg_weight: Optional[float] = None,
                 average: bool = True,
                 bbox_format: str = 'cxcywh',
                 temperature: float = 1.0,
                 ):

        super().__init__(
            fore_th, back_th, alpha, gamma, beta, 
            fore_mean, reg_weight, average, bbox_format)

        self.anchor_assigner = Anchor_Assigner(fore_th, back_th, False, False, bbox_format, label_type='embed')
        self.temperature = temperature
    
    @staticmethod
    def sample_fore_background_boxes(anchor_emb, fore_idx, back_idx, fore_label_emb, num_neg_samples=200):
        fore_emb = anchor_emb[fore_idx]
        back_emb = anchor_emb[back_idx]
        back_emb = back_emb[torch.randperm(num_neg_samples)]
        return fore_emb, fore_label_emb, back_emb
    
    @staticmethod
    def contrastive_loss_v1(anchor_emb, fore_idx, back_idx, fore_label_emb, num_neg_samples=1000, temperature=20.0):
        """
        Original CLIP loss
        
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        images_loss = (-targets.T * self.log_softmax(logits.T)).sum(1)
        """
        fore_emb = anchor_emb[fore_idx]
        back_emb = anchor_emb[back_idx]
        back_emb = back_emb[torch.randperm(num_neg_samples)]

        # fore_emb = F.normalize(fore_emb, dim=1)
        # back_emb = F.normalize(back_emb, dim=1)

        fore_img_similarity = fore_emb @ fore_emb.T
        fore_txt_similarity = fore_label_emb @ fore_label_emb.T
        fore_targets = F.softmax(
            (fore_img_similarity + fore_txt_similarity) / 2 * temperature, dim=-1
        )
        logits = (fore_label_emb @ fore_emb.T) / temperature
        fore_loss = (-fore_targets.T * F.log_softmax(logits.T, dim=-1)).sum(1)
        
        f2b_emb = torch.cat([fore_emb, back_emb], dim=0)
        f2b_similarity = f2b_emb @ f2b_emb.T
        f2b_target = F.softmax(f2b_similarity * temperature, dim=-1)
        back_loss = -f2b_target.T * F.log_softmax((f2b_similarity / temperature).T, dim=-1)
        back_loss[:len(fore_emb), :len(fore_emb)] = 0.0  # NOTE: remove the part that is duplicated with fore_loss
        back_loss = back_loss.sum(1)
        
        return fore_loss + back_loss
    
    @staticmethod
    def contrastive_loss_v2(fore_emb, back_emb, fore_label_emb, temperature=20.0):
        """
        fore_emb: embedding of foreground anchor boxes (N, 640)
        back_emb: embedding of background anchor boxes (batch_size * M, 640)
        fore_label_emb: text embedding of region anchor boxes cover (N, 640)

        M: num_neg_samples = 1000
        """

        fore_img_similarity = fore_emb @ fore_emb.T
        fore_txt_similarity = fore_label_emb @ fore_label_emb.T
        fore_targets = F.softmax(
            (fore_img_similarity + fore_txt_similarity) / 2 * temperature, dim=-1
        )
        logits = (fore_label_emb @ fore_emb.T) / temperature
        fore_loss = (-fore_targets.T * F.log_softmax(logits.T, dim=-1)).sum(1)
        
        f2b_emb = torch.cat([fore_emb, back_emb], dim=0)
        f2b_similarity = f2b_emb @ f2b_emb.T
        f2b_target = F.softmax(f2b_similarity * temperature, dim=-1)
        back_loss = -f2b_target.T * F.log_softmax((f2b_similarity / temperature).T, dim=-1)
        back_loss[:len(fore_emb), :len(fore_emb)] = 0.0  # NOTE: remove the part that is duplicated with fore_loss
        back_loss = back_loss.sum(1)
        
        return (fore_loss + back_loss) / fore_emb.size(0)

    def forward(self,
                preds: Tensor,
                anchors: Tensor,
                labels: Tensor
                ) -> Tuple[Tensor, Tensor, Tensor]:

        if len(preds.shape) != 3:
            raise ValueError("preds should be given in 3d tensor")

        if len(anchors.shape) != 3:
            raise ValueError("anchors should be given in 3d tensor")

        if len(labels.shape) != 3:
            raise ValueError("labels should be given in 3d tensor")

        reg_preds = preds[..., :4]
        cls_preds = preds[..., 4:5]  # NOTE: every box have a one-class classifier for distinguish bg/fg
        embed_preds = preds[..., 5:]

        target_assigns = self.anchor_assigner(labels, anchors)
        cls_losses = []
        reg_losses = []
        emb_losses = []
        embed_tups = []

        for i, assign in enumerate(target_assigns):
            fore_idx = assign['foreground'][0]
            back_idx = assign['background'][0]

            fore_label_emb = assign['foreground'][1][..., 4:]
            fore_label_bbox = assign['foreground'][1][..., :4]
            fore_label = torch.ones_like(cls_preds)

            cls_losses.append(self.focal_smax_loss(cls_preds[i], fore_idx, back_idx, fore_label, self.alpha, self.gamma, self.fore_mean))
            embed_tups.append(self.sample_fore_background_boxes(embed_preds[i], fore_idx, back_idx, fore_label_emb))
            reg_losses.append(self.smooothL1_loss(reg_preds[i], anchors, fore_idx, fore_label_bbox, self.beta, self.fore_mean))

        emb_loss = self.contrastive_loss_v2(
            torch.cat([t[0] for t in embed_tups]),
            torch.cat([t[1] for t in embed_tups]),
            torch.cat([t[2] for t in embed_tups]),
        )
        
        cls_loss = sum(cls_losses)
        reg_loss = sum(reg_losses)
        total_loss = cls_loss + self.reg_weight * reg_loss

        if self.average:
            batch = len(target_assigns)
            total_loss /= batch
            cls_loss /= batch
            reg_loss /= batch
        
        total_loss += emb_loss

        return total_loss, cls_loss, reg_loss