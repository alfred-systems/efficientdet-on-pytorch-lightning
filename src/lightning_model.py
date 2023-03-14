import datetime
import pysnooper
from collections import defaultdict
from pprint import pprint

import open_clip
from loguru import logger
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.detection.mean_ap import MeanAveragePrecision


from src.__init__ import *
from src.utils.bbox import convert_bbox, untransform_bbox
from src.dataset.metric import Evaluate_COCO
from src.model.efficientdet import EfficientDet, ConvNeXtDet, ClipDet, ClipFuseDet
from src.loss.focal_loss import FocalL1_Loss, ContrastiveL1_Loss, clip_loss
from src.utils.nms.hard_nms import Hard_NMS
from src.dataset.train_dataset import CLASS_NAME, GT_CLASS_NAME



class COCO_EfficientDet(pl.LightningModule):

    __doc__ = r"""
        This class is to manage the hyper-parameters at once, involved in training.
    
        Args:
            coeff: coefficient for EfficientDet
            pretrained_backbone: load checkpoints to the model's backbone
            ckpt_path: checkpoint path of only EfficientDet, not COCO_EfficientDet
                if you load the checkpoints of COCO_EfficientDet, use 'load_from_checkpoint'
            fore_th: foreground threshold for the loss function
            back_th: background threshold for the loss function
            alpha: alpha for focal-loss
            gamma: gamma for focal-loss
            beta: beta for smooth-L1 loss
            fore_mean: average the loss values by the number of foregrounds
            reg_weight: weight for smooth-L1 loss 
            average: average the loss values by the number of mini-batch
            iou_th: IoU threshold for Soft-NMS
            conf_th: confidence or score threshold for Soft-NMS
            gaussian: gaussian penalty for Soft-NMS
            sigma: sigma for Soft-NMS
            max_det: max detection number after Soft-NMS
            lr: learning rate  
            lr_exp_base: gamma for the exponential scheduler
            warmup_epochs: warm-up start epochs for the exponential scheduler
            val_annFile: file path of annotation for validation(instances_train2017.json) 
        
        * You can alter NMS or optimizer by modifications of lines.
    """

    def __init__(self,
                 coeff: int,
                 pretrained_backbone: bool = True,
                 ckpt_path: str = None,
                 fore_th: float = 0.5,
                 back_th: float = 0.4,
                 alpha: float = 0.25,
                 gamma: float = 1.5,
                 beta: float = 0.1,
                 fore_mean: bool = True,
                 reg_weight: float = None,
                 average: bool = True,
                 iou_th: float = 0.5,
                 max_det: Optional[int] = 400,
                 lr: float = 1e-4,
                 val_annFile: str = None,
                 background_class: bool = True,
                 freeze_backbone: bool = False,
                 ):

        super().__init__()
        self.save_hyperparameters()

        self.coeff = coeff
        self.pretrained_backbone = pretrained_backbone
        self.ckpt_path = ckpt_path
        self.fore_th = fore_th
        self.back_th = back_th
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.fore_mean = fore_mean
        self.reg_weight = reg_weight
        self.average = average
        self.iou_th = iou_th
        self.max_det = max_det
        self.lr = lr
        self.annFile = val_annFile
        self.background_class = background_class
        self.freeze_backbone = freeze_backbone

        self.model = self.configure_model()
        self.anchors = self.model.anchors
        self.loss = self.configure_loss_function()
        self.nms = self.configure_nms()

        self.val_result_dir = None
        self.test_result_dir = None
        self.plot_freq = 500
        # TODO: https://github.com/Lightning-AI/metrics/issues/1024
        self.val_map = MeanAveragePrecision(box_format="xywh", class_metrics=False)
    
    def load_finetune_checkpoint(self, path):
        m = torch.load(path)['state_dict']
        model_dict = self.state_dict()
        
        for k in m.keys():
            if k in model_dict:
                pval = m[k]
                if pval.shape == model_dict[k].shape:
                    model_dict[k] = pval.clone()
                else:
                    logger.warning(f"[load_finetune_checkpoint] Get a miss-matching parameter {k}: {pval.shape} vs {model_dict[k].shape}")
        self.load_state_dict(model_dict, strict=False)

    def configure_model(self):
        model = EfficientDet(self.coeff, 80, background_class=self.background_class, freeze_backbone=self.freeze_backbone)
        # model = ConvNeXtDet(self.coeff, 80, background_class=self.background_class, freeze_backbone=self.freeze_backbone)

        # if not self.pretrained_backbone:
        #     raise RuntimeError('not suposse to use this option')
        #     self.initialize_weight(model)
        # else:
        #     self.initialize_weight(model.fpn)
        #     self.initialize_weight(model.head)

        if self.ckpt_path:
            ckpt = torch.load(self.ckpt_path)
            assert isinstance(ckpt, OrderedDict), 'please load EfficientDet checkpoints'
            assert next(iter(ckpt)).split('.')[0] != 'model', 'please load EfficientDet checkpoints'
            model.load_state_dict(torch.load(self.ckpt_path))

        return model

    def configure_loss_function(self):
        return FocalL1_Loss(self.fore_th, self.back_th, self.alpha, self.gamma, self.beta,
                          self.fore_mean, self.reg_weight, self.average, 'cxcywh')

    def configure_nms(self):
        return Hard_NMS(self.iou_th, self.max_det, 'cxcywh')

    def configure_optimizers(self):
        param_groups = [
            {"params": self.model.backbone.parameters(), 'lr': self.lr * 0.8},
            {"params": self.model.fpn.parameters(), 'lr': self.lr},
            {"params": self.model.head.parameters(), 'lr': self.lr},
        ]
        if list(self.loss.parameters()):
            param_groups.append({"params": self.loss.parameters(), 'lr': self.lr})
        
        optimizer = AdamW(param_groups, self.lr)
        # optimizer = AdamW(self.model.backbone.parameters(), self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5,
                                      threshold=0.001, threshold_mode='abs', verbose=True)

        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": 'train_loss'}}

    @classmethod
    def initialize_weight(cls, model):
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


    def forward(self, input: torch.Tensor, detect: bool, **kwargs):
        return self.model(input, detect,  **kwargs)


    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        preds, anchors = self.model(inputs, detect=False)
        sync_labels = convert_bbox(labels, 'xywh', 'cxcywh')
        
        sync_labels = sync_labels.to(preds.device)
        anchors = anchors.to(preds.device)  # BUG: anchors is a Tensor, won't be auto move to correct device by DDP

        loss, cls_loss, reg_loss = self.loss(preds, anchors, sync_labels)
        self.log('train_loss', loss)
        self.log('train_cls_loss', cls_loss)
        self.log('train_reg_loss', reg_loss)

        return loss
    
    def plt_wandb_bbox(self, 
                    preds: torch.Tensor, 
                    images: List[np.ndarray], 
                    box_captions: List[List[str]]=None, 
                    ground_truth_boxes=None, 
                    ground_truth_labels=None,
                    ground_truth_captions: List[List[str]]=None, 
        ):
        batch_boxes = []
        
        for i in range(len(preds)):
            pred_pt = convert_bbox(preds[i], 'cxcywh', 'xywh')
            
            json_boxes = []
            pred_pt = pred_pt.cpu().numpy()
            box_pt = pred_pt[..., :4].astype(np.int32).tolist()
            for j, (box, pred) in enumerate(zip(box_pt, pred_pt)):
                score = pred[4]
                cls = int(pred[5])
                
                if score < 0.2: continue
                json_boxes.append({
                    "position" : {
                        "minX" : box[0],
                        "maxX" : box[0] + box[2],
                        "minY" : box[1],
                        "maxY" : box[1] + box[3],
                    },
                    "class_id" : cls + 1,  # NOTE: detector don't have background class, so all class ids is shifted forward by 1
                    # optionally caption each box with its class and score
                    "box_caption" : box_captions[i][j] if box_captions else CLASS_NAME[cls + 1],
                    "domain" : "pixel",
                    "scores" : { "score" : int(100 * score) }
                })
            results = {
                "predictions": {"box_data": json_boxes, "class_labels": CLASS_NAME},
            }
            
            ground_truth = None
            if ground_truth_boxes is not None and ground_truth_labels is not None:
                ground_truth = {
                    "box_data": [
                        {
                            "position" : {
                                "minX" : box[0],
                                "maxX" : box[0] + box[2],
                                "minY" : box[1],
                                "maxY" : box[1] + box[3],
                            },
                            "class_id" : cls + 1,
                            "domain" : "pixel",
                            "box_caption" : ground_truth_captions[i][j] if ground_truth_captions else GT_CLASS_NAME[cls + 1],
                        }
                        for j, (box, cls) in enumerate(zip(ground_truth_boxes[i], ground_truth_labels[i]))
                    ],
                    "class_labels": GT_CLASS_NAME,
                }            
                results["ground_truths"] = ground_truth
            
            batch_boxes.append(results)
        
        self.logger.log_image(key="validation-step-detect", images=images, step=self.global_step, boxes=batch_boxes)
    
    def update_mean_ap(self, preds, scales, pads, extra):
        metrix_pred = []
        metrix_tar = []

        for i, (scale, pad) in enumerate(zip(scales, pads)):
            pred_pt = convert_bbox(preds[i], 'cxcywh', 'xywh')
            pred_pt = untransform_bbox(pred_pt, scale, pad, 'xywh')
            
            boxes_pt = pred_pt[..., :4].to(torch.int32)
            scores = pred_pt[:, 4]
            clses = pred_pt[:, 5].to(torch.int32)
            metrix_pred.append({
                'boxes': boxes_pt,
                'scores': scores,
                'labels': clses,
            })

        for boxes, labels in zip(extra['boxes'], extra['labels']):
            boxes = [box for box in boxes if box[0] >= 0]  # NOTE: we use [-1,-1,-1,-1] box for padding
            labels = labels[labels >= 0]
            boxes = torch.stack(boxes, dim=0) if boxes else torch.zeros([0, 4]).to(device)
            metrix_tar.append({
                'boxes': boxes,
                'labels': labels,
            })

        self.val_map.update(preds=metrix_pred, target=metrix_tar)

    def validation_step(self, batch, batch_idx):
        ids, inputs, scales, pads = batch[:4]
        extra = batch[4]
        
        preds, _ = self.model(inputs, detect=True)
        device = preds.device
        batch_size = preds.size(0)
        preds = self.nms(preds)

        # logger.debug(f"validation_step NODE_RANK: {self.global_rank} {batch_idx}")
        plot =  (
            batch_idx < 3
            and hasattr(self.logger, 'log_image')
        )
        if plot:
            images = [img for img in extra['image_numpy'].cpu().numpy()]
            self.plt_wandb_bbox(preds, images)
        
        self.update_mean_ap(preds, scales, pads, extra)
        
        # for i, (scale, pad) in enumerate(zip(scales, pads)):
        #     preds[i] = convert_bbox(preds[i], 'cxcywh', 'xywh')
        #     preds[i] = untransform_bbox(preds[i], scale, pad, 'xywh')
        return ids, preds


    def test_step(self, batch, batch_idx):
        ids, inputs, scales, pads = batch[:4]
        preds, _ = self.model(inputs, detect=True)
        preds = self.nms(preds)

        for i, (scale, pad) in enumerate(zip(scales, pads)):
            preds[i] = convert_bbox(preds[i], 'cxcywh', 'xywh')
            preds[i] = untransform_bbox(preds[i], scale, pad, 'xywh')

        return ids, preds


    def validation_epoch_end(self, val_step):
        if getattr(self, 'val_map', None) is None or self.trainer.world_size == 1:
            result = OrderedDict()

            for batch in val_step:
                for id, pred in zip(*batch):
                    result[id] = pred

            if not self.val_result_dir:
                self.val_result_dir = os.path.join('result/val', datetime.datetime.now().strftime("run-%Y-%m-%d-%H-%M"))
            result_file = os.path.join(self.val_result_dir, f'epoch-{self.global_rank}-{self.current_epoch}.json')

            val_metric = Evaluate_COCO(result, result_file, self.annFile, test=False)
            # logger.debug(f"validation_epoch_end NODE_RANK: {self.global_rank}")

            self.log('AP', val_metric['AP'], sync_dist=True)
            self.log('AP50', val_metric['AP50'], sync_dist=True)
            self.log('AR', val_metric['AR'], sync_dist=True)
        else:
            for k, v in self.val_map.compute().items():
                self.log(f'val_{k}', v, sync_dist=True)
                logger.info(f'{k}: {v}')
            self.val_map.reset()

    def test_epoch_end(self, test_step):
        result = OrderedDict()

        for batch in test_step:
            for img_id, pred in zip(*batch):
                result[img_id] = pred

        if not self.test_result_dir:
            self.test_result_dir = 'result/test'
        result_file = os.path.join(self.test_result_dir, datetime.datetime.now().strftime("run-%Y-%m-%d-%H-%M.json"))

        Evaluate_COCO(result, result_file, None, test=True)



class Laion400m_EfficientDet(COCO_EfficientDet):
    """
    Re-align model's global average embedding coming out of the BiFPN with CLIP text encoder
    """

    def configure_model(self):
        model = ClipDet(
            self.coeff,
            background_class=self.background_class,
            freeze_backbone=self.freeze_backbone)

        if not self.pretrained_backbone:
            raise RuntimeError('not suposse to use this option')
            self.initialize_weight(model)
        else:
            self.initialize_weight(model.fpn)
            self.initialize_weight(model.head)

        if self.ckpt_path:
            ckpt = torch.load(self.ckpt_path)
            assert isinstance(ckpt, OrderedDict), 'please load EfficientDet checkpoints'
            assert next(iter(ckpt)).split('.')[0] != 'model', 'please load EfficientDet checkpoints'
            model.load_state_dict(torch.load(self.ckpt_path))

        return model
    
    def training_step(self, batch, batch_idx):
        inputs, txt_embed = batch
        preds, anchors, global_avgs = self.model(inputs, detect=False, global_feat=True)
        
        losses = []
        for i, layer_global_avg in enumerate(global_avgs):
            layer_loss = clip_loss(layer_global_avg, txt_embed)
            losses.append(layer_loss)
            self.log(f'gap_train_loss_{i}', layer_loss.mean())
        loss = sum(losses).mean()
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, txt_embed = batch
        preds, anchors, global_avgs = self.model(inputs, detect=False, global_feat=True)
        
        losses = []
        for layer_global_avg in global_avgs:
            losses.append(clip_loss(layer_global_avg, txt_embed))
        loss = sum(losses)
        return loss

    def validation_epoch_end(self, val_step):
        num_samples = 0
        loss_sum = 0
        for batch_loss in val_step:
            num_samples += batch_loss.size(0)
            loss_sum += torch.sum(batch_loss)
        val_loss = loss_sum / num_samples
        self.log('val_loss', val_loss, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        raise NotImplementedError("Embedding alignment is not the main task that meant to be evaled")

    def configure_loss_function(self):
        return ContrastiveL1_Loss(
            self.fore_th, self.back_th, self.alpha, self.gamma, self.beta,
            fore_mean=self.fore_mean, reg_weight=self.reg_weight, average=self.average, bbox_format='cxcywh')


class VisGenome_EfficientDet(COCO_EfficientDet):
    """
    Align anchor box embedding with CLIP text embedding, and use 
    cos-simailiarity as anchor box confidence.
    """

    def configure_model(self):
        model = ClipDet(self.coeff, 
                        background_class=self.background_class,
                        freeze_backbone=self.freeze_backbone)

        if not self.pretrained_backbone:
            raise RuntimeError('not suposse to use this option')
            self.initialize_weight(model)
        else:
            self.initialize_weight(model.fpn)
            self.initialize_weight(model.head)

        if self.ckpt_path:
            ckpt = torch.load(self.ckpt_path)
            assert isinstance(ckpt, OrderedDict), 'please load EfficientDet checkpoints'
            assert next(iter(ckpt)).split('.')[0] != 'model', 'please load EfficientDet checkpoints'
            model.load_state_dict(torch.load(self.ckpt_path))

        return model

    def configure_loss_function(self):
        return ContrastiveL1_Loss(
            self.fore_th, self.back_th, self.alpha, self.gamma, self.beta,
            fore_mean=self.fore_mean, reg_weight=self.reg_weight, average=self.average, bbox_format='cxcywh')
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        preds, anchors = self.model(inputs, detect=False)
        sync_labels = convert_bbox(labels, 'xywh', 'cxcywh')
        
        sync_labels = sync_labels.to(preds.device)
        anchors = anchors.to(preds.device)  # BUG: anchors is a Tensor, won't be auto move to correct device by DDP

        loss, cls_loss, reg_loss, emb_loss  = self.loss(preds, anchors, sync_labels)
        self.log('train_loss', loss)
        self.log('train_cls_loss', cls_loss)
        self.log('train_reg_loss', reg_loss)
        self.log('train_emb_loss', emb_loss)
        self.log('logit_scale', self.loss.logit_scale)

        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, labels, scales, pads = batch[:4]
        sync_labels = convert_bbox(labels, 'xywh', 'cxcywh')
        extra = batch[4]
        
        preds, anchors = self.model(inputs, detect=True)
        loss, cls_loss, _, emb_loss  = self.loss(preds, anchors, sync_labels)

        device = preds.device
        batch_size = preds.size(0)
        
        gt_boxes = labels[..., :4]
        gt_embed = labels[..., 4:]
        
        pred_boxes = preds[..., :4]
        pred_cls = preds[..., 4:5]
        pred_embed = preds[..., 5:]
        
        postitive, scores, captions = self.query_anchors(pred_embed, gt_embed)
        pred_boxes = [b[p] for b, p in zip(pred_boxes, postitive)]
        pred_cls = [c[p] for c, p in zip(pred_cls, postitive)]
        dummy_class_id = [torch.zeros([len(c)], device=device).long() for c in pred_cls]
        
        # nms_preds, nms_captions = self.nms(torch.cat([pred_boxes, scores], dim=-1), pred_meta=captions)
        nms_preds, nms_captions = self.nms.nms(
            pred_boxes, scores, dummy_class_id, 
            num_class=1, pred_meta=captions, softmax=False
        )
        region_captions: List[str] = extra['phrases']
        region_captions = [join_cap.split('&&') for join_cap in region_captions]
        nms_captions = [
            [region_captions[i][j] for j in caps]
            for i, caps in enumerate(nms_captions)
        ]
        tmp = {
            'boxes': gt_boxes, # with -1 padding
            'labels': (gt_boxes.sum(-1) >= 0).long() - 1,
        }

        self.update_mean_ap(
            nms_preds, 
            torch.ones_like(scales), 
            torch.zeros_like(pads), 
            tmp, 
        )

        if batch_size * batch_idx < 20 and hasattr(self.logger, 'log_image'):
            gt_boxes = [
                boxes[boxes.max(dim=-1).values >= 0].cpu().to(torch.int32).numpy().tolist() 
                for boxes in gt_boxes
            ]
            gt_labels = [[0] * len(gt_boxes[i]) for i in range(len(gt_boxes))]
            
            np_imgs = [img.cpu().numpy() for img in extra['image_numpy']]
            self.plt_wandb_bbox(
                nms_preds, np_imgs, box_captions=nms_captions, 
                ground_truth_boxes=gt_boxes, ground_truth_labels=gt_labels
            )
        return {
            'loss': loss,
            'cls_loss': cls_loss,
            'emb_loss': emb_loss,
        }
    
    def validation_epoch_end(self, val_step):
        for k, v in self.val_map.compute().items():
            self.log(f'val_{k}', v, sync_dist=True)
            logger.info(f'{k}: {v}')
        self.val_map.reset()

        accume_losses = defaultdict(lambda: 0.0)
        for step_result in val_step:
            for k, v in step_result.items():
                accume_losses[k] += v
        steps = len(val_step)
        for k, v in accume_losses.items():
            self.log(f'val_{k}', v / steps, sync_dist=True)
            logger.info(f'val_{k}: {v / steps}')

    def query_anchors(self, pred: torch.Tensor, queries: torch.Tensor, similarity_th=.3):
        pred = F.normalize(pred, dim=-1)
        queries = F.normalize(queries, dim=-1)

        positives = []
        pos_similarity = []
        match_captions = []
        
        for i, (pred_per_img, que_per_img) in enumerate(zip(pred, queries)):
            # breakpoint()
            # pred_per_img = pred_per_img[torch.abs(pred_per_img.sum(-1)) > 1e-6]  
            que_per_img = que_per_img[torch.abs(que_per_img.sum(-1)) > 1e-6]  # remove padding
            
            similarity = pred_per_img @ que_per_img.T
            max_score, indies = similarity.max(dim=-1)
            positives.append(max_score > similarity_th)
            pos_similarity.append(max_score[max_score > similarity_th])
            # torch.topk(similarity, 5, dim=0).values.permute(1,0)
            
            captions = indies[max_score > similarity_th]
            match_captions.append(captions)
        
        return (
            positives,
            pos_similarity,
            match_captions,
        )

    
class VisGenome_FuseDet(COCO_EfficientDet):
    """
    
    """
    
    TEXT_MODEL = "convnext_large_d"

    @property
    def text_encoder(self):
        if not hasattr(self, '_text_encoder'):
            model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
                self.TEXT_MODEL, pretrained='laion2b_s26b_b102k_augreg'
            )
            self._text_encoder = model.to(self.device).to(self.dtype)
        return self._text_encoder
    
    @property
    def tokenizer(self):
        if not hasattr(self, '_tokenizer'):
            self._tokenizer = open_clip.get_tokenizer(self.TEXT_MODEL)
        return self._tokenizer

    def configure_model(self):
        model = ClipFuseDet(self.coeff, 
                        background_class=self.background_class,
                        freeze_backbone=self.freeze_backbone)
        return model

    def forward(self, input: torch.Tensor, que_emb: torch.Tensor, detect: bool, **kwargs):
        return self.model(input, que_emb, detect=detect,  **kwargs)
    
    def training_step(self, batch, batch_idx):
        inputs, que_emb, labels = batch
        
        if isinstance(que_emb, (list, tuple)) and isinstance(que_emb[0], str):
            with torch.no_grad():
                input_ids = self.tokenizer(que_emb).to(self.device)
                que_emb = self.text_encoder.encode_text(input_ids)
        
        preds, anchors = self.model(inputs, que_emb, detect=False)
        sync_labels = convert_bbox(labels, 'xywh', 'cxcywh')
        
        sync_labels = sync_labels.to(preds.device)
        anchors = anchors.to(preds.device)  # BUG: anchors is a Tensor, won't be auto move to correct device by DDP

        loss, cls_loss, reg_loss  = self.loss(preds, anchors, sync_labels)
        self.log('train_loss', loss)
        self.log('train_cls_loss', cls_loss)
        self.log('train_reg_loss', reg_loss)

        return loss
    
    # @pysnooper.snoop()
    def validation_step(self, batch, batch_idx):
        inputs, que_emb, labels, scales, pads, extra = batch

        if isinstance(que_emb, (list, tuple)) and isinstance(que_emb[0], str):
            input_ids = self.tokenizer(que_emb).to(self.device)
            que_emb = self.text_encoder.encode_text(input_ids)
        
        preds, _ = self.model(inputs, que_emb, detect=True)
        device = preds.device
        batch_size = preds.size(0)

        ks = torch.topk(preds[..., 4], k=max(500, self.max_det), dim=-1).indices
        preds = [p[k] for p, k in zip(preds, ks)]
        preds = torch.stack(preds, dim=0)

        nms_preds = self.nms(preds)
        region_captions: List[str] = extra['phrases']
        region_captions = [join_cap.split('&&') for join_cap in region_captions]
        nms_captions = [
            caps * len(nms_preds[i])
            for i, caps in enumerate(region_captions)
        ]

        gt_boxes = labels[..., :4]
        tmp = {
            'boxes': gt_boxes, # with -1 padding
            'labels': (gt_boxes.sum(-1) >= 0).long() - 1,
        }
        self.update_mean_ap(
            nms_preds, 
            torch.ones_like(scales), 
            torch.zeros_like(pads), 
            tmp, 
        )

        if batch_size * batch_idx < 20 and hasattr(self.logger, 'log_image'):
            gt_boxes = [
                boxes[boxes.sum(dim=-1) >= 0].cpu().to(torch.int32).numpy().tolist() 
                for boxes in gt_boxes
            ]
            gt_labels = [[0] * len(gt_boxes[i]) for i in range(len(gt_boxes))]
            
            np_imgs = [img.cpu().numpy() for img in extra['image_numpy']]
            self.plt_wandb_bbox(
                nms_preds, np_imgs, 
                box_captions=nms_captions, 
                ground_truth_boxes=gt_boxes, 
                ground_truth_labels=gt_labels,
                ground_truth_captions=region_captions,
            )
    
    def validation_epoch_end(self, val_step):
        for k, v in self.val_map.compute().items():
            self.log(f'val_{k}', v, sync_dist=True)
            logger.info(f'{k}: {v}')
        self.val_map.reset()
    
    def inference_step(self, images, query_input_ids, text_embed=None, th=0.2):
        if text_embed is None:
            text_embed = self.text_encoder.encode_text(query_input_ids)
        preds, _ = self.model(images, text_embed, detect=True)
        
        # preds = preds[preds[..., 4] >= th]
        ks = torch.topk(preds[..., 4], k=max(500, self.max_det), dim=-1).indices
        preds = [p[k] for p, k in zip(preds, ks)]
        preds = torch.stack(preds, dim=0)
        print('max: ', preds[..., -1].max())
        
        preds = convert_bbox(preds, 'cxcywh', 'xyxy')
        nms_preds = torch.stack(self.nms(preds))
        boxes = nms_preds[..., :4]
        scores = nms_preds[..., 4]
        return boxes, scores
