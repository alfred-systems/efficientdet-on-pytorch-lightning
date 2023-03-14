import os
import json
import glob
import random
import copy

from src.utils.bbox import batch_iou
from src.dataset.utils import *
from src.dataset.bbox_augmentor import Bbox_Augmentor
from pycocotools.coco import COCO
from torchvision.datasets.vision import VisionDataset
from tqdm import tqdm
from loguru import logger
from collections import defaultdict


CLASS_TABLE = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 
    11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 
    21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 
    31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 
    41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 
    51: 45, 52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53, 
    60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 
    70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 
    80: 70, 81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 
    90: 79
}
INV_CLASS_TABLE = {v: k for k, v in CLASS_TABLE.items()}
CLASS_NAME = {
    0: '__background__',
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    12: 'stop sign',
    13: 'parking meter',
    14: 'bench',
    15: 'bird',
    16: 'cat',
    17: 'dog',
    18: 'horse',
    19: 'sheep',
    20: 'cow',
    21: 'elephant',
    22: 'bear',
    23: 'zebra',
    24: 'giraffe',
    25: 'backpack',
    26: 'umbrella',
    27: 'handbag',
    28: 'tie',
    29: 'suitcase',
    30: 'frisbee',
    31: 'skis',
    32: 'snowboard',
    33: 'sports ball',
    34: 'kite',
    35: 'baseball bat',
    36: 'baseball glove',
    37: 'skateboard',
    38: 'surfboard',
    39: 'tennis racket',
    40: 'bottle',
    41: 'wine glass',
    42: 'cup',
    43: 'fork',
    44: 'knife',
    45: 'spoon',
    46: 'bowl',
    47: 'banana',
    48: 'apple',
    49: 'sandwich',
    50: 'orange',
    51: 'broccoli',
    52: 'carrot',
    53: 'hot dog',
    54: 'pizza',
    55: 'donut',
    56: 'cake',
    57: 'chair',
    58: 'couch',
    59: 'potted plant',
    60: 'bed',
    61: 'dining table',
    62: 'toilet',
    63: 'tv',
    64: 'laptop',
    65: 'mouse',
    66: 'remote',
    67: 'keyboard',
    68: 'cell phone',
    69: 'microwave',
    70: 'oven',
    71: 'toaster',
    72: 'sink',
    73: 'refrigerator',
    74: 'book',
    75: 'clock',
    76: 'vase',
    77: 'scissors',
    78: 'teddy bear',
    79: 'hair drier',
    80: 'toothbrush',
}
GT_CLASS_NAME = {k: f'gt_{v}' for k, v in CLASS_NAME.items()}

class COCO_Detection(VisionDataset):

    num_classes = 80
    coco_cat = (i for i in range(1, 91))
    missing_cat = (12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91)
    collect_fn = make_mini_batch

    def __init__(self,
                 root: str,
                 annFile: str='',
                 bbox_augmentor: Optional[Bbox_Augmentor]=None,
                 background_class=True,
                 **kwargs):

        super().__init__(root)

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.num_classes += int(background_class)

        self.augmentor = bbox_augmentor
        if self.augmentor:
            assert self.augmentor.format == 'coco', \
                'the bounding box format must be coco, (x_min, y_min, width, height)'
            assert self.augmentor.ToTensor is True, \
                'the image should be returned as a tensor'

        self.cat_table = category_filter(self.coco_cat, self.missing_cat)

    def __len__(self):
        return len(self.ids)
    
    @staticmethod
    def make_mini_batch(
        sample
    ) -> Tuple[Tensor, Tensor]:

        images, labels, target_nums = [], [], []

        zero_labels = []
        max_target_num = 0

        for image, label in sample:
            images.append(image)
            labels.append(label)
            target_num = label.size(0)
            target_nums.append(target_num)
            max_target_num = max(max_target_num, target_num)

        for label in labels:
            target_num = label.size(0)
            zero_fill = torch.zeros((max_target_num - target_num, label.size(1)), dtype=label.dtype, device=label.device)
            zero_label = torch.cat((label, zero_fill), dim=0)
            zero_labels.append(zero_label)

        images = torch.stack(images)
        labels = torch.stack(zero_labels)

        return images, labels

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        coco = self.coco
        img_id = self.ids[index]

        img_path = coco.loadImgs(img_id)[0]['file_name']
        image = cv2.imread(os.path.join(self.root, img_path))
        target = coco.loadAnns(coco.getAnnIds(imgIds=img_id))

        bboxes, category_ids = [], []

        for i, t in enumerate(target):
            bbox = t['bbox']
            if 0.0 in bbox[2:]:
                bbox[2] += 1e-7
                bbox[3] += 1e-7

            bboxes.append(bbox)
            category_ids.append(t['category_id'])


        if self.augmentor:
            transform = self.augmentor(image, bboxes, category_ids)
            image, bboxes, category_ids = transform.values()
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image)

        for i, cat_id in enumerate(category_ids):
            new_id = self.cat_table[cat_id]
            category_ids[i] = make_one_hot(self.num_classes, new_id)

        if category_ids:
            category_ids = torch.stack(category_ids)
        else:
            category_ids = torch.tensor([], dtype=torch.int8)

        bboxes = torch.from_numpy(np.asarray(bboxes))
        label = torch.cat((bboxes, category_ids), dim=1)

        if len(label) == 0:
            label = torch.zeros((0, self.num_classes + 4), dtype=torch.int8)

        return image, label


class COCOFuseDet(COCO_Detection):
    """
    COCO dataset loader that use caption to indicate box label instead of class id.
    """

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        coco = self.coco
        img_id = self.ids[index]

        img_path = coco.loadImgs(img_id)[0]['file_name']
        image = np_image = cv2.imread(os.path.join(self.root, img_path))
        target = coco.loadAnns(coco.getAnnIds(imgIds=img_id))

        bboxes, category_ids = [], []

        for i, t in enumerate(target):
            bbox = t['bbox']
            if 0.0 in bbox[2:]:
                bbox[2] += 1e-7
                bbox[3] += 1e-7

            bboxes.append(bbox)
            category_ids.append(t['category_id'])

        one_cls = random.choice(category_ids)
        assign = [c == one_cls for c in category_ids]
        bboxes = [b for b, a in zip(bboxes, assign) if a == 1]
        category_ids = [one_cls] * len(bboxes)
        

        if self.augmentor:
            transform = self.augmentor(image, bboxes, category_ids)
            image, bboxes, category_ids = transform.values()
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image)

        cls_name = CLASS_NAME[self.cat_table[one_cls]]
        phrases = [f"all {cls_name} in the image"] * len(bboxes)
        category_ids = torch.ones([len(bboxes)], dtype=torch.float)

        bboxes = torch.from_numpy(np.asarray(bboxes))
        label = torch.cat((bboxes, category_ids), dim=1)

        if len(label) == 0:
            label = torch.zeros((0, self.num_classes + 4), dtype=torch.int8)

        if self.split == 'train':
            return image, phrases, label
        else:
            h, w, c = np_image.shape
            c, _h, _w = image.shape

            scale = _h / h
            diff = np.abs(h - w)
            p1 = diff // 2
            p2 = diff - diff // 2
            pad = (0, p1, 0, p2) if w >= h else (p1, 0, p2, 0)
            pad = torch.tensor(pad)
            # print(tmp, bboxes, scale, pad)
            extra = {
                'phrases': '&&'.join(phrases),
                'image_numpy': transform['image_numpy'],
            }
            return image, phrases, label, scale, pad, extra


class VisualGenome(VisionDataset):
    
    MODEL = "convnext_large_d"
    EMBED_SIZE = 768
    collect_fn = None
    max_det = 300  # NOTE: VG have max 267/ min 3/ avg 50 regions per image

    def __init__(self, img_dir: str, 
                 annFile: str = '', 
                 bbox_augmentor: Optional[Bbox_Augmentor]=None,
                 split='train',
                 offline_embed=True,
                 **kwargs):
        assert os.path.exists(img_dir)
        
        self.img_dir = img_dir
        with open(annFile, mode='r') as f:
            self.region_anno = json.load(f)
        self.augmentor = bbox_augmentor
        
        self.cache_file = os.path.join(img_dir, "embed_cache.fp16.pth")
        self.cache_dir = os.path.join(img_dir, "region_embeds")
        
        n = len(self.region_anno)
        if split == 'train':
            self.region_anno_subset = self.region_anno[:int(n * 0.95)]
        else:
            self.region_anno_subset = self.region_anno[int(n * 0.95):]
            self.augmentor.with_np_image = True
        self.split = split
        self.offline_embed = offline_embed
    
    @property
    def phrase_embed(self):
        """
        HACK: lazy loading to avoid multiprocess pickle on after __init__
        """
        if not hasattr(self, '_phrase_embed'):
            if not os.path.exists(self.cache_file):
                logger.info(f"Generate VisualGenome region description text embedding: {self.cache_file}")
                self._split_embed_table_by_img(
                    self._create_region_embed(self.cache_file),
                    self.cache_dir
                )
            logger.info(f"Loading VisualGenome region description embedding cache: {self.cache_file}")
            self._phrase_embed: Dict[int, Tensor] = torch.load(self.cache_file)
            # self._phrase_embed: Dict[int, Tensor] = defaultdict(lambda: torch.ones([640]).float())
        return self._phrase_embed
    
    def _create_region_embed(self, cache_file: str):
        import open_clip
        
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(self.MODEL, pretrained='laion2b_s26b_b102k_augreg')
        tokenizer = open_clip.get_tokenizer(self.MODEL)
        model = model.to('cuda')
        
        embed_table = {}
        with torch.no_grad():
            m = len(self.region_anno)
            for i in tqdm(range(m)):
                for region in self.region_anno[i]['regions']:
                    reg_id = region['region_id']
                    phrase = region['phrase']
                    input_ids = tokenizer(phrase)
                    embed = model.encode_text(input_ids.to('cuda'))
                    assert embed.ndim == 2 and embed.size(0) == 1
                    embed_table[reg_id] = embed.cpu()[0]
                print(f"_create_region_embed {i}/{m}")
        
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        torch.save(embed_table, cache_file)
        return embed_table
    
    def _split_embed_table_by_img(self, phrase_embed: Dict[int, Tensor], cache_dir: str):
        os.makedirs(cache_dir, exist_ok=True)
        m = len(self.region_anno)
        for i in tqdm(range(m)):
            img_id = self.region_anno[i]['id']
            table = {}
            for region in self.region_anno[i]['regions']:
                reg_id = region['region_id']
                table[reg_id] = phrase_embed[reg_id]
            torch.save(table, os.path.join(cache_dir, f"{img_id}.pth"))
    
    def __len__(self):
        return len(self.region_anno_subset)

    def subsample(self, boxes, th=0.2):
        """
        There are many overlaped bbox for the same object in visual genome, used to
        desction different attribute/state of the same object.
        So we need to avoid using multiple overlaped boxes at once in order to make 
        contrastive loss work.
        """
        assert th <= 0.5
        pt_boxes = torch.tensor(boxes).unsqueeze(dim=0)
        iou_mtx = batch_iou(pt_boxes, pt_boxes, format='xywh')[0]
        assign = [-1] * len(boxes)  # -1 for to be assign, 0 for discarded, 1 for assigned as label
        
        for i, iou_1toN in enumerate(iou_mtx):
            if assign[i] > -1:
                continue
            
            overlap = []  # include "i" itself
            close = []
            abort = False
            
            for j, iou in enumerate(iou_1toN):
                if iou >= th and assign[j] > -1:
                    abort = True
                    break
                if iou >= 0.5:
                    assign[j] = 0
                    overlap.append(j)
                elif 0.5 > iou >= th:
                    assign[j] = 0
                    close.append(j)
            
            if not abort:
                assign[random.choice(overlap)] = 1
                if close:
                    assign[random.choice(close)] = 1
        return assign

    def keep_similiar_captions(self, src_emb, all_emb):
        if isinstance(all_emb, list):
            all_emb = torch.stack(all_emb)
        all_emb = F.normalize(all_emb, dim=1)
        src_emb = F.normalize(src_emb, dim=0)

        sim = all_emb @ src_emb.unsqueeze(dim=0).T
        alikes = sim > 0.93
        return alikes
    
    def __getitem__(self, index: int):
        meta = self.region_anno_subset[index]
        img_id: int = meta['id']
        np_image = cv2.imread(os.path.join(self.img_dir, f"{img_id}.jpg"))
        embed_table = torch.load(os.path.join(self.cache_dir, f"{img_id}.pth"))

        ih, iw, ic = np_image.shape
        bboxes = []
        phr_embed = []
        phrases = []
        for region in meta['regions']:
            x: int = min(max(0, region['x']), iw)  # xmin
            y: int = min(max(0, region['y']), ih)  # ymin
            w: int = min(iw - x, region['width'])
            h: int = min(ih - y, region['height'])
            phrase: str = region['phrase']
            reg_id: int = region['region_id']

            if w <= 5 or h <= 5:
                # discard extremly small/incorrectly labeled bbox
                continue

            phrases.append(phrase)
            phr_embed.append(embed_table[reg_id].to(torch.float32))
            bboxes.append([x, y, w, h])
        
        assign = self.subsample(copy.deepcopy(bboxes))
        bboxes = [b for b, a in zip(bboxes, assign) if a == 1]
        phr_embed = [b for b, a in zip(phr_embed, assign) if a == 1]
        phrases = [b for b, a in zip(phrases, assign) if a == 1]

        if self.augmentor:
            category_ids = [0] * len(bboxes)  # TODO: decide to make use of category label or not
            # HACK: we use category_ids's slot to place bbox embedding, so we only keep embeds that still inside the image after augmentation
            transform = self.augmentor(np_image, bboxes, phr_embed)  
            image, bboxes, phr_embed = transform['image'], transform['bboxes'], transform['category_ids']
        else:
            image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image)
        
        for _ in range(len(bboxes), self.max_det):
            bboxes.append([-1, -1, -1, -1])
            phr_embed.append(torch.zeros([self.EMBED_SIZE], dtype=torch.float32))
            # phr_embed.append(torch.zeros_like(phr_embed[0]))
        
        phr_embed = torch.stack(phr_embed)  # (n, 640)
        bboxes = torch.tensor(bboxes, dtype=torch.float32)  # (n, 4)
        labels = torch.cat([bboxes, phr_embed], dim=-1)

        if self.split == 'train':
            return image, labels
        else:
            h, w, c = np_image.shape
            _h, _w, _ = image.shape

            scale = _h / h
            diff = np.abs(h - w)
            p1 = diff // 2
            p2 = diff - diff // 2
            pad = (0, p1, 0, p2) if w >= h else (p1, 0, p2, 0)
            pad = torch.tensor(pad)
            extra = {
                'phrases': '&&'.join(phrases),
                'image_numpy': transform['image_numpy'],
            }
            return image, labels, scale, pad, extra


class VisualGenomeFuseDet(VisualGenome):

    max_det = 300
    find_all_objects = b'\xe8R(\xbe\x10\xa4M=\xa9B\\>\xab2\xda>\xe6%y\xbf*\n\xa6\xbe\xb2I\xb2\xbe0\x164\xbf\x0c\xd4\x19?\xf6\x1c\xc4\xbe\xf6\xfe\xaa\xbe%\x08\x0e\xbf\x86\xf4\x07>h$\xd2=\xb4\xc7P=o\xd1V>\x08\xc7\xa2\xbe\xd9eM?\x9d}|?\xd0!w>3\xec[\xbf:\xd3|\xc0\xce\x18\x0b=\xaa)Q\xbe\xe8\x9b\xa7>\xd7\xea ?N\xaf?\xbf\xa8lU>\xde\x1b\xfb\xbe\x1a \xb3\xbe\x03\xf12\xbf<t\xda>>]\xaa>\\\xb8\x17\xbc\xd0\x06O\xbe\x16/\xd2\xbe\xcf\xed\xf9>G\x05\xca\xbe\x9e\xc9\x08\xbf6hx\xbc\xd5\xe4\x17?*\x17\x82\xbe\x8e\xd1k\xbe\x90|5\xbe\xfe2&>\xd6\xf4>?Ehs>`\x1c\x07\xbf!Y&\xbfBT\xa8\xbd\xacU\x89>\x9c\x06$>\xd5\x18\xd1>\x18\xcb\xce=z&\xf9\xbe_\xdf\x8f\xbfBe\xbc=h\xb9n\xbe\xba\xdc\x97\xbc\xa9\xd9\xe0>\xfa+D\xbe\tK\xd0\xbe\x96\x92i\xbd\xbe\xf7&\xbe?h1\xbf}\x01\x97\xbe\xd1NS>Vm\x12>d\x9bX\xbf\xec\x93!?\xcaK\x00\xbfFN\x7f>}X8\xbf.\xaa\xf8\xbd\x89\xafA?6aJ?\x85\xceN=\xe8\xc6\x96=\xcd*/? \x07\xc0=*\x054\xbe\xf2K\x86\xbe",\x1c\xbf\xed\xef\xc9\xbe\xd1\xef\xdf=\xc4,T\xbf\x8a\xfc\x1d\xbf\xd8\x01#\xbf\x9a\x9e0=\tU\xba\xbeg\xaf\x02=\xca0\xb7=f\xfbX\xbe3\x96\x8b\xbe\x17\xde\x0b\xbfD\x11\xbc\xbd\x1a\xea+\xbf\x9b\x1dX\xbe\x11\xb2(\xbd(S\xc7\xbd\xeem\x98?\xf8\xa8w\xbeR\xc4\xc1=\xbd\xfc*?\x91\xdb\xc6\xbeNX"?\xae\x82\xf1\xbe@\x8a*\xbe\x12\x11(?\xbf\xacZ?\x8c2\x03>R\x06\x02\xbf\xc7\x03$>V\xb0\x8d=Q\xba\xca\xbe\x82\x04(\xbe\xc0`1;S\xaa"?6\xc2\xe5>\xd1\xb6\xf7\xbeX\xa2I\xbf\x8b\xbf\xbd\xbeF|\x16?\xd6P\xef>\x14]\x06\xc02\x9a\xfc\xbe\xe1\x9a\xe9>\x8e\x03\n\xbfAF\xb3>~\xd7g>-\x018>Cu\x19>\x1e\xa4\xb5>\x958\x8e\xbe\xfc\xa6/\xbe\xec\x0e\x02=\xf4)\xe5\xbe\x97\xd91\xbf\xa7\xbc\xd2>~\x06\xdd\xbe|Ht\xbfVy\x88\xbf\xfb\x7f\xa1>\xf9\xfa\x07>\x86\x06\xa4>\xf0\x97\xbe:j=\xd7=\x86U)<LJ\xb9\xbe\x85y\x8b?\xcaO\xcc\xbe\x81\xcd9?\xa3\xd3N\xbf\xad\x85\xa0>\xe6\x89P\xbe\x0b_\xe9\xbd\x7fL\x1a>\x8ak\x9c\xbc\xa1\xb5l?\xd2\xd2\xbd\xbdj\xedU\xbe\x075H>$\x84\xf0>\xcc\xaa\x16?Vu\x97\xbf\x04\xa5\x16>\xd0\xf7\xb6=t\xb6\x90\xbeN\xe1\xab=\xc4\x1f:>\xf9\xc0b\xbf\xd9\xf3!\xbf\x9d\xc5]>O\x0b\xb1<\x89\xd4r\xbeFXF\xbf&\x16s>\xe7\x9c\xa1>\xbe\xc5\xd2\xbe%\xd0>\xbd.,\x98\xbd\xd0\xfb\xe3>\x83\x96\x9a\xbd\xa5j\x9a?\x0e\x9d\xa3<\x99f\xec?\xac\xeb\xbe\xbe3v"\xbf@\xb7\xab>>X4\xbe\x90\xc9\xfc\xbc{{\x86>m#j>t\xe8\x9b\xbf\x98\xa0\xbc>\x92]\xb8=\x91\x1e@?\x1f\x06\x85\xbc-\xd3\r>")\x10?D\xd4]\xbe\n:\xf8=#<\x07\xbe~\x8cn\xbd\xfa\xf2\x03\xbf.\xc3\xdf\xbey\xa9!\xbe\xb6f\x08\xbf\xa5`y\xbeSn>\xbd|\x158\xbf`\x1b\x82>\xea*P>\x1e\x942=\x07\xb7\x8c>\xf1\xf4.\xbfeOW>\xde]$?g\xba\xeb>\x15\xdf\'>\xc2\xce\x03\xbfc\x0eH\xbf,w\xad>\xa0/\xac\xbde\xc1\x10\xbf\xf5s\x08?\xb1\x8a\xfb>ZB)>\xe0\xcfO?Z1O?`\x03V\xbb\x83\x9e\xf2>\x891\xc0>\xd0\x90,\xbfwC\xb4?L\xb4\x83\xbe^\xe3\xc9>\x8fD\x8d>\xd6\x1ff\xbe\xcaOn\xbe\xfb\xec\xdc>O\x93#\xbe\xc1\xfc\x90\xbe\xe5\xdax> uT>\tY\xa4>\x94_+\xbe\xba\xdcW\xbd\x12\x81\x0c>\x9c\x80\x12?\x1d\x00\xbf\xbe8\x13+\xbe\xf4\x9a\xce>\xb1\xb9\xaf>)\xdf-><\xdc\x06=\xb5\xfe]?\x89h\x83\xbd\xa3\x15\x9b\xbe\xcc.\xb9\xbe\xec[s\xbd\xca\x93\x08?\xb0MS\xbd`}F>\xf8\xc7v<\x8e\x19;\xbf\xd6"X>\xbc\xb1t>\xe2T\xfd\xbe\xe9\x8f\xc5>\x85-\x83>\xbd\xbb\x05>K\x98\x1b?\xd3\xb0\xeb\xbe\x1b\x0e\xcc>S\x80\xe3<\xf2\xfaL><\xb9\xd9<\x9c\xbf\x18\xbe+[\x1b\xbe\xca\xc6\xce>\xdao\xbc\xbe%\xe7\x86>N\xcc\xe5\xbd\n\xc3\xf1\xbc\x06$\x0c>t\xd9F?\xde\xeb\xfd\xbe\xda\x10\x15\xbeDe\xf8\xbd3c\x84\xbf\xb9\x1d\x8d>\xb1\x85\x13\xbdV3F\xbd\xfd\x9b\\\xbd\xc1!\xc8\xbd\xfdZ-\xbd\x96\xc9\x82<\xc0%\xe6\xbe\xfe\x05e=\xa6+\x87>\xc5\xc9\x83>\x88R_?\xf10\xcf>\x02x\xde>\xbb9\x19\xbf\xf8\xd9\x05\xbfx\xad9?Q/\x92?\xec\xe8\x1c<\x80\x83\x0b\xbaN\xe4\xe3>\x10\xd8\x12\xbe\xae\xa4\xf4=%U6?\x86\xfd\xf8>h\x1b\x94=\xc7\x81N\xbd\xbd/\x85\xbe\xbcV\xda>\xf2\xe6\xd4@N\xa63\xbc\xc6UU?K\xc8F\xbd\x16o\xf5\xbe_\xec@>w\xb1\x89=\x12yS?,c\xf0\xbc[\xda$?n\xc0-=\x91\x17\x8f\xbe\x98\x16%\xbf\x84\xd9\xa6\xbeT\xc8\t?h\x7f\xef\xbd\x9d?\x86\xbdL\x85\xab>\xa6\x83\xc8\xbe\xea\xcc\xe6\xbdz$\xa5>89\xc0>\x91\xf1D>\xb7\xbaX>\x8brY>\xf8\x12\x1e?\xe6j\x04\xbf\xeb\x85E?(\\5\xbf\x13$~?\xd2\xc9)?,\x18\r?I\x11\xad>\xed\n#\xbf\xe2\xf2K>Z\xba\x00?\x9e\x1d\xd0\xbd`\x14\xf5>w+3>X\xbf\xa4?\x1d\x1d\x8f\xbel\xa9\xc4\xbd\x0f\x05D\xbf\x0c\xb9\x8e\xbe\x14c\x8f\xbcr\xce\xac\xbf\x8f_h?\x83t\x83\xbfi#u\xbeW:c\xbf\x7f\xc1\xee>\x17\x088\xbf5\xcf,>\x04\x1a0<\xcb\xea\xd3>\xaa\x965\xbe\xc3\xab\xf6\xbefl\xc0>!\x94\xb9\xbf\xe4W\xeb>\x8c\x0b>?\xb5\x1cm\xbeK|\x13>xx\x1c\xbf\xd6!\xe5\xbe\xd3m\xf0>\xf8\\\xbb>\xf6\xdf\x0f\xbfm\xdf[?l\x02(>\xb49i\xbfy\xaa\x08\xbf\xc3\xbaR\xbe\xa2O\xe0\xbe\xa1\xc7\x86\xbe\xa1^\x11=\x95\xa1\xf8\xbe%E]\xbf\xff\xc1D=\x0b\xcc6=\x90\x00\xa5\xbe\xcb\x8f\x06\xbf]E\x14>m\n\xe7>;\x83\xc4>\x86|\x03=\xed\xb3\x14\xbfT\xc9?=D\\\xa4>a\x96\xc1>\x1bb\xd6\xbe@\xea\xe6\xbe\xff\x821?\xf5\xbc\x0f>\xe4U\x85\xbc\xae\\\xff>%U\xb8\xbeV\xd6\x15>/9Z?\xb7?\xee\xbe\xbf)*\xbe\xaa\xb9\xa3>\x80\xd0\x01>\xde?u?\xeft\xb5>k\x81i>\xbf\xdb\xb4>\xc4\x95\xfb\xbe\x82\xd4\xfb>#\xe7h>\xa01@>\xf3(u\xbf\x06\xd3}>\x8c\xb9d\xbf\x8e\x02h\xbd\xe0+\x9c>\xde\x13s>pl8>\xdfe\xc7>6\x11/?y\xa1Q\xbe\xa6\x8f\x06?\x17M\x9d?t\x10&>\xbd,\x8d>L[?\xbf\x83\xd3\x17\xbf\x9e\x80X\xbeD\x8d\x01?\xdcU\xee\xbeL{Z\xbf\x18\x1a\xcd\xbdf_\xb8>\xc2\x9f\x01=\xfa\xda%\xbfw\xef\x14\xbf/\xce\x13\xbf.\x03\x1e>\xef"\xf6\xbe\xcb\xaf\xba>8\x9b\xca>\x19\xc7N\xbf\x8e\xd7\x1d>*\xf8d\xbf\x1d\x9b\x0c\xbf>\x04(=_M\x8a\xbc\xde\x1f\xab\xbf{&\xb0\xbe\xbb \xd9\xbe_\xdd\xfe\xbd\x96\x8b\x91\xbf\n7\xfd\xbe\x92\x04\x9c\xbe\xe2i\x85\xbe\x9a\xe1\xec\xbe\xfeV\xaf>`?.\xbe\xd7\x0e\\>\xd1\xfb{\xbe\xa1\rh\xbe7h\xca\xbd\xfb}\x07@\xc7K\x11\xbc\x16\x1a\x08\xc1@E\xc5>\x90\x1f\xbb=%\xf7\x06=\x1b\xc4\x85?\xe5w\xf6>\xbc\xfd@\xbfZi\x11?\xe8\xee\xc8>\xea9\xe4?\x14\x85\xa9\xbeMI\x0c?\xa5T\x81>\x96L\x9c?v/Z\xbd\xad\xc9\xfe>\xa3\xde\xd3>\x08\xdb\xb2\xbcs\xd2\xab>\x93\xd0\n?\xa7t\xfb\xbe?\xa7\x12@\xa8\xb6F\xbf\xac/\xf7\xbc\xf1Jd?\x8d\x19\xd3=\xae\xd5\x1b\xbf\xa1\x8c\xa6\xbe\x8c\xb2G\xbf\xea\xb4\xad\xbdD\xf7\x02?.1\xd8>j^\x8b\xbcKO\xd7\xben\xa8>\xbe\xb59\xac\xbd\x8c\xacI?\xc4\xb7\x06>\x17\x08\xd1>m\x19\x8e\xbem\x02O\xbd\x86\x8d\xb8>z\xa0z\xbf2\xc9\xf6>\xeaT\xb1=<p\x9e\xbe\xa1$\x16\xbf\xcaiY\xbe\x1a\xda\x8e>\x86\xcc\xfa<W\xcd=\xbf\xcc\x01C\xbe\x8f\xaf\r>(\xa6\x17\xbd\xb4\xab\xe8\xbe\x80\x04\xe7\xb8\xd9\xb3\x01\xbf\xdb\x9f;\xbf\xee\xb5\xd8\xbcw\xd2D\xbfiH\x13?\x95\x17?\xbe\x89\xeb_\xbf\x80D4>{\x06\x87>:\x1c\xde\xbe\xeb\xfa\xf3>p\xad<=\xa1\x06\'\xbfW\x08A\xbf\xfa\xdc\x13>$\xde\x04?\x1c\x07\xb7<\xb4\xe9\x9f\xbe\x04X\x95<i\x0b\x82?\xa8\xab\xfc\xbd\xa1p\xe2>\xf4\x00\x17\xbd\x8b\x19\xa2\xbe-\x15B\xbf\x9bT\xba\xbeW\xe4\x18\xbf\xeb\x98>>\x13\xc4\xaf>\x9a\xa3\x98=\xac\xae\xbf\xbeD\xe2\x1f\xbe\xdbP\x01\xbe\xf4\xe4\x12\xbe\xcc\xa1\x13\xbe\xd09\xd1=\xa5\xde\t?l:9\xbf\xee?q>\'\x13\xb9\xbe\xef31?1\xee\xe5=`\xc3\xac9\xe5\xb8\x8c\xbd\xa5Z*>\xcf\x14\x90={\x7f\xfe=\x9aQ\xf7=\x8d\xdf\n>\xe4\xe2\xd3>fnO\xbf\xbd,R>\x80\xd4\x86\xbe0\x05-\xbf_{\x9a>P\x8b\xad\xbe\xb1\x0e\xbd\xbd\x829\xba>\x80\xd9\x96\xbe2\xae[\xbf1=^\xbf\xdfvy\xbf@:h\xbf\x99\xc4E\xbd\xe2K\r\xbfM\xc1\x10\xbeN\xcf\xa8<\xfb\xaf\x8d<X\xf5\xdc;\xf2\xd9}>]4\x8b\xbd\xf6\xa5\xf2\xbe\x89\x01?\xbfX,Q\xbe\x07\xa5\x01?~|[>\xd1\x07g\xbfe\xa7\x80?Or#\xbf+\xfc\x99\xbe\x86\xbc\x89\xbex\xca\x03?\xd2\xf6\xf4>\xcd\xf1\xca>\xf4\xb7\xf9\xbcw\x9f\xe6\xbe"\xc7\xdd>\xf5\xd8\x11?\t\xd5\x98\xbe\x9f\x03[>t\xc5\xd3\xbcE\xa9\xe5\xbe\xbf\x07g>F\xdbe>\x14\xc1\xb0>\x9c\x1a\x11?\x9a\xa4}\xbe\x81X\xd0>\xc1\xa3A\xbfu\x18\x98\xbe'
    
    def __getitem__(self, index: int):
        meta = self.region_anno_subset[index]
        img_id: int = meta['id']
        np_image = cv2.imread(os.path.join(self.img_dir, f"{img_id}.jpg"))
        embed_table = torch.load(os.path.join(self.cache_dir, f"{img_id}.pth"))

        ih, iw, ic = np_image.shape
        bboxes = []
        phr_embed = []
        phrases = []
        for region in meta['regions']:
            x: int = min(max(0, region['x']), iw)
            y: int = min(max(0, region['y']), ih)
            w: int = min(iw - x, region['width'])
            h: int = min(ih - y, region['height'])
            phrase: str = region['phrase']
            reg_id: int = region['region_id']

            if w <= 5 or h <= 5:
                # discard extremly small/incorrectly labeled bbox
                continue

            phrases.append(phrase)
            phr_embed.append(embed_table[reg_id].to(torch.float32))
            bboxes.append([x, y, w, h])
        
        if self.augmentor:
            category_ids = [0] * len(bboxes)  # TODO: decide to make use of category label or not
            # HACK: we use category_ids's slot to place bbox embedding, so we only keep embeds that still inside the image after augmentation
            box_idx = list(range(len(bboxes)))
            transform = self.augmentor(np_image, bboxes, box_idx)
            image, bboxes, box_idx = transform['image'], transform['bboxes'], transform['category_ids']
            phrases = [phrases[id] for id in box_idx]
            phr_embed = [phr_embed[id] for id in box_idx]
        else:
            image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image)
        
        if self.split == 'train' and random.random() < 0.1:
            phr_embed = torch.frombuffer(copy.copy(self.find_all_objects), dtype=torch.float32)
            phrases = ['find all objects']
        else:
            if self.split != 'train':
                assign = self.subsample(copy.deepcopy(bboxes))
                bboxes = [b for b, a in zip(bboxes, assign) if a == 1]
                phr_embed = [b for b, a in zip(phr_embed, assign) if a == 1]
                phrases = [b for b, a in zip(phrases, assign) if a == 1]

            # NOTE: one query phrase can have multiple matching boxes
            pick_one = random.randint(0, len(bboxes) - 1)
            assign = self.keep_similiar_captions(phr_embed[pick_one], phr_embed)
            phr_embed = phr_embed[pick_one]
            bboxes = [b for b, a in zip(bboxes, assign) if a]
            phrases = [b for b, a in zip(phrases, assign) if a]
        
        num_box = len(bboxes)
        pad_box = 0
        for _ in range(num_box, self.max_det + 1):
            bboxes.append([-1, -1, -1, -1])
            pad_box += 1
        
        fixed_one_cls_onehot = torch.cat([
            torch.ones([num_box, 1], dtype=torch.float32),
            torch.zeros([pad_box, 1], dtype=torch.float32),
        ], dim=0)
        bboxes = torch.tensor(bboxes, dtype=torch.float32)  # (n, 4)
        labels = torch.cat([bboxes, fixed_one_cls_onehot], dim=-1)

        if self.split == 'train':
            if self.offline_embed:
                return image, phr_embed, labels
            else:
                # NOTE: if we are using clip model that is different that the one used to precompute text embedding
                return image, phrase, labels
        else:
            h, w, c = np_image.shape
            c, _h, _w = image.shape

            scale = _h / h
            diff = np.abs(h - w)
            p1 = diff // 2
            p2 = diff - diff // 2
            pad = (0, p1, 0, p2) if w >= h else (p1, 0, p2, 0)
            pad = torch.tensor(pad)
            # print(tmp, bboxes, scale, pad)
            extra = {
                'phrases': '&&'.join(phrases),
                'image_numpy': transform['image_numpy'],
                'image_id': img_id,
            }
            if self.offline_embed:
                return image, phr_embed, labels, scale, pad, extra
            else:
                return image, phrase, labels, scale, pad, extra


class Laion400M(VisionDataset):

    MODEL = "convnext_large_d"
    collect_fn = None

    def __init__(self, 
                 img_dir: str, 
                 bbox_augmentor: Optional[Bbox_Augmentor]=None,
                 **kwargs):
        self.img_dir = img_dir
        self._list_dataset_files(img_dir)
        self.img_ids = sorted(self.data_dict.keys())
        self.augmentor = bbox_augmentor        
    
    @property
    def phrase_embed(self):
        if not hasattr(self, '_phrase_embed'):
            self.cache_file = os.path.join(self.img_dir, "embed_cache.pth")
            if not os.path.exists(self.cache_file):
                self._create_region_embed(self.cache_file)
            self._phrase_embed = torch.load(self.cache_file)
        return self._phrase_embed
    
    def _list_dataset_files(self, root_dir: str):
        assert os.path.exists(root_dir)
        
        imgs = glob.glob(os.path.join(root_dir, "**/*.jpg"), recursive=True)
        captions = glob.glob(os.path.join(root_dir, "**/*.txt"), recursive=True)
        data_dict = {}
        for img_path in imgs:
            img_id = os.path.basename(img_path).replace('.jpg', '')
            data_dict[img_id] = {'img_path':  img_path}
        for cap_txt in captions:
            img_id = os.path.basename(cap_txt).replace('.txt', '')
            if img_id in data_dict:
                with open(cap_txt, mode='r') as f:
                    data_dict[img_id]['caption'] = f.read()
        self.data_dict = {
            k: v for k, v in data_dict.items() 
            if 'img_path' in v and 'caption' in v
        }
    
    def _create_region_embed(self, cache_file: str):
        import open_clip
        
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(self.MODEL, pretrained='laion2b_s26b_b102k_augreg')
        tokenizer = open_clip.get_tokenizer(self.MODEL)
        model = model.to('cuda')
        
        embed_table = {}
        with torch.no_grad():
            for img_id, meta in self.data_dict.items():
                caption = meta['caption']
                input_ids = tokenizer(caption)
                embed = model.encode_text(input_ids.to('cuda'))
                assert embed.ndim == 2 and embed.size(0) == 1
                embed_table[img_id] = embed.cpu()[0]
        
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        torch.save(embed_table, cache_file)
    
    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, index: int):
        img_id: str = self.img_ids[index]

        image = cv2.imread(self.data_dict[img_id]['img_path'])
        
        if self.augmentor:
            transform = self.augmentor(image)
            image = transform['image']
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image)
        
        txt_embed = self.phrase_embed[img_id]
        return image, txt_embed
