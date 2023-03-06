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


class VisualGenome(VisionDataset):
    
    MODEL = "convnext_base_w"
    EMBED_SIZE = 640
    collect_fn = None
    max_det = 300  # NOTE: VG have max 267/ min 3/ avg 50 regions per image

    def __init__(self, img_dir: str, 
                 annFile: str = '', 
                 bbox_augmentor: Optional[Bbox_Augmentor]=None,
                 split='train',
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
        
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(self.MODEL, pretrained='laion2B-s13B-b82K')
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

    max_det = 100
    
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
            return image, phr_embed, labels
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
            return image, phr_embed, labels, scale, pad, extra


class Laion400M(VisionDataset):

    MODEL = "convnext_base_w"
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
        
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(self.MODEL, pretrained='laion2B-s13B-b82K')
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
