import os
import json

from src.dataset.utils import *
from src.dataset.bbox_augmentor import Bbox_Augmentor
from pycocotools.coco import COCO
from torchvision.datasets.vision import VisionDataset


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


class COCO_Detection(VisionDataset):

    num_classes = 80
    coco_cat = (i for i in range(1, 91))
    missing_cat = (12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91)

    def __init__(self,
                 root: str,
                 annFile: str,
                 bbox_augmentor: Optional[Bbox_Augmentor],
                 background_class=True,):

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
    MODEL = "convnext_base.clip_laion2b"

    def __init__(self, img_dir: str, 
                 region_anno_path: str, 
                 bbox_augmentor: Optional[Bbox_Augmentor]):
        self.img_dir = img_dir
        with open(region_anno_path, mode='r') as f:
            self.region_anno = json.load(f)
        self.augmentor = bbox_augmentor
    
    def create_region_embed(self, cache_file: str):
        import open_clip
        
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('convnext_base_w', pretrained='laion2B-s13B-b82K')
        tokenizer = open_clip.get_tokenizer('convnext_base_w')
        model = model.to('cuda')

        raise NotImplementedError("this is a in-completed implementation, dont use it yet")
        
        embed_table = {}
        for i in range(len(self)):
            reg_id = self.region_anno[i]['region_id']
            phrase = self.region_anno[i]['phrase']
            input_ids = tokenizer(phrase)
            embed = model.encode_text(input_ids)
            assert embed.ndim == 2 and embed.size(0) == 1
            embed_table[reg_id] = embed.cpu()[0]
        
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        torch.save(embed_table)
    
    def __len__(self):
        return len(self.region_anno)
    
    def __getitem__(self, index: int):
        meta = self.region_anno[index]
        img_id: int = meta['image_id']
        image = cv2.imread(os.path.join(self.img_dir, f"{img_id}.jpg"))
        
        if self.augmentor:
            transform = self.augmentor(image, bboxes, category_ids)
            image, bboxes, category_ids = transform.values()
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image)
        
        for region in meta['regions']:
            x: int = region['x']
            y: int = region['y']
            w: int = region['width']
            h: int = region['height']
            phrase: str = region['phrase']
            reg_id: int = region['region_id']
