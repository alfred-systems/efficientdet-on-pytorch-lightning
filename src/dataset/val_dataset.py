from src.dataset.bbox_augmentor import *
from src.dataset.utils import imagenet_fill
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from src.dataset.train_dataset import CLASS_TABLE


class Validate_Detection(Dataset):

    def __init__(self,
                 root: str,
                 annFile: str,
                 img_size: int,
                 dataset_stat: Tuple = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                 max_det=100
                 ):

        self.coco = COCO(annFile)
        self.root = root
        self.img_paths = os.listdir(self.root)
        self.img_paths.sort()

        self.augmentor = Bbox_Augmentor(total_prob=1, min_area=0, min_visibility=0,
                                        dataset_stat=dataset_stat, ToTensor=True,
                                        with_label=False, with_np_image=True,)

        self.augmentor.append(A.LongestMaxSize(img_size, p=1))
        self.augmentor.append(A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT, value=imagenet_fill(), p=1))
        self.augmentor.make_compose()

        self.img_size = img_size
        self.dataset_stat = dataset_stat
        self.max_det = max_det


    def __len__(self):
        return len(self.img_paths)


    def __getitem__(self, index: int) -> Tuple[str, Tensor, float, Tensor]:
        img_path = self.img_paths[index]
        img_id = img_path.split(".")[0]

        image = cv2.imread(os.path.join(self.root, img_path))
        h, w, _ = image.shape

        scale = self.img_size / max(h, w)

        diff = np.abs(h - w)
        p1 = diff // 2
        p2 = diff - diff // 2
        pad = (0, p1, 0, p2) if w >= h else (p1, 0, p2, 0)
        pad = torch.tensor(pad)

        data = self.augmentor(image, None, None)
        image = data['image']
        # image = image.to(device=device)

        coco = self.coco
        target = coco.loadAnns(coco.getAnnIds(imgIds=int(img_id)))

        bboxes, category_ids = [], []
        for i, t in enumerate(target):
            bbox = t['bbox']
            if 0.0 in bbox[2:]:
                bbox[2] += 1e-7
                bbox[3] += 1e-7
            
            bboxes.append(bbox)
            category_ids.append(CLASS_TABLE[t['category_id']])
        data['boxes'] = torch.tensor(bboxes + [[-1] * 4] * (self.max_det - len(bboxes)))
        data['labels'] = torch.LongTensor(category_ids + [-1] * (self.max_det - len(bboxes)))

        return img_id, image, scale, pad, data

