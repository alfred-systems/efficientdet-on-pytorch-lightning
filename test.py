import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.abspath('./src')))
# from src.__init__ import *
from torch.utils.data import DataLoader

# parser = argparse.ArgumentParser()
# parser.add_argument("--config-name", dest='config_name', default=None, type=str)
# args = parser.parse_args()


# @hydra.main(version_base=None, config_path='./config/', config_name=args.config_name)
# def test(cfg: DictConfig):
#     from src.lightning_model import COCO_EfficientDet
#     from src.dataset.val_dataset import Validate_Detection
#     from torch.utils.data import DataLoader
#     from src.utils.config_trainer import Config_Trainer

#     # lightning model
#     pl_model = COCO_EfficientDet(**cfg.model.model, **cfg.model.loss, **cfg.model.nms, **cfg.model.optimizer)

#     # dataset and dataloader
#     test_set = Validate_Detection(cfg.dataset.test.root, pl_model.model.img_size, cfg.dataset.dataset_stat)
#     test_loader = DataLoader(test_set, batch_size=1)

#     # trainer
#     cfg_trainer = Config_Trainer(cfg.trainer)()
#     trainer = pl.Trainer(**cfg_trainer, logger=False, num_sanity_val_steps=0)

#     # run test
#     if 'ckpt_path' in cfg:
#         trainer.test(pl_model, test_loader, ckpt_path=cfg.ckpt_path)
#     else:
#         raise RuntimeError('no checkpoint is given')


def dataset_sanity():
    import matplotlib.pyplot as plt
    # from src.dataset.val_dataset import COCO_Detection
    from src.dataset.val_dataset import Validate_Detection
    # from src.dataset.bbox_augmentor import debug_augmentor
    # augmentor = debug_augmentor(512)
    # dataset = COCO_Detection(
    #     "/home/ron/Downloads/mscoco/train2017", 
    #     "/home/ron/Downloads/mscoco/annotations_trainval2017/instances_train2017.json",
    #     augmentor)
    dataset = Validate_Detection(
        "/home/ron/Downloads/mscoco/val2017", 
        "/home/ron/Downloads/mscoco/annotations_trainval2017/instances_val2017.json",
        512)
    test_loader = DataLoader(dataset, batch_size=2)

    for batch in test_loader:
        image = batch[0]
        extra = batch[-1]
        print(extra)
        break
    
    for i in range(10):
        image = dataset[i][0]
        extra = dataset[i][-1]
        print(extra)


def inferece():
    from src.lightning_model import COCO_EfficientDet
    from src.dataset.val_dataset import Validate_Detection

    pl_model = COCO_EfficientDet.load_from_checkpoint("./log/COCO_EfficientDet/n7ulowov/checkpoints/epoch=1-step=7394.ckpt")
    print('model loaded')
    pl_model = pl_model.to('cuda')
    test_set = Validate_Detection("/home/ron_zhu/efficlipdet/sample", pl_model.model.img_size, [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
    test_loader = DataLoader(test_set, batch_size=1)
    for batch in test_loader:
        print(batch[0])
        predict = pl_model(batch[1], detect=True)[0]
        box = predict[..., :4]
        cls = predict[..., 4:]
        # cls = torch.nn.functional.softmax(cls, dim=-1)
        # cls = torch.nn.functional.sigmoid(cls)
        for a, b in zip(cls[0], box[0]):
            # print(a)
            breakpoint()
            if a.max() > 0.6:
                print(b.int())
                # print(a)
        print(predict.shape)
    # print(pl_model)


def dataset_warmup():
    from src.dataset.train_dataset import Laion400M, VisualGenome
    dataset = Laion400M("/home/ron_zhu/laion-400m/train_data", None)

    
if __name__ == "__main__":
    # test()
    # inferece()
    # dataset_sanity()
    dataset_warmup()
