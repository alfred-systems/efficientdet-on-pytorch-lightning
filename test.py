import os
import sys
import time
import argparse
import torch
from loguru import logger

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


@logger.catch
def dataset_sanity():
    import matplotlib.pyplot as plt
    # from src.dataset.val_dataset import COCO_Detection
    from src.dataset.val_dataset import Validate_Detection
    from src.dataset.bbox_augmentor import debug_augmentor, bbox_safe_augmentor
    from src.dataset.train_dataset import Laion400M, VisualGenome, VisualGenomeFuseDet

    # augmentor = debug_augmentor(512)
    # dataset = COCO_Detection(
    #     "/home/ron/Downloads/mscoco/train2017", 
    #     "/home/ron/Downloads/mscoco/annotations_trainval2017/instances_train2017.json",
    #     augmentor)
    # dataset = Validate_Detection(
    #     "/home/ron/Downloads/mscoco/val2017", 
    #     "/home/ron/Downloads/mscoco/annotations_trainval2017/instances_val2017.json",
    #     512)
    dataset = VisualGenomeFuseDet(
        "/home/ron_zhu/visual_genome/VG_100K", 
        "/home/ron_zhu/visual_genome/region_descriptions.json", 
        bbox_safe_augmentor(384),
        split='train',
        offline_embed=False,
    )

    # fp16 = {k: v.to(torch.float16) for k, v in dataset.phrase_embed.items()}
    # torch.save(fp16, dataset.cache_file.replace(".pth", ".fp16.pth"))
    test_loader = DataLoader(dataset, batch_size=8, num_workers=0)

    for batch in test_loader:
        image = batch[0]
        extra = batch[-1]
        # print(extra)
        breakpoint()
    
    for i in range(10):
        image = dataset[i][0]
        extra = dataset[i][-1]
        print(extra)


def inferece():
    from src.lightning_model import COCO_EfficientDet, VisGenome_FuseDet
    from src.dataset.val_dataset import Validate_Detection
    from src.dataset.train_dataset import Laion400M, VisualGenome, VisualGenomeFuseDet
    from src.dataset.bbox_augmentor import debug_augmentor, bbox_safe_augmentor

    pl_model = VisGenome_FuseDet.load_from_checkpoint("last.ckpt")
    print('model loaded')
    device = 'cpu'
    pl_model = pl_model.to(device).to(torch.float16)
    pl_model.eval()
    
    # test_set = Validate_Detection("/home/ron_zhu/efficlipdet/sample", pl_model.model.img_size, [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
    test_set = VisualGenomeFuseDet(
        "/home/ron_zhu/visual_genome/VG_100K", 
        "/home/ron_zhu/visual_genome/region_descriptions.json", 
        bbox_safe_augmentor(512),
        split='val'
    )
    for batch_size in [1, 1, 4, 8, 16, 32]:
        test_loader = DataLoader(test_set, batch_size=batch_size)

        with torch.no_grad():
            times = []
            for i, batch in enumerate(test_loader):
                if i > 40: break
                # print(batch[0])
                t1 = time.time()
                img = batch[0].to(device).to(torch.float16)
                text = batch[1].to(device).to(torch.float16)

                predict = pl_model(img, text, detect=True)[0]
                
                # box = predict[..., :4]
                # cls = predict[..., 4:]
                # cls = torch.nn.functional.softmax(cls, dim=-1)
                # cls = torch.nn.functional.sigmoid(cls)
                # for a, b in zip(cls[0], box[0]):
                #     # print(a)
                #     breakpoint()
                #     if a.max() > 0.6:
                #         print(b.int())
                #         # print(a)
                # print(predict.shape)
                
                td = time.time() - t1
                times.append(td)
        speed_ms = sum(times) / len(times) * 1000
        print(f'batch_size: {batch_size}, inference time per bactch ~ ', f"{speed_ms:.4f} ms")

    # print(pl_model)

@logger.catch
def dataset_warmup():
    from src.dataset.train_dataset import Laion400M, VisualGenome
    from src.dataset.bbox_augmentor import debug_augmentor
    dataset = VisualGenome(
        "/home/ron_zhu/visual_genome/VG_100K", 
        "/home/ron_zhu/visual_genome/region_descriptions.json", 
        debug_augmentor(384),
    )
    dataset._split_embed_table_by_img(
        dataset.phrase_embed,
        "/home/ron_zhu/visual_genome/VG_100K/region_embeds"
    )


    # dataset = Laion400M("/home/ron_zhu/laion-400m/train_data", debug_augmentor(384))
    
    # import torch
    # import open_clip
    # from src.loss.focal_loss import clip_loss
    # from PIL import Image

    # model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
    #     Laion400M.MODEL, pretrained='laion2b_s26b_b102k_augreg')
    # tokenizer = open_clip.get_tokenizer(Laion400M.MODEL)
    # # model = model.to('cuda')

    # image, txt_emb = [], []
    # for i in range(16):
    #     img, txt = dataset[i]
    #     img = Image.fromarray(img.permute(1, 2, 0).numpy())
    #     image.append(preprocess_val(img))
    #     txt_emb.append(txt)
    # image = torch.stack(image, dim=0)
    # txt_emb = torch.stack(txt_emb, dim=0)
    # img_emb = model.encode_image(image)
    # loss = clip_loss(img_emb, txt_emb)  # mean ~= 2.7
    # print(loss)
    # breakpoint()
    # return 

    
if __name__ == "__main__":
    # test()
    # inferece()
    dataset_sanity()
    # dataset_warmup()
