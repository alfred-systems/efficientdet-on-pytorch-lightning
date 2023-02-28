import timm
import torch

# name = "hf_hub:timm/vit_base_patch16_clip_384.laion2b_ft_in12k_in1k"
# name = "convnext_base.clip_laion2b"
# name = "efficientnet_b0"
# model = timm.create_model(name, pretrained=True, features_only=True)
# X = torch.randn(2, 3, 512, 512) 
# out = model(X)
# print([o.shape for o in out])
# breakpoint()
# print(out)
"""
name = "convnext_base.clip_laion2b"
(Pdb) p [o.shape for o in out]
386x386
[torch.Size([2, 128, 96, 96]), torch.Size([2, 256, 48, 48]), torch.Size([2, 512, 24, 24]), torch.Size([2, 1024, 12, 12])]
512x512
[torch.Size([2, 128, 128, 128]), torch.Size([2, 256, 64, 64]), torch.Size([2, 512, 32, 32]), torch.Size([2, 1024, 16, 16])]

name = "efficientnet_b0"
512x512
[torch.Size([2, 16, 256, 256]), torch.Size([2, 24, 128, 128]), torch.Size([2, 40, 64, 64]), torch.Size([2, 112, 32, 32]), torch.Size([2, 320, 16, 16])]
"""

def check_coco_val_json():
    import os
    import json
    from collections import defaultdict
    from PIL import Image
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches


    with open("result/val/run-2023-02-03-23-06/epoch-0-20.json") as f:
        coco = f.read()

    coco_json = json.loads(coco)
    pred_by_img = defaultdict(list)

    for pred in coco_json:
        imd_id = pred['image_id']
        pred_by_img[imd_id].append(pred)

    for k, preds in pred_by_img.items():
        path = os.path.join("/home/ron/Downloads/mscoco/val2017/", f"{k:012}.jpg")
        pil_img = Image.open(path)

        # Create figure and axes
        fig, ax = plt.subplots()
        # Display the image
        ax.imshow(pil_img)

        for pred in preds:
            x, y, w, h = [int(v) for v in pred['bbox']]
            # Create a Rectangle patch
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')

            # Add the patch to the Axes
            ax.add_patch(rect)
        plt.show()

        input('enter...')

def check_vg_anno_json():
    import os
    import json
    from collections import defaultdict
    from PIL import Image
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches


    with open("/home/ron/Downloads/region_descriptions.json") as f:
        anno = f.read()

    anno_json = json.loads(anno)
    pred_by_img = defaultdict(list)

    # for k, preds in pred_by_img.items():
    img_id = 1
    path = os.path.join("/home/ron/Downloads/", f"{img_id}.jpg")
    pil_img = Image.open(path)

    img_reg = [a for a in anno_json if a['id'] == img_id][0]

    # Create figure and axes
    fig, ax = plt.subplots()
    # Display the image
    ax.imshow(pil_img)

    for region in img_reg['regions']:
        x, y, w, h = region['x'], region['y'], region['width'], region['height']
        # Create a Rectangle patch
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)
    plt.show()

    input('enter...')

check_vg_anno_json()