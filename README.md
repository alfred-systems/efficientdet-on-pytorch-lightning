# Pipeline for Object Detection in PyTorch Lightning

This is a [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/)-based pipeline for training EfficientDet on COCO 2017 dataset. For better efficiency, it logs with [Wandb](https://docs.wandb.ai/), configures training with yaml files and manages them with [Hydra](https://hydra.cc/docs/intro/), and defines augmentation with [Albumentations](https://albumentations.ai/docs/). But if you want to train a different model, you can easily create a new training by slight modifications to the template.  

**NOTE**: Lot of the code is probabily taken from [Yet-Another-EfficientDet-Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch), you can go back and check the correctnest of the code if needed.

### TODO

- [X] Sanity check trianing a MS-COCO B0 detector and eval its performance
- [ ] Replace the detector backbone with CLIP-like model, and train it on MS-COCO
- [ ] Freeze CLIP-like backbone, and train it on MS-COCO
- [ ] Add support for Visual-Gnome dataset and bounding box description
- [ ] Replace classification head with embedding head for per anchor CLIP-like training target

### Requirements

```
pytorch >= 1.11.0
pytorch-lightning >= 1.6.5
wandb >= 0.12.21
hydra >= 1.2.0
albumentations >= 1.2.1
```
***Warning*** : There is an issue that on pytorch 1.12, checkpoints from Adam and AdamW cannot be resumed for another training. 

## Updates
- **[22.09.12]** Initial commit


## Training
### 1. Configure Trainer
Configure pl.Trainer with a single yaml file. The file will pass to it its arguments and every modules in Callbacks, Profiler, Logger, Strategy, and Plugins of PyTorch Lightning through a Config_Trainer object.  

```  
Trainer:
  max_epochs: 100
  accelerator: gpu
  accumulate_grad_batches: 8
  log_every_n_steps: 100

Callbacks:
  EarlyStopping:
    monitor: AP
    patience: 10
    check_on_train_epoch_end: False

  LearningRateMonitor:
    logging_interval: step

Profiler:
  PyTorchProfiler:
```

### 2. Configure the experiment
Define every hyperparameters involved in training and how to log on Wandb. COCO_EfficientDet class is a template that defines the training, validation and test process, and it contains all the hyperparameters required for training, such as foreground threshold for focal loss or learning rate.  

What's new about logging here is that you can also log artifacts in Wandb, which are not supported by original WandbLogger of PyTorch Lightning. 

```
defaults:
  - _self_
  - dataset: COCO2017
  - trainer: default

batch_size: 4
num_workers: 0
ckpt_path:

model:
  model:
    coeff: 0
  loss:
    fore_th: 0.5
    back_th: 0.4
  nms:
    iou_th: 0.5
    max_det: 400
  optimizer:
    lr: 0.0001

log:
  name: exp0-run0
  project: COCO_EfficientDet
  save_dir: ./log
  artifact_type: find_lr
  artifact_name: experiment_0
  artifact_description: lr=0.0001 | scheduler=ReduceLROnPlateau | monitor=AP
  artifact_save_files:
    trainer: config/trainer/default.yaml
```
If you want to continue the previous training, give the checkpoint file from Artifacts of Wandb or local directory to *ckpt_path*.


### 3. Train
Once you've configured the trainer and experiment, type the command line like below. The experiment file must be located under *config* directory and its name must be typed without *.yaml*.
```
python train.py --config-name experiment_0
```

### 4. Test
Enter the checkpoint file of the model's weight that you want to test to *ckpt_path*. Then the result converted to COCO-style will be saved as a json file.
```
python test.py --config-name experiment_0
```

## License
BSD 3-Clause License Copyright (c) 2022, hotcouscous1
