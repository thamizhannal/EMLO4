# Dog Breed Classifier

### [Dogs Dataset](https://www.kaggle.com/datasets/khushikhushikhushi/dog-breed-image-dataset)
Dogs Breed classifier dataset 
###### step 1: Download kaggle dataset using CLI
Install the kaggle-cli and download the dog-breed-image dataset using below commands
```
$pip install kaggle-cli
$kaggle datasets download khushikhushikhushi/dog-breed-image-dataset
```

### Directory tree Structure
```sh.
$ tree 
├── configs
│   ├── callbacks
│   ├── data
│   ├── experiment
│   ├── hydra
│   ├── logger
│   ├── model
│   ├── paths
│   ├── trainer
│   ├── train_old.yaml
│   └── train.yaml
├── data
├── Dockerfile
├── .gitignore
├── .python-version
├── .env
├── pyproject.toml
├── README.md
├── src
│   ├── datamodules
│   ├── infer.py
│   ├── models
│   ├── train.py
│   └── utils
└── tests
    ├── conftest.py
    ├── datamodules
    ├── models
    └── test_train.py

```

# Classifier Train and Test using docker compose module
```sh
$docker compose build train

$docker compose run -it train  
WARN[0000] /Users/vehere/mlworks/EMLO4/week4_assignement/lightning-hydra/docker-compose.yml: the attribute `version` is obsolete, it will be ignored, please remove it to avoid potential confusion 
WARN[0000] Found orphan containers ([lightning-hydra-train-run-7cfaf0d6a84b lightning-hydra-train-run-8e75c757ef77]) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up. 
task_name: train
tags:
- finetune
train: true
test: true
ckpt_path: null
seed: 42
data:
  _target_: src.datamodules.dogs_datamodule.DogsBreadDataModule
  data_dir: ${paths.data_dir}/dogs_dataset
  batch_size: 32
  num_workers: 0
  pin_memory: true
model:
  _target_: src.models.dogs_classifier.DogsBreedClassifier
  model_name: resnet18
  num_classes: 10
  pretrained: true
  lr: 0.001
  weight_decay: 1.0e-05
  scheduler_factor: 0.1
  scheduler_patience: 10
  min_lr: 1.0e-06
  trainable: false
callbacks:
  model_checkpoint:
    _target_: lightning.pytorch.callbacks.ModelCheckpoint
    dirpath: ${paths.output_dir}/checkpoints
    filename: epoch_{epoch:03d}
    monitor: val/acc
    verbose: false
    save_last: false
    save_top_k: 1
    mode: max
    auto_insert_metric_name: false
    save_weights_only: false
    every_n_train_steps: null
    train_time_interval: null
    every_n_epochs: null
    save_on_train_epoch_end: null
  early_stopping:
    _target_: lightning.pytorch.callbacks.EarlyStopping
    monitor: val/acc
    min_delta: 0.0
    patience: 10
    verbose: false
    mode: max
    strict: true
    check_finite: true
    stopping_threshold: null
    divergence_threshold: null
    check_on_train_epoch_end: null
  model_summary:
    _target_: lightning.pytorch.callbacks.RichModelSummary
    max_depth: -1
  rich_progress_bar:
    _target_: lightning.pytorch.callbacks.RichProgressBar
  learning_rate_monitor:
    _target_: lightning.pytorch.callbacks.LearningRateMonitor
    logging_interval: step
logger:
  tensorboard:
    _target_: lightning.pytorch.loggers.tensorboard.TensorBoardLogger
    save_dir: ${paths.output_dir}/tensorboard/
    name: null
    log_graph: false
    default_hp_metric: true
    prefix: ''
trainer:
  _target_: lightning.Trainer
  default_root_dir: ${paths.output_dir}
  min_epochs: 1
  max_epochs: 3
  accelerator: auto
  devices: 1
  check_val_every_n_epoch: 3
  deterministic: false
paths:
  root_dir: ${oc.env:PROJECT_ROOT}
  data_dir: ${paths.root_dir}/data/
  log_dir: ${paths.root_dir}/logs/
  output_dir: ${hydra:runtime.output_dir}
  work_dir: ${hydra:runtime.cwd}

[2024-10-03 17:20:32,179][__main__][INFO] - Instantiating datamodule <src.datamodules.dogs_datamodule.DogsBreadDataModule>
[2024-10-03 17:20:32,309][__main__][INFO] - Instantiating model <src.models.dogs_classifier.DogsBreedClassifier>
[2024-10-03 17:20:32,832][timm.models._builder][INFO] - Loading pretrained weights from Hugging Face hub (timm/resnet18.a1_in1k)
model.safetensors: 100%|████████████████████████████████████████████████████████████████████████████████████████| 46.8M/46.8M [01:25<00:00, 546kB/s]
[2024-10-03 17:21:59,287][timm.models._hub][INFO] - [timm/resnet18.a1_in1k] Safe alternative available for 'pytorch_model.bin' (as 'model.safetensors'). Loading weights using safetensors.
[2024-10-03 17:21:59,350][timm.models._builder][INFO] - Missing keys (fc.weight, fc.bias) discovered while loading pretrained weights. This is expected if model is being adapted.
[2024-10-03 17:21:59,422][__main__][INFO] - Instantiating callback <lightning.pytorch.callbacks.ModelCheckpoint>
[2024-10-03 17:21:59,438][__main__][INFO] - Instantiating callback <lightning.pytorch.callbacks.EarlyStopping>
[2024-10-03 17:21:59,439][__main__][INFO] - Instantiating callback <lightning.pytorch.callbacks.RichModelSummary>
[2024-10-03 17:21:59,439][__main__][INFO] - Instantiating callback <lightning.pytorch.callbacks.RichProgressBar>
[2024-10-03 17:21:59,441][__main__][INFO] - Instantiating callback <lightning.pytorch.callbacks.LearningRateMonitor>
[2024-10-03 17:21:59,442][__main__][INFO] - Instantiating logger <lightning.pytorch.loggers.tensorboard.TensorBoardLogger>
[2024-10-03 17:21:59,447][__main__][INFO] - Instantiating trainer <lightning.Trainer>
Trainer already configured with model summary callbacks: [<class 'lightning.pytorch.callbacks.rich_model_summary.RichModelSummary'>]. Skipping setting a default `ModelSummary` callback.
GPU available: False, used: False
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
2024-10-03 17:21:59 | INFO     | src.utils.logging_utils:wrapper:17 - Starting train
[2024-10-03 17:21:59,478][__main__][INFO] - Starting training!
┏━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┓
┃    ┃ Name                        ┃ Type                 ┃ Params ┃ Mode  ┃
┡━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━┩
│ 0  │ model                       │ ResNet               │ 11.2 M │ train │
│ 1  │ model.conv1                 │ Conv2d               │  9.4 K │ train │
│ 2  │ model.bn1                   │ BatchNorm2d          │    128 │ train │
│ 3  │ model.act1                  │ ReLU                 │      0 │ train │
│ 4  │ model.maxpool               │ MaxPool2d            │      0 │ train │
│ 5  │ model.layer1                │ Sequential           │  147 K │ train │
│ 6  │ model.layer1.0              │ BasicBlock           │ 74.0 K │ train │
│ 7  │ model.layer1.0.conv1        │ Conv2d               │ 36.9 K │ train │
│ 8  │ model.layer1.0.bn1          │ BatchNorm2d          │    128 │ train │
│ 9  │ model.layer1.0.drop_block   │ Identity             │      0 │ train │
│ 10 │ model.layer1.0.act1         │ ReLU                 │      0 │ train │
│ 11 │ model.layer1.0.aa           │ Identity             │      0 │ train │
│ 12 │ model.layer1.0.conv2        │ Conv2d               │ 36.9 K │ train │
│ 13 │ model.layer1.0.bn2          │ BatchNorm2d          │    128 │ train │
│ 14 │ model.layer1.0.act2         │ ReLU                 │      0 │ train │
│ 15 │ model.layer1.1              │ BasicBlock           │ 74.0 K │ train │
│ 16 │ model.layer1.1.conv1        │ Conv2d               │ 36.9 K │ train │
│ 17 │ model.layer1.1.bn1          │ BatchNorm2d          │    128 │ train │
│ 18 │ model.layer1.1.drop_block   │ Identity             │      0 │ train │
│ 19 │ model.layer1.1.act1         │ ReLU                 │      0 │ train │
│ 20 │ model.layer1.1.aa           │ Identity             │      0 │ train │
│ 21 │ model.layer1.1.conv2        │ Conv2d               │ 36.9 K │ train │
│ 22 │ model.layer1.1.bn2          │ BatchNorm2d          │    128 │ train │
│ 23 │ model.layer1.1.act2         │ ReLU                 │      0 │ train │
│ 24 │ model.layer2                │ Sequential           │  525 K │ train │
│ 25 │ model.layer2.0              │ BasicBlock           │  230 K │ train │
│ 26 │ model.layer2.0.conv1        │ Conv2d               │ 73.7 K │ train │
│ 27 │ model.layer2.0.bn1          │ BatchNorm2d          │    256 │ train │
│ 28 │ model.layer2.0.drop_block   │ Identity             │      0 │ train │
│ 29 │ model.layer2.0.act1         │ ReLU                 │      0 │ train │
│ 30 │ model.layer2.0.aa           │ Identity             │      0 │ train │
│ 31 │ model.layer2.0.conv2        │ Conv2d               │  147 K │ train │
│ 32 │ model.layer2.0.bn2          │ BatchNorm2d          │    256 │ train │
│ 33 │ model.layer2.0.act2         │ ReLU                 │      0 │ train │
│ 34 │ model.layer2.0.downsample   │ Sequential           │  8.4 K │ train │
│ 35 │ model.layer2.0.downsample.0 │ Conv2d               │  8.2 K │ train │
│ 36 │ model.layer2.0.downsample.1 │ BatchNorm2d          │    256 │ train │
│ 37 │ model.layer2.1              │ BasicBlock           │  295 K │ train │
│ 38 │ model.layer2.1.conv1        │ Conv2d               │  147 K │ train │
│ 39 │ model.layer2.1.bn1          │ BatchNorm2d          │    256 │ train │
│ 40 │ model.layer2.1.drop_block   │ Identity             │      0 │ train │
│ 41 │ model.layer2.1.act1         │ ReLU                 │      0 │ train │
│ 42 │ model.layer2.1.aa           │ Identity             │      0 │ train │
│ 43 │ model.layer2.1.conv2        │ Conv2d               │  147 K │ train │
│ 44 │ model.layer2.1.bn2          │ BatchNorm2d          │    256 │ train │
│ 45 │ model.layer2.1.act2         │ ReLU                 │      0 │ train │
│ 46 │ model.layer3                │ Sequential           │  2.1 M │ train │
│ 47 │ model.layer3.0              │ BasicBlock           │  919 K │ train │
│ 48 │ model.layer3.0.conv1        │ Conv2d               │  294 K │ train │
│ 49 │ model.layer3.0.bn1          │ BatchNorm2d          │    512 │ train │
│ 50 │ model.layer3.0.drop_block   │ Identity             │      0 │ train │
│ 51 │ model.layer3.0.act1         │ ReLU                 │      0 │ train │
│ 52 │ model.layer3.0.aa           │ Identity             │      0 │ train │
│ 53 │ model.layer3.0.conv2        │ Conv2d               │  589 K │ train │
│ 54 │ model.layer3.0.bn2          │ BatchNorm2d          │    512 │ train │
│ 55 │ model.layer3.0.act2         │ ReLU                 │      0 │ train │
│ 56 │ model.layer3.0.downsample   │ Sequential           │ 33.3 K │ train │
│ 57 │ model.layer3.0.downsample.0 │ Conv2d               │ 32.8 K │ train │
│ 58 │ model.layer3.0.downsample.1 │ BatchNorm2d          │    512 │ train │
│ 59 │ model.layer3.1              │ BasicBlock           │  1.2 M │ train │
│ 60 │ model.layer3.1.conv1        │ Conv2d               │  589 K │ train │
│ 61 │ model.layer3.1.bn1          │ BatchNorm2d          │    512 │ train │
│ 62 │ model.layer3.1.drop_block   │ Identity             │      0 │ train │
│ 63 │ model.layer3.1.act1         │ ReLU                 │      0 │ train │
│ 64 │ model.layer3.1.aa           │ Identity             │      0 │ train │
│ 65 │ model.layer3.1.conv2        │ Conv2d               │  589 K │ train │
│ 66 │ model.layer3.1.bn2          │ BatchNorm2d          │    512 │ train │
│ 67 │ model.layer3.1.act2         │ ReLU                 │      0 │ train │
│ 68 │ model.layer4                │ Sequential           │  8.4 M │ train │
│ 69 │ model.layer4.0              │ BasicBlock           │  3.7 M │ train │
│ 70 │ model.layer4.0.conv1        │ Conv2d               │  1.2 M │ train │
│ 71 │ model.layer4.0.bn1          │ BatchNorm2d          │  1.0 K │ train │
│ 72 │ model.layer4.0.drop_block   │ Identity             │      0 │ train │
│ 73 │ model.layer4.0.act1         │ ReLU                 │      0 │ train │
│ 74 │ model.layer4.0.aa           │ Identity             │      0 │ train │
│ 75 │ model.layer4.0.conv2        │ Conv2d               │  2.4 M │ train │
│ 76 │ model.layer4.0.bn2          │ BatchNorm2d          │  1.0 K │ train │
│ 77 │ model.layer4.0.act2         │ ReLU                 │      0 │ train │
│ 78 │ model.layer4.0.downsample   │ Sequential           │  132 K │ train │
│ 79 │ model.layer4.0.downsample.0 │ Conv2d               │  131 K │ train │
│ 80 │ model.layer4.0.downsample.1 │ BatchNorm2d          │  1.0 K │ train │
│ 81 │ model.layer4.1              │ BasicBlock           │  4.7 M │ train │
│ 82 │ model.layer4.1.conv1        │ Conv2d               │  2.4 M │ train │
│ 83 │ model.layer4.1.bn1          │ BatchNorm2d          │  1.0 K │ train │
│ 84 │ model.layer4.1.drop_block   │ Identity             │      0 │ train │
│ 85 │ model.layer4.1.act1         │ ReLU                 │      0 │ train │
│ 86 │ model.layer4.1.aa           │ Identity             │      0 │ train │
│ 87 │ model.layer4.1.conv2        │ Conv2d               │  2.4 M │ train │
│ 88 │ model.layer4.1.bn2          │ BatchNorm2d          │  1.0 K │ train │
│ 89 │ model.layer4.1.act2         │ ReLU                 │      0 │ train │
│ 90 │ model.global_pool           │ SelectAdaptivePool2d │      0 │ train │
│ 91 │ model.global_pool.pool      │ AdaptiveAvgPool2d    │      0 │ train │
│ 92 │ model.global_pool.flatten   │ Flatten              │      0 │ train │
│ 93 │ model.fc                    │ Linear               │  5.1 K │ train │
│ 94 │ train_acc                   │ MulticlassAccuracy   │      0 │ train │
│ 95 │ test_acc                    │ MulticlassAccuracy   │      0 │ train │
│ 96 │ valid_acc                   │ MulticlassAccuracy   │      0 │ train │
└────┴─────────────────────────────┴──────────────────────┴────────┴───────┘
Trainable params: 11.2 M                                                                                                                            
Non-trainable params: 0                                                                                                                             
Total params: 11.2 M                                                                                                                                
Total estimated model params size (MB): 44                                                                                                          
Modules in train mode: 97                                                                                                                           
Modules in eval mode: 0                                                                                                                             
/venv/lib/python3.9/site-packages/lightning/pytorch/trainer/connectors/data_connector.py:424: The 'val_dataloader' does not have many workers which 
may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=9` in the `DataLoader` to improve performance.
/venv/lib/python3.9/site-packages/lightning/pytorch/trainer/connectors/data_connector.py:424: The 'train_dataloader' does not have many workers 
which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=9` in the `DataLoader` to improve 
performance.
/venv/lib/python3.9/site-packages/lightning/pytorch/loops/fit_loop.py:298: The number of training batches (22) is smaller than the logging interval 
Trainer(log_every_n_steps=50). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.
Epoch 2/2  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 22/22 0:01:42 • 0:00:00 0.22it/s v_num: 0.000 train/loss_step: 0.298 train/acc_step: 1.000      
                                                                                     train/loss_epoch: 0.432 train/acc_epoch: 0.978                 
Epoch 2/2  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 22/22 0:01:42 • 0:00:00 0.22it/s v_num: 0.000 train/loss_step: 0.298 train/acc_step: 1.000      
                                                                                     train/loss_epoch: 0.074 train/acc_epoch: 0.999 val/loss_step:  
                                                                                     0.006 val/acc_step: 1.000 val/loss_epoch: 0.037 val/acc_epoch: 
                                                                                     1.000                                                          
[2024-10-03 17:27:10,849][__main__][INFO] - Training metrics:
{'train/loss': tensor(0.0741), 'train/loss_step': tensor(0.2975), 'train/acc': tensor(0.9985), 'train/acc_step': tensor(1.), 'train/loss_epoch': tensor(0.0741), 'train/acc_epoch': tensor(0.9985), 'lr-Adam': tensor(0.0010), 'val/loss': tensor(0.0375), 'val/loss_epoch': tensor(0.0375), 'val/acc': tensor(1.), 'val/acc_epoch': tensor(1.)}
2024-10-03 17:27:10 | INFO     | src.utils.logging_utils:wrapper:20 - Finished train
2024-10-03 17:27:10 | INFO     | src.utils.logging_utils:wrapper:17 - Starting test
[2024-10-03 17:27:10,853][__main__][INFO] - Starting testing!
[2024-10-03 17:27:10,853][__main__][INFO] - Loading best checkpoint: /app/outputs/2024-10-03/17-20-32/checkpoints/epoch_002.ckpt
Restoring states from the checkpoint path at /app/outputs/2024-10-03/17-20-32/checkpoints/epoch_002.ckpt
Loaded model weights from the checkpoint at /app/outputs/2024-10-03/17-20-32/checkpoints/epoch_002.ckpt
/venv/lib/python3.9/site-packages/lightning/pytorch/trainer/connectors/data_connector.py:424: The 'test_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=9` in the `DataLoader` to improve performance.
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│      test/acc_epoch       │    0.9949495196342468     │
│      test/loss_epoch      │    0.0262161772698164     │
└───────────────────────────┴───────────────────────────┘
Testing ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7/7 0:00:05 • 0:00:00 1.47it/s  
[2024-10-03 17:27:16,666][__main__][INFO] - Test metrics:
[{'test/loss_epoch': 0.0262161772698164, 'test/acc_epoch': 0.9949495196342468}]
2024-10-03 17:27:16 | INFO     | src.utils.logging_utils:wrapper:20 - Finished test

```
## confusion matrix
- Train

    ![Training](./assets/confusion_matrix(train).png)

- Test

    ![Testing](./assets/confusion_matrix(test).png)



# Inference
```sh 
$docker compose build infer

$ docker compose run infer         
WARN[0000] /Users/vehere/mlworks/EMLO4/week4_assignement/lightning-hydra/docker-compose.yml: the attribute `version` is obsolete, it will be ignored, please remove it to avoid potential confusion 
WARN[0000] Found orphan containers ([lightning-hydra-train-run-df33365a3abd lightning-hydra-infer-run-1125c9f27285 lightning-hydra-infer-run-191a9ef42700 lightning-hydra-infer-run-06986838e86d lightning-hydra-train-run-5c2e0adde427 lightning-hydra-infer-run-400136e0471a lightning-hydra-train-run-bd8b23ce1c07 lightning-hydra-train-run-7cfaf0d6a84b lightning-hydra-train-run-8e75c757ef77]) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up. 
2024-10-03 17:39:30 | INFO     | utils.logging_utils:wrapper:17 - Starting main
Model loaded successfully!
model.safetensors: 100%|████████████████████████████████████████████████████████████████████████████████████████| 46.8M/46.8M [01:35<00:00, 492kB/s]
Model eval completed!
input_foler:/app/data/dogs_dataset/validation, output_folder:/app/output
File:/app/data/dogs_dataset/validation/Yorkshire_Terrier/Yorkshire Terrier_52.jpg
⠋ Processing images...2024-10-03 17:41:11 | INFO     | utils.logging_utils:wrapper:17 - Starting load_image
2024-10-03 17:41:11 | INFO     | utils.logging_utils:wrapper:20 - Finished load_image
2024-10-03 17:41:11 | INFO     | utils.logging_utils:wrapper:17 - Starting infer
⠹ Processing images...2024-10-03 17:41:12 | INFO     | utils.logging_utils:wrapper:20 - Finished infer
2024-10-03 17:41:12 | INFO     | utils.logging_utils:wrapper:17 - Starting save_prediction_image
⠦ Processing images...2024-10-03 17:41:12 | INFO     | utils.logging_utils:wrapper:20 - Finished save_prediction_image
Processed Yorkshire Terrier_52.jpg: Yorkshire_Terrier (0.99)
File:/app/data/dogs_dataset/validation/Yorkshire_Terrier/Yorkshire Terrier_67.jpg
⠦ Processing images...2024-10-03 17:41:12 | INFO     | utils.logging_utils:wrapper:17 - Starting load_image
2024-10-03 17:41:12 | INFO     | utils.logging_utils:wrapper:20 - Finished load_image
2024-10-03 17:41:12 | INFO     | utils.logging_utils:wrapper:17 - Starting infer
2024-10-03 17:41:12 | INFO     | utils.logging_utils:wrapper:20 - Finished infer
2024-10-03 17:41:12 | INFO     | utils.logging_utils:wrapper:17 - Starting save_prediction_image
⠋ Processing images...2024-10-03 17:41:12 | INFO     | utils.logging_utils:wrapper:20 - Finished save_prediction_image
Processed Yorkshire Terrier_67.jpg: Yorkshire_Terrier (0.99)
File:/app/data/dogs_dataset/validation/Yorkshire_Terrier/Yorkshire Terrier_8.jpg
⠋ Processing images...2024-10-03 17:41:12 | INFO     | utils.logging_utils:wrapper:17 - Starting load_image
⠙ Processing images...2024-10-03 17:41:12 | INFO     | utils.logging_utils:wrapper:20 - Finished load_image
2024-10-03 17:41:12 | INFO     | utils.logging_utils:wrapper:17 - Starting infer
⠏ Processing images...2024-10-03 17:41:13 | INFO     | utils.logging_utils:wrapper:20 - Finished infer
2024-10-03 17:41:13 | INFO     | utils.logging_utils:wrapper:17 - Starting save_prediction_image
⠙ Processing images...2024-10-03 17:41:13 | INFO     | utils.logging_utils:wrapper:20 - Finished save_prediction_image
Processed Yorkshire Terrier_8.jpg: Yorkshire_Terrier (0.99)
File:/app/data/dogs_dataset/validation/Yorkshire_Terrier/Yorkshire Terrier_20.jpg
⠙ Processing images...2024-10-03 17:41:13 | INFO     | utils.logging_utils:wrapper:17 - Starting load_image
2024-10-03 17:41:13 | INFO     | utils.logging_utils:wrapper:20 - Finished load_image
2024-10-03 17:41:13 | INFO     | utils.logging_utils:wrapper:17 - Starting infer
⠼ Processing images...2024-10-03 17:41:13 | INFO     | utils.logging_utils:wrapper:20 - Finished infer
2024-10-03 17:41:13 | INFO     | utils.logging_utils:wrapper:17 - Starting save_prediction_image
⠴ Processing images...2024-10-03 17:41:14 | INFO     | utils.logging_utils:wrapper:20 - Finished save_prediction_image
Processed Yorkshire Terrier_20.jpg: Yorkshire_Terrier (1.00)
File:/app/data/dogs_dataset/validation/Yorkshire_Terrier/Yorkshire Terrier_84.jpg
⠴ Processing images...2024-10-03 17:41:14 | INFO     | utils.logging_utils:wrapper:17 - Starting load_image
2024-10-03 17:41:14 | INFO     | utils.logging_utils:wrapper:20 - Finished load_image
2024-10-03 17:41:14 | INFO     | utils.logging_utils:wrapper:17 - Starting infer
⠦ Processing images...2024-10-03 17:41:14 | INFO     | utils.logging_utils:wrapper:20 - Finished infer
2024-10-03 17:41:14 | INFO     | utils.logging_utils:wrapper:17 - Starting save_prediction_image
⠏ Processing images...2024-10-03 17:41:14 | INFO     | utils.logging_utils:wrapper:20 - Finished save_prediction_image
Processed Yorkshire Terrier_84.jpg: Yorkshire_Terrier (0.99)
File:/app/data/dogs_dataset/validation/Yorkshire_Terrier/Yorkshire Terrier_82.jpg
⠏ Processing images...2024-10-03 17:41:14 | INFO     | utils.logging_utils:wrapper:17 - Starting load_image
2024-10-03 17:41:14 | INFO     | utils.logging_utils:wrapper:20 - Finished load_image
2024-10-03 17:41:14 | INFO     | utils.logging_utils:wrapper:17 - Starting infer
⠋ Processing images...2024-10-03 17:41:14 | INFO     | utils.logging_utils:wrapper:20 - Finished infer
2024-10-03 17:41:14 | INFO     | utils.logging_utils:wrapper:17 - Starting save_prediction_image
⠴ Processing images...2024-10-03 17:41:14 | INFO     | utils.logging_utils:wrapper:20 - Finished save_prediction_image
Processed Yorkshire Terrier_82.jpg: Yorkshire_Terrier (0.99)
File:/app/data/dogs_dataset/validation/Yorkshire_Terrier/Yorkshire Terrier_100.jpg
⠴ Processing images...2024-10-03 17:41:14 | INFO     | utils.logging_utils:wrapper:17 - Starting load_image
2024-10-03 17:41:14 | INFO     | utils.logging_utils:wrapper:20 - Finished load_image
2024-10-03 17:41:14 | INFO     | utils.logging_utils:wrapper:17 - Starting infer
⠦ Processing images...2024-10-03 17:41:14 | INFO     | utils.logging_utils:wrapper:20 - Finished infer
2024-10-03 17:41:14 | INFO     | utils.logging_utils:wrapper:17 - Starting save_prediction_image
⠧ Processing images...2024-10-03 17:41:15 | INFO     | utils.logging_utils:wrapper:20 - Finished save_prediction_image
Processed Yorkshire Terrier_100.jpg: Yorkshire_Terrier (0.99)
File:/app/data/dogs_dataset/validation/Yorkshire_Terrier/Yorkshire Terrier_11.jpg
⠧ Processing images...2024-10-03 17:41:15 | INFO     | utils.logging_utils:wrapper:17 - Starting load_image
2024-10-03 17:41:15 | INFO     | utils.logging_utils:wrapper:20 - Finished load_image
2024-10-03 17:41:15 | INFO     | utils.logging_utils:wrapper:17 - Starting infer
⠇ Processing images...2024-10-03 17:41:15 | INFO     | utils.logging_utils:wrapper:20 - Finished infer
2024-10-03 17:41:15 | INFO     | utils.logging_utils:wrapper:17 - Starting save_prediction_image
⠋ Processing images...2024-10-03 17:41:15 | INFO     | utils.logging_utils:wrapper:20 - Finished save_prediction_image
Processed Yorkshire Terrier_11.jpg: Yorkshire_Terrier (0.99)
File:/app/data/dogs_dataset/validation/Yorkshire_Terrier/Yorkshire Terrier_17.jpg
⠋ Processing images...2024-10-03 17:41:15 | INFO     | utils.logging_utils:wrapper:17 - Starting load_image
2024-10-03 17:41:15 | INFO     | utils.logging_utils:wrapper:20 - Finished load_image
2024-10-03 17:41:15 | INFO     | utils.logging_utils:wrapper:17 - Starting infer
⠙ Processing images...2024-10-03 17:41:15 | INFO     | utils.logging_utils:wrapper:20 - Finished infer
2024-10-03 17:41:15 | INFO     | utils.logging_utils:wrapper:17 - Starting save_prediction_image
⠼ Processing images...2024-10-03 17:41:15 | INFO     | utils.logging_utils:wrapper:20 - Finished save_prediction_image
Processed Yorkshire Terrier_17.jpg: Yorkshire_Terrier (1.00)
File:/app/data/dogs_dataset/validation/Yorkshire_Terrier/Yorkshire Terrier_43.jpg
⠼ Processing images...2024-10-03 17:41:15 | INFO     | utils.logging_utils:wrapper:17 - Starting load_image
2024-10-03 17:41:15 | INFO     | utils.logging_utils:wrapper:20 - Finished load_image
2024-10-03 17:41:15 | INFO     | utils.logging_utils:wrapper:17 - Starting infer
⠴ Processing images...2024-10-03 17:41:15 | INFO     | utils.logging_utils:wrapper:20 - Finished infer
2024-10-03 17:41:15 | INFO     | utils.logging_utils:wrapper:17 - Starting save_prediction_image
⠇ Processing images...2024-10-03 17:41:15 | INFO     | utils.logging_utils:wrapper:20 - Finished save_prediction_image
Processed Yorkshire Terrier_43.jpg: Yorkshire_Terrier (1.00)
File:/app/data/dogs_dataset/validation/Dachshund/Dachshund_30.jpg
⠇ Processing images...2024-10-03 17:41:15 | INFO     | utils.logging_utils:wrapper:17 - Starting load_image
2024-10-03 17:41:15 | INFO     | utils.logging_utils:wrapper:20 - Finished load_image
2024-10-03 17:41:15 | INFO     | utils.logging_utils:wrapper:17 - Starting infer
⠏ Processing images...2024-10-03 17:41:15 | INFO     | utils.logging_utils:wrapper:20 - Finished infer
2024-10-03 17:41:15 | INFO     | utils.logging_utils:wrapper:17 - Starting save_prediction_image
⠋ Processing images...2024-10-03 17:41:16 | INFO     | utils.logging_utils:wrapper:20 - Finished save_prediction_image
Processed Dachshund_30.jpg: Dachshund (1.00)
File:/app/data/dogs_dataset/validation/Dachshund/Dachshund_41.jpg
⠋ Processing images...2024-10-03 17:41:16 | INFO     | utils.logging_utils:wrapper:17 - Starting load_image
2024-10-03 17:41:16 | INFO     | utils.logging_utils:wrapper:20 - Finished load_image
2024-10-03 17:41:16 | INFO     | utils.logging_utils:wrapper:17 - Starting infer
⠙ Processing images...2024-10-03 17:41:16 | INFO     | utils.logging_utils:wrapper:20 - Finished infer
⠸ Processing images...2024-10-03 17:41:16 | INFO     | utils.logging_utils:wrapper:17 - Starting save_prediction_image
⠼ Processing images...2024-10-03 17:41:16 | INFO     | utils.logging_utils:wrapper:20 - Finished save_prediction_image
Processed Dachshund_41.jpg: Dachshund (0.90)
File:/app/data/dogs_dataset/validation/Dachshund/Dachshund_67.jpg
....
....
File:/app/data/dogs_dataset/validation/Rottweiler/Rottweiler_87.jpg
⠼ Processing images...2024-10-03 17:41:35 | INFO     | utils.logging_utils:wrapper:17 - Starting load_image
2024-10-03 17:41:35 | INFO     | utils.logging_utils:wrapper:20 - Finished load_image
2024-10-03 17:41:35 | INFO     | utils.logging_utils:wrapper:17 - Starting infer
2024-10-03 17:41:35 | INFO     | utils.logging_utils:wrapper:20 - Finished infer
2024-10-03 17:41:35 | INFO     | utils.logging_utils:wrapper:17 - Starting save_prediction_image
⠧ Processing images...2024-10-03 17:41:35 | INFO     | utils.logging_utils:wrapper:20 - Finished save_prediction_image
Processed Rottweiler_87.jpg: Rottweiler (0.99)
File:/app/data/dogs_dataset/validation/Rottweiler/Rottweiler_75.jpg
⠧ Processing images...2024-10-03 17:41:35 | INFO     | utils.logging_utils:wrapper:17 - Starting load_image
2024-10-03 17:41:35 | INFO     | utils.logging_utils:wrapper:20 - Finished load_image
2024-10-03 17:41:35 | INFO     | utils.logging_utils:wrapper:17 - Starting infer
2024-10-03 17:41:35 | INFO     | utils.logging_utils:wrapper:20 - Finished infer
2024-10-03 17:41:35 | INFO     | utils.logging_utils:wrapper:17 - Starting save_prediction_image
⠏ Processing images...2024-10-03 17:41:35 | INFO     | utils.logging_utils:wrapper:20 - Finished save_prediction_image
Processed Rottweiler_75.jpg: Rottweiler (1.00)
File:/app/data/dogs_dataset/validation/Rottweiler/Rottweiler_91.jpg
⠏ Processing images...2024-10-03 17:41:35 | INFO     | utils.logging_utils:wrapper:17 - Starting load_image
2024-10-03 17:41:35 | INFO     | utils.logging_utils:wrapper:20 - Finished load_image
2024-10-03 17:41:35 | INFO     | utils.logging_utils:wrapper:17 - Starting infer
⠙ Processing images...2024-10-03 17:41:36 | INFO     | utils.logging_utils:wrapper:20 - Finished infer
2024-10-03 17:41:36 | INFO     | utils.logging_utils:wrapper:17 - Starting save_prediction_image
⠹ Processing images...2024-10-03 17:41:36 | INFO     | utils.logging_utils:wrapper:20 - Finished save_prediction_image
Processed Rottweiler_91.jpg: Rottweiler (1.00)
2024-10-03 17:41:36 | INFO     | utils.logging_utils:wrapper:20 - Finished main
```


# Tensorboard logs
```sh
tensorboard --logdir outputs/ --load_fast=false
```

