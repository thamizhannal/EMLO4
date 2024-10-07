## Dog Breed Classifier
The objective is to use the Kaggle Dog Breed image dataset and train it using pre-existing resnet18 and classify images.

### DataSet Description
This dataset contains a collection of images for 10 different dog breeds, meticulously gathered and organized to facilitate various computer vision tasks such as image classification and object detection. The dataset includes the following breeds:

Golden Retriever
German Shepherd
Labrador Retriever
Bulldog
Beagle
Poodle
Rottweiler
Yorkshire Terrier
Boxer
Dachshund


### Download kaggle dataset using CLI
Install the kaggle-cli and download the dog-breed-image dataset using below commands
```bash
$pip install kaggle-cli
$kaggle datasets download khushikhushikhushi/dog-breed-image-dataset
```
### Docker-compose.yml

``` yml
version: '3.8'

services:
  base:
    build:
      context: .
      dockerfile: Dockerfile
    shm_size: "2gb"
    volumes:
      - ./data:/app/data
      - ./src:/app/src
      - ./checkpoints:/app/checkpoints
      - ./lightning_logs:/app/lightning_logs
    environment:
      - PYTHONUNBUFFERED=1
  train:
    extends:
      service: base
    command: python /app/src/train.py
    volumes:
      - dogbreedclassifier:/app/src
      - checkpoints:/app/checkpoints
      - data:/app/data    
  
  evaluate:
    extends:
      service: base
    command: python /app/src/eval.py
    volumes:
      - dogbreedclassifier:/app/src
      - checkpoints:/app/checkpoints
      - data:/app/data
  infer:
    extends:
      service: base
    command: python /app/src/infer.py
    volumes:
      - dogbreedclassifier:/app/src
      - checkpoints:/app/checkpoints
      - output:/app/data/output

volumes:
  dogbreedclassifier:
  data:
  checkpoints:
  output:

```

### Docker Build

####  $ docker compose build . 

``` bash

(py3.10) (baseDogCatImgClassifier % docker compose build .       
WARN[0000] /Users/vehere/mlworks/EMLO4/code/EMLO4/DogCatImgClassifier/docker-compose.yml: the attribute `version` is obsolete, it will be ignored, please remove it to avoid potential confusion 
no such service: .
(py3.10) (base) vehere@veheres-MacBook-Pro-3 DogCatImgClassifier % docker compose build  
WARN[0000] /Users/vehere/mlworks/EMLO4/code/EMLO4/DogCatImgClassifier/docker-compose.yml: the attribute `version` is obsolete, it will be ignored, please remove it to avoid potential confusion 
[+] Building 8.9s (28/40)                                                                                                                                                  docker:desktop-linux
 => [train internal] load build definition from Dockerfile                                                                                                                                 0.0s
 => => transferring dockerfile: 263B                                                                                                                                                       0.0s
 => [infer internal] load build definition from Dockerfile                                                                                                                                 0.0s
 => => transferring dockerfile: 263B                                                                                                                                                       0.0s
 => [evaluate internal] load build definition from Dockerfile                                                                                                                              0.0s
 => => transferring dockerfile: 263B                                                                                                                                                       0.0s
 => [base internal] load build definition from Dockerfile                                                                                                                                  0.0s
 => => transferring dockerfile: 263B                                                                                                                                                       0.0s
 => [train internal] load metadata for docker.io/library/python:3.12.0-slim                                                                                                                1.9s
 => [train auth] library/python:pull token for registry-1.docker.io                                                                                                                        0.0s
 => [base internal] load .dockerignore                                                                                                                                                     0.0s
 => => transferring context: 2B                                                                                                                                                            0.0s
 => [infer internal] load .dockerignore                                                                                                                                                    0.0s
 => => transferring context: 2B                                                                                                                                                            0.0s
 => [train internal] load .dockerignore                                                                                                                                                    0.0s
 => => transferring context: 2B                                                                                                                                                            0.0s
 => [evaluate internal] load .dockerignore                                                                                                                                                 0.0s
 => => transferring context: 2B                                                                                                                                                            0.0s
 => [evaluate 1/6] FROM docker.io/library/python:3.12.0-slim@sha256:19a6235339a74eca01227b03629f63b6f5020abc21142436eced6ec3a9839a76                                                       0.1s
 => => resolve docker.io/library/python:3.12.0-slim@sha256:19a6235339a74eca01227b03629f63b6f5020abc21142436eced6ec3a9839a76                                                                0.0s
 => [base internal] load build context                                                                                                                                                     0.1s
 => => transferring context: 87.04kB                                                                                                                                                       0.1s
 => [infer internal] load build context                                                                                                                                                    0.1s
 => => transferring context: 87.04kB                                                                                                                                                       0.1s
 => [evaluate internal] load build context                                                                                                                                                 0.1s
 => => transferring context: 87.04kB                                                                                                                                                       0.1s
 => [train internal] load build context                                                                                                                                                    0.1s
 => => transferring context: 87.04kB                                                                                                                                                       0.0s
 => CACHED [evaluate 2/6] WORKDIR /app                                                                                                                                                     0.0s
 => CACHED [evaluate 3/6] COPY requirements.txt /app/                                                                                                                                      0.0s
 => CACHED [evaluate 4/6] COPY data /app/                                                                                                                                                  0.0s
 => CACHED [train 5/6] RUN pip3 install --no-cache-dir -r requirements.txt                                                                                                                 0.0s
 => [evaluate 6/6] COPY . .                                                                                                                                                                1.1s
 => [train] exporting to image                                                                                                                                                             5.6s
 => => exporting layers                                                                                                                                                                    4.4s
 => => exporting manifest sha256:d2d98858b713b13274aa8b9ec3b9820d20f864ee32b97a1bb137c5ef2332fac8                                                                                          0.0s
 => => exporting config sha256:fd87c5060a965d82690836183f7485cf48fe12b95effe2f149bfbc1d6469baa0                                                                                            0.0s
 => => exporting attestation manifest sha256:7094cee46a761046b3ec75300a10663e4a6301e52aaf2175c23e4e23ad55dd3c                                                                              0.0s
 => => exporting manifest list sha256:73a8589e2137c7360b6e7dcc1518757aed9f49a2ebbff942a1d1ef9d4f12085f                                                                                     0.0s
 => => naming to docker.io/library/dogcatimgclassifier-train:latest                                                                                                                        0.0s
 => => unpacking to docker.io/library/dogcatimgclassifier-train:latest                                                                                                                     1.1s
 => [base] exporting to image                                                                                                                                                              5.6s
 => => exporting layers                                                                                                                                                                    4.4s
 => => exporting manifest sha256:32ab4964b7869616eb03a55d0a221bbb0500ddcd009326189f231c6a60ed791d                                                                                          0.0s
 => => exporting config sha256:7fbf0c6b2bdfe7776c98e19366cc008b4ab5f511e130b587f223504a4b0066bf                                                                                            0.0s
 => => exporting attestation manifest sha256:050c75a156d6288dcb7149515a9605eff349ca1424a2fd1f47eb1d164c6e95f0                                                                              0.0s
 => => exporting manifest list sha256:d5b592bb9dfa6e5010dcacebac5a786df112270a627afee88a8a06391b55d7b8                                                                                     0.0s
 => => naming to docker.io/library/dogcatimgclassifier-base:latest                                                                                                                         0.0s
 => => unpacking to docker.io/library/dogcatimgclassifier-base:latest                                                                                                                      1.1s
 => [infer] exporting to image                                                                                                                                                             5.6s
 => => exporting layers                                                                                                                                                                    4.4s
 => => exporting manifest sha256:90bf043df3f2fe1948d1ab44ae5fe98cde5d00af8926ce92d3de3bbf934ba9b7                                                                                          0.0s
 => => exporting config sha256:310860297b5c5299e857bcaddf5c7b837c3862837ed6b4d7964bbf21c4ab50bf                                                                                            0.0s
 => => exporting attestation manifest sha256:06d7222beb6503b3e921eb9bb5b0d981c831ceb63b351dc3ff588839325b55d9                                                                              0.0s
 => => exporting manifest list sha256:48384acece9b4d7c8c1e866791bc3505e5fb59404f9b8986dd8398e0e6b8997e                                                                                     0.0s
 => => naming to docker.io/library/dogcatimgclassifier-infer:latest                                                                                                                        0.0s
 => => unpacking to docker.io/library/dogcatimgclassifier-infer:latest                                                                                                                     1.1s
 => [evaluate] exporting to image                                                                                                                                                          5.7s
 => => exporting layers                                                                                                                                                                    4.4s
 => => exporting manifest sha256:76c1d7b2d61b35ea5f528e826be5949af2ea67a18cd7103a8534d1b6626c721f                                                                                          0.0s
 => => exporting config sha256:b7b913a88e18b751059ea8f6c54d5072ceec6aeb67c7bff7c06c46f0240fa9a6                                                                                            0.0s
 => => exporting attestation manifest sha256:d3ba9fb70385912058349de0a536c80a2940840d36d65ab7159bf95de169300f                                                                              0.0s
 => => exporting manifest list sha256:6c78d43ff88b5ea9e182d65ea2a4aab1fc04b430f9c3b74f4347d3b4af8a60b9                                                                                     0.0s
 => => naming to docker.io/library/dogcatimgclassifier-evaluate:latest                                                                                                                     0.0s
 => => unpacking to docker.io/library/dogcatimgclassifier-evaluate:latest                                                                                                                  1.1s
 => [infer] resolving provenance for metadata file                                                                                                                                         0.0s
 => [base] resolving provenance for metadata file                                                                                                                                          0.0s
 => [train] resolving provenance for metadata file                                                                                                                                         0.0s
 => [evaluate] resolving provenance for metadata file                                                                                                                                      0.0s
```
### Docker Train

#### docker compose run -it train 

``` bash
(py3.10) (base) $ docker compose run -it train 
WARN[0000] /Users/vehere/mlworks/EMLO4/code/EMLO4/DogCatImgClassifier/docker-compose.yml: the attribute `version` is obsolete, it will be ignored, please remove it to avoid potential confusion 
WARN[0000] Found orphan containers ([dogcatimgclassifier-train-run-8c4135a48a5a dogcatimgclassifier-train-run-9f7f9f35e068 dogcatimgclassifier-train-run-764a2a8788c5]) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up. 
Seed set to 42
ds=967, train=773, test=98, val=98
ds=967, train=773, test=98, val=98
Model fit called!
Initialize data module completed!
Num of the classes:10
Hyper parameters saved!
model.safetensors: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 46.8M/46.8M [00:13<00:00, 3.51MB/s]
Resnet18 model created!
DogBreedClassifier __init__ completed!
Initialize DogBreedClassifier completed!
checkpoint callback completed!
GPU available: False, used: False
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
Initialize Trainer completed!
ds=967, train=773, test=98, val=98
ds=967, train=773, test=98, val=98
Model fit called!
/usr/local/lib/python3.12/site-packages/lightning/pytorch/callbacks/model_checkpoint.py:654: Checkpoint directory /app/checkpoints exists and is not empty.

  | Name      | Type               | Params | Mode 
---------------------------------------------------------
0 | model     | ResNet             | 11.2 M | train
1 | train_acc | MulticlassAccuracy | 0      | train
2 | val_acc   | MulticlassAccuracy | 0      | train
3 | test_acc  | MulticlassAccuracy | 0      | train
---------------------------------------------------------
11.2 M    Trainable params
0         Non-trainable params
11.2 M    Total params
44.727    Total estimated model params size (MB)
97        Modules in train mode
0         Modules in eval mode
Epoch 0: 100%|██████████████| 25/25 [01:54<00:00,  0.22it/s, v_num=0, train_loss_step=1.310, train_acc_step=0.400, val_loss=0.776, val_acc=0.885, train_loss_epoch=1.690, train_acc_epoch=0.599]Epoch 0, global step 25: 'val_loss' reached 0.77609 (best 0.77609), saving model to '/app/checkpoints/best-checkpoint-v1.ckpt' as top 1                                                         
Epoch 1:  16%|▏| 4/25 [00:22<01:58,  0.18it/s, v_num=0, train_loss_step=0.733, train_acc_step=0.906, val_loss=0.776, val_acc=0.885,            Epoch 1: 100%|█| 25/25 [01:57<00:00,  0.21it/s, v_num=0, train_loss_step=0.548, train_acc_step=0.800, val_loss=0.127, val_acc=0.958, train_loss_epoch=0.391, train_acc_eEpoch 1, global step 50: 'val_loss' reached 0.12678 (best 0.12678), saving model to '/app/checkpoints/best-checkpoint-v1.ckpt' as top 1                                 
Epoch 2: 100%|█| 25/25 [01:51<00:00,  0.22it/s, v_num=0, train_loss_step=0.817, train_acc_step=0.800, val_loss=0.0582, val_acc=0.990, train_loss_epoch=0.0856, train_accEpoch 2, global step 75: 'val_loss' reached 0.05822 (best 0.05822), saving model to '/app/checkpoints/best-checkpoint-v1.ckpt' as top 1                                 
Epoch 3:  20%|▏| 5/25 [00:25<01:41,  0.20it/s, v_num=0, train_loss_step=0.0214, train_acc_step=1.000, val_loss=0.0582, val_acc=0.990, tEpoch 3: 100%|█| 25/25 [01:56<00:00,  0.21it/s, v_num=0, train_loss_step=0.111, train_acc_step=1.000, val_loss=0.017, val_acc=1.000, trEpoch 3, global step 100: 'val_loss' reached 0.01703 (best 0.01703), saving model to '/app/checkpoints/best-checkpoint-v1.ckpt' as top 1
Epoch 4: 100%|█| 25/25 [02:08<00:00,  0.19it/s, v_num=0, train_loss_step=0.152, train_acc_step=1.000, val_loss=0.0194, val_acc=0.990, train_loss_epoch=0.019Epoch 4, global step 125: 'val_loss' was not in top 1                                                                                                       
`Trainer.fit` stopped: `max_epochs=5` reached.
Epoch 4: 100%|█| 25/25 [02:08<00:00,  0.19it/s, v_num=0, train_loss_step=0.152, train_acc_step=1.000, val_loss=0.0194, val_acc=0.990, train_loss_epoch=0.019
Train.fit completed!
ds=967, train=773, test=98, val=98
ds=967, train=773, test=98, val=98
Testing DataLoader 0: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:02<00:00,  1.66it/s]
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
       Test metric             DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
        test_acc                    1.0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Best model path: /app/checkpoints/best-checkpoint-v1.ckpt
(py3.10) (base)  DogCatImgClassifier %

```

### Docker Evaludate

####  $ docker compose run -it evaluate

```bash
(py3.10) (base)$  docker compose run -it evaluate
WARN[0000] /Users/vehere/mlworks/EMLO4/code/EMLO4/DogCatImgClassifier/docker-compose.yml: the attribute `version` is obsolete, it will be ignored, please remove it to avoid potential confusion 
WARN[0000] Found orphan containers ([dogcatimgclassifier-evaluate-run-9eec5daf6b4d dogcatimgclassifier-train-run-db0eed9364b6 dogcatimgclassifier-train-run-8c4135a48a5a dogcatimgclassifier-train-run-9f7f9f35e068 dogcatimgclassifier-train-run-764a2a8788c5]) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up. 
ds=967, train=773, test=98, val=98
ds=967, train=773, test=98, val=98
Hyper parameters saved!
model.safetensors: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 46.8M/46.8M [00:07<00:00, 6.35MB/s]
Resnet18 model created!
DogBreedClassifier __init__ completed!
GPU available: False, used: False
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
ds=967, train=773, test=98, val=98
ds=967, train=773, test=98, val=98
Testing DataLoader 0: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:02<00:00,  1.40it/s]
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
       Test metric             DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
        test_acc                    1.0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
[{'test_acc': 1.0}]
(py3.10) (base) vehere@veheres-MacBook-Pro-3 extracted % 
```
### Docker Infer

#### $ docker compose run -it infer     

``` bash
(py3.10) $ docker compose run -it infer
WARN[0000] /Users/vehere/mlworks/EMLO4/code/EMLO4/DogCatImgClassifier/docker-compose.yml: the attribute `version` is obsolete, it will be ignored, please remove it to avoid potential confusion 
WARN[0000] Found orphan containers ([dogcatimgclassifier-infer-run-e276edf849be dogcatimgclassifier-infer-run-bd2048cdcc2f dogcatimgclassifier-infer-run-5d801f2df0d5 dogcatimgclassifier-infer-run-f667c0ffaefc dogcatimgclassifier-infer-run-0152a6ed8143 dogcatimgclassifier-infer-run-49db6f5559ec dogcatimgclassifier-infer-run-2248f21ace5b dogcatimgclassifier-infer-run-7f964aafe5ad dogcatimgclassifier-infer-run-6f6c68d36170 dogcatimgclassifier-infer-run-b2ced9def98a dogcatimgclassifier-infer-run-a5052cbee555 dogcatimgclassifier-base-run-3fbaab6e1d0e dogcatimgclassifier-infer-run-418c2faf5df9 dogcatimgclassifier-infer-run-35503f5c6fdc dogcatimgclassifier-infer-run-ee8f638bcfc5 dogcatimgclassifier-infer-run-f9e9736db827 dogcatimgclassifier-infer-run-fe0a4d354431 dogcatimgclassifier-infer-run-1c4820c4916b dogcatimgclassifier-evaluate-run-7851f9aa440d dogcatimgclassifier-train-run-17f8da830106 dogcatimgclassifier-infer-run-f437fe299271 dogcatimgclassifier-infer-run-9b24039a74a9 dogcatimgclassifier-evaluate-run-0cc56812ba98 dogcatimgclassifier-train-run-6ed805cfc898 dogcatimgclassifier-infer-run-f638fedb3101 dogcatimgclassifier-infer-run-742e7815c0bc dogcatimgclassifier-infer-run-7d49dbe70d98 dogcatimgclassifier-infer-run-2a78af649841 dogcatimgclassifier-infer-run-db9245fd604b dogcatimgclassifier-infer-run-057e0a4c8163 dogcatimgclassifier-infer-run-1a76ebb7d5d1 dogcatimgclassifier-infer-run-6075221c8404 dogcatimgclassifier-infer-run-73e4d1c6092c dogcatimgclassifier-infer-run-21e0ad8ec356 dogcatimgclassifier-infer-run-530f25bbd453 dogcatimgclassifier-infer-run-e58d29051bb0 dogcatimgclassifier-infer-run-0c9b9cd9fdb3 dogcatimgclassifier-infer-run-537813893dd2 dogcatimgclassifier-infer-run-fbe495ea8c66 dogcatimgclassifier-infer-run-6129942020b6 dogcatimgclassifier-evaluate-run-63d998ee3ffb dogcatimgclassifier-evaluate-run-9eec5daf6b4d dogcatimgclassifier-train-run-db0eed9364b6 dogcatimgclassifier-train-run-8c4135a48a5a dogcatimgclassifier-train-run-9f7f9f35e068 dogcatimgclassifier-train-run-764a2a8788c5]) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up. 
ds=967, train=773, test=98, val=98
ds=967, train=773, test=98, val=98
Model fit called!
num classes:10
Hyper parameters saved!
model.safetensors: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 46.8M/46.8M [00:11<00:00, 3.95MB/s]
Resnet18 model created!
DogBreedClassifier __init__ completed!
Model loaded sucessfully!
image_files=10
Predicted: Rottweiler (Confidence: 1.00)
output_file:/app/data/extracted/val_input/Rottweiler/Rottweiler_8.jpg
Predicted: Dachshund (Confidence: 1.00)
output_file:/app/data/extracted/val_input/Dachshund/Dachshund_8.jpg
Predicted: Yorkshire_Terrier (Confidence: 0.99)
output_file:/app/data/extracted/val_input/Yorkshire_Terrier/Yorkshire Terrier_8.jpg
Predicted: Beagle (Confidence: 1.00)
output_file:/app/data/extracted/val_input/Beagle/Beagle_8.jpg
Predicted: Golden_Retriever (Confidence: 0.99)
output_file:/app/data/extracted/val_input/Golden_Retriever/Golden Retriever_8.jpg
Predicted: Poodle (Confidence: 0.97)
output_file:/app/data/extracted/val_input/Poodle/Poodle_8.jpg
Predicted: Labrador_Retriever (Confidence: 0.97)
output_file:/app/data/extracted/val_input/Labrador_Retriever/Labrador Retriever_9.jpg
Predicted: German_Shepherd (Confidence: 1.00)
output_file:/app/data/extracted/val_input/German_Shepherd/German Shepherd_8.jpg
Predicted: Boxer (Confidence: 1.00)
output_file:/app/data/extracted/val_input/Boxer/Boxer_8.jpg
Predicted: Bulldog (Confidence: 0.99)
output_file:/app/data/extracted/val_input/Bulldog/Bulldog_8.jpg
(py3.10) (base) vehere@veheres-MacBook-Pro-3 DogCatImgClassifier % 


```
