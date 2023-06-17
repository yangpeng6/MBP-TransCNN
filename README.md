# MBP-TransCNN
This repo holds code for [A Deep Learning Approach for Automated Segmentation of Magnetic Bright Points in the Solar Photosphere](https://github.com/yangpeng6/MBPs-dataset)

## Usage

### 1. Download Google pre-trained ViT models
* [Get models in this link](https://console.cloud.google.com/storage/vit_models/): R50-ViT-B_16, ViT-B_16, ViT-L_16...
```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz &&
mkdir ../model/vit_checkpoint/imagenet21k &&
mv {MODEL_NAME}.npz ../model/vit_checkpoint/imagenet21k/{MODEL_NAME}.npz
```

### 2. Prepare data

Please go to ["./datasets/README.md"](datasets_n/README.md) for details, or please send an Email to jienengchen01 AT gmail.com to request the preprocessed data. If you would like to use the preprocessed data, please use it for research purposes and do not redistribute it.

### 3. Environment

Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

### 4. Train/Test

- Run the train script on MBP dataset. The batch size can be reduced to 2 or 1 to save memory (please also decrease the base_lr linearly), and both can reach similar performance.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset MBP --vit_name R50-ViT-B_16
```

- Run the test script on MBP dataset.

```bash
python test.py --dataset MBP --vit_name R50-ViT-B_16
```

## Reference
* [TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation](https://arxiv.org/pdf/2102.04306.pdf)
* [Google ViT](https://github.com/google-research/vision_transformer)
* [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
* [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)

