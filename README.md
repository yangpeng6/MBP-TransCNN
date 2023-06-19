# MBP-TransCNN
This repo holds code for [A Deep Learning Approach for Automated Segmentation of Magnetic Bright Points in the Solar Photosphere](https://github.com/yangpeng6/MBP-TransCNN)

## Usage

### 1. Download Google pre-trained ViT models
* Please access the following link to obtain the models: (https://drive.google.com/drive/folders/1susnn0cs_a1W8QTMGQUpieQMzkDdlEX8?usp=sharing). Download the models and copy them to the "model" folder.

### 2. Prepare data

Please go to ["./datasets/README.md"](datasets_n/README.md) for details.You can download the training and testing datasets from the following link: (https://drive.google.com/drive/folders/1KPda-LbK4u8I5ya_szdvXhnPZoIM9wVp?usp=sharing).

### 3. Environment

Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

### 4. Train/Test

- Run the train script on MBP dataset. The batch size can be reduced to 2 or 1 to save memory (please also decrease the base_lr linearly), and both can reach similar performance.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset MBP --vit_name R50-ViT-B_16
```

- Run the test script on MBP dataset.

```bash
python test.py --dataset MBP --vit_name ./model/TU_pretrain_R50-ViT-B_16_skip3_epo100_bs1_640/epoch_99.pth
```
```

## Reference
* [TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation](https://arxiv.org/pdf/2102.04306.pdf)
* [Google ViT](https://github.com/google-research/vision_transformer)
* [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
* [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)

