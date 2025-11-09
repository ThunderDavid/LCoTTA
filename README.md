# LCoTTA

This is Official implementation of the NeurIPS 2025 paper“Lifelong Test-Time Adaptation via Online Learning in  Tracked Low-Dimensional Subspace” based on PyTorch. Our subspace-based method achieves robust and stable performance in long-term unsupervised test-time adaptation tasks, without resetting or accessing source domain data in any form.

***The framework of this repository is built upon [this codebase](https://github.com/mariodoebler/test-time-adaptation/blob/main/README.md), while our method is independently developed.**
## Classification  
This repository contains a wide range of TTA methods for comparing classification performance on the ImageNet-C dataset under long-term adaptation scenarios (e.g., 50×15 epochs). A brief overview of the repository's main features is provided below:

### Datasets  
- ImageNet-C: The open-source ImageNet-C dataset can be downloaded [here](https://zenodo.org/records/2235448#.Yj2RO_co_mF).

### Models  
- ResNet-50: The pre-trained weights for ResNet-50 are provided in `/classification/ckpt/`, so no additional download is required.  
- ViT-B/16: The pre-trained weights for ViT are obtained using the [timm library](https://github.com/huggingface/pytorch-image-models).

### Get Started  

1. First, download the dataset and set the dataset path in configration file.  
2. Run the following command to start testing:
```
test_time.py --cfg cfgs/[architecture]/imagenet_c/[methods].yaml
```
Here, `[methods]` refers to the TTA method configuration file and `[architecture]` is the model name, which can be either `R50` or `ViT`.
