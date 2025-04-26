# Vision Transformer (ViT) - Hymenoptera Classification

This project trains a Vision Transformer (ViT) model for binary image classification: ants vs bees!

---

## What is ViT?

ViT stands for **Vision Transformer**, a deep learning model that applies the Transformer architecture (originally from NLP) directly to image patches instead of words.
- An image is split into small fixed-size patches
- Each patch is flattened and passed as a sequence to a standard Transformer Encoder
- Outperforms CNNs on large datasets when trained properly

Paper: ["An Image is Worth 16x16 Words"](https://arxiv.org/abs/2010.11929)

---

## Requirements
Install required libraries 

```bash
pip install -r requirements.txt
```

## How to Run

Clone the repository, install the dependencies, download the pretrained weights, set up the dataset, and run the training script.

```bash
git clone https://github.com/your-username/ViT-Hymenoptera-Classification.git

cd ViT-Hymenoptera-Classification

mkdir -p checkpoint

wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz -P checkpoint/

# (Prepare the hymenoptera_data folder structure with ants/ and bees/ inside train/ and val/ folders)

python src/train.py

```

## Contributor

Individual work by author

## Data Source

Kaggle

