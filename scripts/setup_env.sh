#!/bin/bash

echo "Cloning ViT-PyTorch repository..."
git clone https://github.com/jeonsworld/ViT-pytorch.git
cd ViT-pytorch/

echo "Downloading pretrained weights..."
mkdir -p checkpoint
wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz -P checkpoint/

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Environment setup complete!"
