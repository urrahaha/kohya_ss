#!/bin/bash
# Download SRPO Reward Models Script
# This script downloads the required reward models for SRPO training

set -e

echo "============================================"
echo "SRPO Reward Models Download Script"
echo "============================================"
echo ""

# Create directories
echo "Creating directories..."
mkdir -p models/reward_models/hps_v2.1
mkdir -p models/reward_models/clip
mkdir -p models/reward_models/pickscore

echo "✓ Directories created"
echo ""

# Download HPS-v2.1 (required)
echo "Downloading HPS-v2.1 reward model (required)..."
echo "This may take a few minutes..."
huggingface-cli download xswu/HPSv2 HPS_v2.1_compressed.pt --local-dir models/reward_models/hps_v2.1

echo "✓ HPS-v2.1 downloaded"
echo ""

# Download CLIP (required for HPS)
echo "Downloading CLIP model (required for HPS)..."
huggingface-cli download laion/CLIP-ViT-H-14-laion2B-s32B-b79K open_clip_pytorch_model.bin --local-dir models/reward_models/clip

echo "✓ CLIP model downloaded"
echo ""

# Optional: Download PickScore
read -p "Download PickScore model? (optional, y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Downloading PickScore model..."
    huggingface-cli download yuvalkirstain/PickScore_v1 --local-dir models/reward_models/pickscore
    echo "✓ PickScore downloaded"
else
    echo "Skipping PickScore download"
fi

echo ""
echo "============================================"
echo "✓ Download complete!"
echo "============================================"
echo ""
echo "Required models installed:"
echo "  - HPS-v2.1: models/reward_models/hps_v2.1/"
echo "  - CLIP: models/reward_models/clip/"
echo ""
echo "You can now enable SRPO training in the GUI or command line."
echo "See docs/SRPO_Quick_Start.md for usage instructions."
echo ""
