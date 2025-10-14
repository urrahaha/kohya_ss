# SRPO Quick Start Guide

## What is SRPO?

**SRPO (Style Reward Preference Optimization)** directly aligns SDXL LoRA models with human preferences using reward models. It can improve perceptual quality in ~10 minutes of training.

## Setup (5 minutes)

### 1. Install Dependencies

```bash
pip install hpsv2
```

### 2. Download Reward Models

```bash
# Create directories
mkdir -p models/reward_models/hps_v2.1
mkdir -p models/reward_models/clip

# Download HPS (recommended)
huggingface-cli download xswu/HPSv2 HPS_v2.1_compressed.pt --local-dir models/reward_models/hps_v2.1
huggingface-cli download laion/CLIP-ViT-H-14-laion2B-s32B-b79K open_clip_pytorch_model.bin --local-dir models/reward_models/clip
```

## GUI Training (Easiest)

1. Open Kohya_ss GUI
2. Go to "LoRA" tab
3. Configure your model and dataset as normal
4. Scroll to **"SRPO Training"** accordion
5. Check **"Enable SRPO Training"**
6. Select **"HPS"** as reward model
7. Use default parameters (they work well!)
8. Start training

## Command Line Training

```bash
accelerate launch sdxl_train_network.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --dataset_config="dataset.toml" \
  --output_dir="./output/srpo_lora" \
  --network_module="networks.lora" \
  --network_dim=32 \
  --network_alpha=16 \
  --learning_rate=5e-6 \
  --max_train_steps=100 \
  --mixed_precision="bf16" \
  --gradient_checkpointing \
  --srpo_enable \
  --srpo_reward_model="HPS"
```

## Key Benefits

✅ **Faster**: Visible improvements in ~10 minutes  
✅ **Better Quality**: Direct perceptual alignment  
✅ **No Reward Hacking**: Proper regularization prevents artifacts  
✅ **Controllable**: Dynamic style preference control  

## Customize for Your Style (NEW!)

### Want to train anime instead of photorealism?

**Option 1: Disable Reward Model** (recommended for anime - removes bias completely)

In the GUI:
1. Uncheck **"Use Reward Model"**
2. Train as normal

Command line:
```bash
--srpo_enable
# Don't include --srpo_use_reward_model flag
```

**Option 2: Keep Reward Model + Custom Control Words** (tries to guide toward anime)

In the GUI, expand **"Custom Control Words"** and enter:

**Positive:** `cel-shaded, vibrant colors, stylized, sharp lineart`  
**Negative:** `photorealistic, 3D render, realistic lighting, depth of field`

Or command line:
```bash
--srpo_use_reward_model \
--srpo_positive_controls "cel-shaded" "vibrant colors" "stylized" \
--srpo_negative_controls "photorealistic" "3D render" "realistic"
```

## Important Notes

⚠️ **VRAM**: SRPO uses more memory (online generation)  
⚠️ **Speed**: Slower than cached training but much faster than traditional RL  
⚠️ **Compatibility**: Works best with SDXL LoRA training  
✨ **NEW**: Custom control words let you target any style!

## Need Help?

📖 **Full Documentation**: See `SRPO_Training_Guide.md`  
🐛 **Issues**: Open GitHub issue with "SRPO" tag  
📄 **Research**: [arXiv:2509.06942](https://arxiv.org/abs/2509.06942)  

---

**Start with defaults, they work great! Happy training! 🎨**
