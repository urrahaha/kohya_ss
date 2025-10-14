# SRPO Training Guide for Kohya_ss

## Overview

**SRPO (Style Reward Preference Optimization)** is a cutting-edge training technique that directly aligns diffusion models using reward models for preference learning. This integration brings SRPO to SDXL LoRA training in Kohya_ss.

Based on the research paper: [Directly Aligning the Full Diffusion Trajectory with Fine-Grained Human Preference](https://arxiv.org/abs/2509.06942)

Original implementation: [Tencent-Hunyuan/SRPO](https://github.com/Tencent-Hunyuan/SRPO)

## Key Features

### 🚀 Direct Alignment
- **No Traditional RL**: SRPO uses a novel sampling strategy that directly optimizes with analytical gradients
- **Faster Training**: Achieves visible improvements in ~10 minutes of training
- **Single Image Rollout**: More efficient than methods requiring multiple rollouts (e.g., GRPO)

### 🎯 Quality Improvements
- **Free of Reward Hacking**: Improved training strategy prevents common reward model exploitation
- **Direct Regularization**: Uses negative rewards without requiring KL divergence
- **Perceptual Quality**: Improves image quality without overfitting to color/saturation

### 🎨 Controllable Fine-tuning
- **Dynamic Text Conditions**: On-the-fly adjustment of reward preference toward specific styles
- **Control Words**: Automatically adds style control words (e.g., "Detailed", "Natural-lighting") during training

## Prerequisites

### 1. Install Required Dependencies

```bash
# Install HPS-v2 if using HPS reward model
pip install hpsv2

# Core dependencies (usually already installed with kohya_ss)
pip install torch torchvision transformers diffusers
```

### 2. Download Reward Model Checkpoints

#### For HPS-v2.1 (Recommended)
```bash
# Create directories
mkdir -p models/reward_models/hps_v2.1
mkdir -p models/reward_models/clip

# Download HPS checkpoint
huggingface-cli download xswu/HPSv2 HPS_v2.1_compressed.pt --local-dir models/reward_models/hps_v2.1

# Download CLIP checkpoint
huggingface-cli download laion/CLIP-ViT-H-14-laion2B-s32B-b79K open_clip_pytorch_model.bin --local-dir models/reward_models/clip
```

#### For PickScore (Optional)
```bash
mkdir -p models/reward_models/pickscore

# Download PickScore model
python scripts/huggingface/download_hf.py --repo_id yuvalkirstain/PickScore_v1 --local_dir models/reward_models/pickscore
```

## GUI Configuration

### Basic Setup

1. **Enable SRPO Training**
   - Navigate to the "SRPO Training" accordion in the GUI
   - Check "Enable SRPO Training"

2. **Select Reward Model**
   - Choose from: HPS, PickScore, or CLIP
   - **Recommended**: HPS-v2.1 for best results

### Core Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| **Timestep Length** | 100 | 10-1000 | Number of diffusion timesteps for training |
| **Guidance Scale** | 3.5 | 1.0-20.0 | CFG scale for sampling (higher = stronger guidance) |
| **Groundtruth Ratio** | 0.9 | 0.0-1.0 | Ratio for image recovery from noise |
| **Reward Threshold** | 0.7 | 0.0-2.0 | ReLU threshold to prevent reward hacking |

### Use Reward Model (NEW!)

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Use Reward Model** | True | Enable/disable reward model guidance. Disable for styles (e.g., anime) where existing models are biased |

**When disabled:** SRPO uses a preference-weighted denoising objective instead of external reward signals. This removes photorealism bias but may require more experimentation.

### Custom Control Words (NEW!)

**Control what style SRPO optimizes toward** by providing your own positive and negative preference words.

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Positive Controls** | "Natural-lighting, Detail, Detailed, Real" | Comma-separated words for desired qualities |
| **Negative Controls** | "Concept art, Painting, Anime, Flat, Oil" | Comma-separated words for undesired qualities |

**Examples by Style:**

**Anime Training:**
- Positive: `cel-shaded, vibrant colors, stylized, sharp lineart, anime aesthetic`
- Negative: `photorealistic, 3D render, realistic lighting, depth of field, hyperrealistic`

**Illustration/Painterly:**
- Positive: `artistic, painterly, expressive brushwork, traditional media, fine art`
- Negative: `photograph, raw photo, candid shot, smartphone photo, snapshot`

**Photorealism (Default):**
- Positive: `detailed, natural-lighting, real, photorealistic, high resolution`
- Negative: `flat, anime, painting, oily, cg artwork, stylized`

Leave blank to use hardcoded defaults (optimized for photorealism).

### Advanced Parameters

#### Discount Parameters
Control reward weighting across timesteps:

**Positive Branch** (rewards realistic/detailed styles):
- Start: 0.1
- End: 0.25

**Inversion Branch** (penalizes flat/oily textures):
- Start: 0.3
- End: 0.01

#### Training Timestep Range
Focus training on specific diffusion stages:
- **Start Timestep**: 5 (default)
- **End Timestep**: 25 (default)

⚠️ **Important**:
- Too early (>0.99): May cause structural distortions
- Too late: Encourages color-based reward hacking

## Training Recommendations

### 1. Batch Size
- **Recommended**: 32 or higher for best quality
- Larger batch sizes generally improve results
- Adjust based on available VRAM

### 2. Learning Rate
- **Range**: 1e-5 to 1e-6
- Start with 5e-6 for most models
- Lower learning rates for fine-tuning pretrained LoRAs

### 3. Training Steps
- **Quick results**: 20-50 steps (~10 minutes)
- **Full training**: 100-500 steps
- Monitor sample images to prevent overfitting

### 4. Memory Optimization
SRPO requires online generation, which uses more VRAM:
- Enable `gradient_checkpointing` to save memory
- Disable `cache_latents` (automatically disabled with SRPO)
- Use `mixed_precision` (bf16 or fp16)
- Consider `vae_batch_size` for large images

## Command Line Usage

### Basic SRPO Training

```bash
accelerate launch --num_cpu_threads_per_process=2 sdxl_train_network.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --dataset_config="dataset_config.toml" \
  --output_dir="./output/srpo_lora" \
  --output_name="my_lora_srpo" \
  --network_module="networks.lora" \
  --network_dim=32 \
  --network_alpha=16 \
  --train_batch_size=1 \
  --learning_rate=5e-6 \
  --max_train_steps=100 \
  --mixed_precision="bf16" \
  --gradient_checkpointing \
  --save_every_n_steps=50 \
  --srpo_enable \
  --srpo_reward_model="HPS" \
  --srpo_timestep_length=100 \
  --srpo_guidance_scale=3.5
```

### With Advanced Parameters

```bash
accelerate launch --num_cpu_threads_per_process=2 sdxl_train_network.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --dataset_config="dataset_config.toml" \
  --output_dir="./output/srpo_lora" \
  --output_name="my_lora_srpo" \
  --network_module="networks.lora" \
  --network_dim=32 \
  --network_alpha=16 \
  --train_batch_size=1 \
  --learning_rate=5e-6 \
  --max_train_steps=200 \
  --mixed_precision="bf16" \
  --gradient_checkpointing \
  --save_every_n_steps=50 \
  --srpo_enable \
  --srpo_reward_model="HPS" \
  --srpo_timestep_length=100 \
  --srpo_discount_pos 0.1 0.25 \
  --srpo_discount_inv 0.3 0.01 \
  --srpo_train_timestep 5 25 \
  --srpo_groundtruth_ratio=0.9 \
  --srpo_guidance_scale=3.5 \
  --srpo_reward_threshold=0.7 \
  --srpo_positive_controls "detailed" "natural-lighting" "real" "photorealistic" \
  --srpo_negative_controls "flat" "anime" "painting" "cg artwork"
```

### Training Anime/Illustration Styles

**Option 1: With Custom Control Words** (tries to guide reward model toward anime)
```bash
accelerate launch --num_cpu_threads_per_process=2 sdxl_train_network.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --dataset_config="dataset_config.toml" \
  --output_dir="./output/srpo_anime_lora" \
  --output_name="my_anime_lora_srpo" \
  --network_module="networks.lora" \
  --network_dim=32 \
  --network_alpha=16 \
  --train_batch_size=1 \
  --learning_rate=5e-6 \
  --max_train_steps=200 \
  --mixed_precision="bf16" \
  --gradient_checkpointing \
  --save_every_n_steps=50 \
  --srpo_enable \
  --srpo_use_reward_model \
  --srpo_reward_model="CLIP" \
  --srpo_timestep_length=100 \
  --srpo_discount_pos 0.1 0.25 \
  --srpo_discount_inv 0.3 0.01 \
  --srpo_train_timestep 5 25 \
  --srpo_groundtruth_ratio=0.9 \
  --srpo_guidance_scale=3.5 \
  --srpo_reward_threshold=0.7 \
  --srpo_positive_controls "cel-shaded" "vibrant colors" "stylized" "sharp lineart" \
  --srpo_negative_controls "photorealistic" "3D render" "realistic lighting" "depth of field"
```

**Option 2: Without Reward Model** (removes photorealism bias entirely)
```bash
accelerate launch --num_cpu_threads_per_process=2 sdxl_train_network.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --dataset_config="dataset_config.toml" \
  --output_dir="./output/srpo_anime_lora_no_reward" \
  --output_name="my_anime_lora_srpo_no_reward" \
  --network_module="networks.lora" \
  --network_dim=32 \
  --network_alpha=16 \
  --train_batch_size=1 \
  --learning_rate=5e-6 \
  --max_train_steps=200 \
  --mixed_precision="bf16" \
  --gradient_checkpointing \
  --save_every_n_steps=50 \
  --srpo_enable \
  --srpo_timestep_length=100 \
  --srpo_discount_pos 0.1 0.25 \
  --srpo_discount_inv 0.3 0.01 \
  --srpo_train_timestep 5 25 \
  --srpo_groundtruth_ratio=0.9 \
  --srpo_guidance_scale=3.5
```
*Note: `--srpo_use_reward_model` is omitted to disable reward model. Control words are not used when reward model is disabled.*

## Troubleshooting

### Issue: CUDA Out of Memory

**Solutions**:
1. Enable gradient checkpointing: `--gradient_checkpointing`
2. Reduce batch size: `--train_batch_size=1`
3. Use smaller network: `--network_dim=16`
4. Enable CPU offloading if needed

### Issue: Reward Hacking (Oversaturated/Unnatural Colors)

**Solutions**:
1. Increase `srpo_reward_threshold` (try 0.8 or 0.9)
2. Adjust discount parameters to reduce aggressive optimization
3. Train for fewer steps
4. Use different reward model (try PickScore instead of HPS)

### Issue: Structural Distortions

**Solutions**:
1. Adjust `srpo_train_timestep` range (avoid very early timesteps)
2. Start range: 5-10 (not too early)
3. Reduce `srpo_guidance_scale` (try 2.5-3.0)
4. Check `srpo_groundtruth_ratio` is not too low

### Issue: Slow Training

**Solutions**:
1. SRPO does online generation, which is slower than cached training
2. Use `--vae_batch_size` to batch VAE operations
3. Ensure `cache_text_encoder_outputs` is enabled (if not using dynamic prompts)
4. Use fewer gradient accumulation steps

## Comparison: SRPO vs Standard Training

| Aspect | Standard Training | SRPO Training |
|--------|------------------|---------------|
| **Training Speed** | Fast (cached latents) | Slower (online generation) |
| **Quality Improvement** | Depends on dataset | Direct perceptual alignment |
| **Memory Usage** | Lower | Higher (online rollout) |
| **Dataset Requirements** | Many high-quality images | Can work with fewer images |
| **Overfitting Risk** | Higher | Lower (regularized) |
| **Fine Control** | Limited | Dynamic style control |

## Best Practices

1. **Start Simple**: Begin with default parameters and HPS reward model
2. **Monitor Samples**: Generate sample images frequently to catch issues early
3. **Gradual Adjustment**: Make small parameter changes and observe effects
4. **Mix with Standard**: Consider training with standard method first, then fine-tune with SRPO
5. **Dataset Quality**: While SRPO can work with fewer images, quality still matters
6. **Experiment**: SRPO parameters are research-grade; experimentation is encouraged

## Technical Details

### Direct-Align Strategy

SRPO uses a three-step process for each training iteration:

1. **Online Rollout**: Generate image from noise using current model
2. **Inject Noise**: Add controlled noise at intermediate timestep
3. **Inverse/Denoise**: Take one gradient-enabled diffusion step
4. **Recover Image**: Decode latents and compute reward
5. **Backpropagate**: Update model based on reward signal

### Reward Model Integration

SRPO supports multiple reward models:

- **HPS-v2.1**: Human Preference Score, trained on large-scale human ratings
- **PickScore**: Alternative preference model with different training data
- **CLIP**: Standard CLIP similarity (less accurate but widely available)

## References

- Paper: [Directly Aligning the Full Diffusion Trajectory with Fine-Grained Human Preference](https://arxiv.org/abs/2509.06942)
- Original Code: [Tencent-Hunyuan/SRPO](https://github.com/Tencent-Hunyuan/SRPO)
- Project Page: [https://tencent.github.io/srpo-project-page/](https://tencent.github.io/srpo-project-page/)

## Support and Feedback

For issues specific to SRPO integration in Kohya_ss:
- Open an issue on the Kohya_ss GitHub repository
- Provide logs, configuration, and sample images when possible
- Tag issues with "SRPO" for easier tracking

For questions about the SRPO method itself:
- Refer to the original paper and repository
- Check the SRPO project page for examples and explanations

---

**Happy Training! 🎨✨**
