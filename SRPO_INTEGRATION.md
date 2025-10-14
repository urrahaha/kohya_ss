# SRPO Integration for Kohya_ss

## Summary

This integration adds **SRPO (Style Reward Preference Optimization)** training support to Kohya_ss, specifically for **SDXL LoRA training**. SRPO directly aligns diffusion models with human preferences using reward models, achieving quality improvements faster than traditional methods.

## What's New

### Core Components

1. **Reward Models** (`sd-scripts/library/srpo_reward_models.py`)
   - HPS-v2.1 (Human Preference Score)
   - PickScore
   - CLIP
   - Factory function for easy model selection

2. **Training Utilities** (`sd-scripts/library/srpo_train_utils.py`)
   - SDXL-specific SRPO training loop
   - Direct-Align strategy implementation
   - Online rollout and reward computation
   - Gradient-based optimization

3. **Training Script Integration** (`sd-scripts/train_network.py`)
   - Command-line arguments for all SRPO parameters
   - Validation and compatibility checks
   - Automatic configuration adjustments

4. **GUI Support** (`kohya_gui/`)
   - New SRPO training parameters class
   - Integrated into LoRA GUI
   - Configuration save/load support
   - Comprehensive tooltips and documentation

### Files Added

```
sd-scripts/library/
├── srpo_reward_models.py       # Reward model implementations
└── srpo_train_utils.py         # SRPO training utilities

kohya_gui/
└── class_srpo_training.py      # GUI parameter class

docs/
├── SRPO_Training_Guide.md      # Complete documentation
└── SRPO_Quick_Start.md         # Quick start guide

scripts/
└── download_srpo_models.sh     # Model download helper
```

### Files Modified

```
sd-scripts/
└── train_network.py            # Added SRPO arguments

kohya_gui/
└── lora_gui.py                 # Integrated SRPO parameters
```

## Features

✅ **Direct Preference Alignment**: No traditional RL complexity  
✅ **Fast Training**: Visible improvements in ~10 minutes  
✅ **Multiple Reward Models**: HPS, PickScore, CLIP support  
✅ **GUI Integration**: Easy checkbox-based configuration  
✅ **Command Line Support**: Full parameter control  
✅ **Memory Optimized**: Gradient checkpointing support  
✅ **Prevention of Reward Hacking**: ReLU threshold regularization  
✅ **Dynamic Style Control**: Controllable text conditions  

## Quick Start

### 1. Install Dependencies

```bash
pip install hpsv2
```

### 2. Download Reward Models

```bash
bash scripts/download_srpo_models.sh
```

Or manually:
```bash
huggingface-cli download xswu/HPSv2 HPS_v2.1_compressed.pt --local-dir models/reward_models/hps_v2.1
huggingface-cli download laion/CLIP-ViT-H-14-laion2B-s32B-b79K --local-dir models/reward_models/clip
```

### 3. Enable in GUI

1. Open LoRA training tab
2. Expand "SRPO Training" accordion
3. Check "Enable SRPO Training"
4. Select "HPS" as reward model
5. Start training!

### 4. Or Use Command Line

```bash
accelerate launch sdxl_train_network.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --dataset_config="dataset.toml" \
  --output_dir="./output" \
  --network_module="networks.lora" \
  --network_dim=32 \
  --learning_rate=5e-6 \
  --max_train_steps=100 \
  --mixed_precision="bf16" \
  --gradient_checkpointing \
  --srpo_enable \
  --srpo_reward_model="HPS"
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--srpo_enable` | False | Enable SRPO training mode |
| `--srpo_use_reward_model` | True | Use reward model for guidance (disable for anime/styles with biased models) |
| `--srpo_reward_model` | "HPS" | Reward model (HPS/PickScore/CLIP) - only used if use_reward_model is True |
| `--srpo_timestep_length` | 100 | Number of timesteps |
| `--srpo_discount_pos` | [0.1, 0.25] | Positive branch discount range |
| `--srpo_discount_inv` | [0.3, 0.01] | Inversion branch discount range |
| `--srpo_train_timestep` | [5, 25] | Training timestep range |
| `--srpo_groundtruth_ratio` | 0.9 | Groundtruth ratio for recovery |
| `--srpo_guidance_scale` | 3.5 | CFG scale for sampling |
| `--srpo_reward_threshold` | 0.7 | Reward threshold (prevents hacking) |
| `--srpo_positive_controls` | None (uses defaults) | Custom positive control words (space-separated) |
| `--srpo_negative_controls` | None (uses defaults) | Custom negative control words (space-separated) |

### NEW: Disable Reward Model for Anime/Illustration

Existing reward models (HPS, CLIP, PickScore) are biased toward photorealism. For anime or illustration training, you can disable the reward model:

```bash
--srpo_enable
# Simply omit --srpo_use_reward_model to disable
```

When disabled, SRPO uses a preference-weighted denoising objective instead of external reward signals.

### NEW: Custom Control Words

You can customize which style qualities SRPO optimizes toward (only used when reward model is enabled):

**For Anime:**
```bash
--srpo_use_reward_model \
--srpo_positive_controls "cel-shaded" "vibrant colors" "stylized" "sharp lineart" \
--srpo_negative_controls "photorealistic" "3D render" "realistic lighting"
```

**For Illustration:**
```bash
--srpo_use_reward_model \
--srpo_positive_controls "artistic" "painterly" "expressive brushwork" \
--srpo_negative_controls "photograph" "raw photo" "candid shot"
```

Leave empty to use defaults (optimized for photorealism).

## Documentation

- **Complete Guide**: `docs/SRPO_Training_Guide.md`
- **Quick Start**: `docs/SRPO_Quick_Start.md`

## Technical Details

### Architecture

```
┌─────────────────────────────────────────┐
│         SRPO Training Loop              │
├─────────────────────────────────────────┤
│  1. Online Rollout                      │
│     └─ Generate image from noise        │
│  2. Inject Noise                        │
│     └─ Add controlled noise at timestep │
│  3. Inverse/Denoise Step                │
│     └─ One gradient-enabled step        │
│  4. Recover Image                       │
│     └─ Decode latents via VAE           │
│  5. Compute Reward                      │
│     └─ Reward model evaluation          │
│  6. Backpropagation                     │
│     └─ Update LoRA weights              │
└─────────────────────────────────────────┘
```

### Reward Models

- **HPS-v2.1**: Trained on 430k human preference ratings
- **PickScore**: Alternative preference model
- **CLIP**: Standard CLIP similarity (fallback)

### Memory Requirements

SRPO requires more VRAM due to online generation:
- **Minimum**: 16GB (with optimizations)
- **Recommended**: 24GB+
- **Optimizations**: gradient_checkpointing, mixed_precision

## Compatibility

✅ **Supported**:
- SDXL LoRA training
- Standard LoRA, LoCon, LoHa, etc.
- Mixed precision (fp16, bf16)
- Gradient checkpointing
- Multi-GPU training

⚠️ **Not Supported**:
- SD 1.5 models (SDXL only)
- Full model fine-tuning
- Cached latents (disabled automatically)

## Troubleshooting

### Out of Memory
- Enable `--gradient_checkpointing`
- Reduce `--train_batch_size`
- Use smaller `--network_dim`

### Reward Hacking
- Increase `--srpo_reward_threshold`
- Adjust discount parameters
- Train for fewer steps

### Slow Training
- SRPO is slower than cached training (by design)
- Use `--vae_batch_size` to batch operations
- Enable text encoder output caching

## Research & Credits

**Original Paper**: [Directly Aligning the Full Diffusion Trajectory with Fine-Grained Human Preference](https://arxiv.org/abs/2509.06942)

**Authors**: Xiangwei Shen, Zhimin Li, et al. (Tencent Hunyuan)

**Original Implementation**: [Tencent-Hunyuan/SRPO](https://github.com/Tencent-Hunyuan/SRPO)

**Integration**: Adapted for Kohya_ss with focus on SDXL LoRA training

## License

This integration follows the original SRPO license (Apache 2.0) and Kohya_ss license.

## Contributing

Contributions are welcome! Please:
1. Test thoroughly with SDXL models
2. Document parameter changes
3. Include sample outputs
4. Tag PRs with "SRPO"

## Support

- **Issues**: GitHub Issues with "SRPO" tag
- **Documentation**: `docs/SRPO_Training_Guide.md`
- **Original Paper**: https://arxiv.org/abs/2509.06942

---

**Happy Training with SRPO! 🚀✨**
