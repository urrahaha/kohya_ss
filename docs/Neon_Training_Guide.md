# Neon Training Guide for Kohya_ss

## Overview

**Neon (Negative Extrapolation from Self-Training)** is a groundbreaking post-training technique that **improves your LoRA by training on synthetic data, then extrapolating away from the degradation**. It turns the curse of model collapse into a powerful self-improvement signal.

Based on the research paper: [Neon: Negative Extrapolation From Self-Training Improves Image Generation](https://arxiv.org/abs/2510.03597)

Original implementation: [VITA-Group/Neon](https://github.com/VITA-Group/Neon)

## Key Features

### 🔄 Self-Improvement from Synthetic Data
- **No new real data needed**: Uses model's own generated images
- **Prevents model collapse**: Instead of degrading, the model improves
- **Extremely efficient**: <1% additional training compute

### 🎯 The Algorithm (Simple!)

1. **Train your LoRA normally** → save as "base checkpoint" (θ_base)
2. **Generate synthetic dataset** using your base LoRA (1k-10k images)
3. **Fine-tune on synthetic** for ~50-200 steps → "auxiliary checkpoint" (θ_aux)
4. **Neon merge**: `θ_neon = (1+w)·θ_base - w·θ_aux`

The magic: By extrapolating **away** from the synthetic-trained weights, you push toward better coverage of the real data distribution!

### ✨ Why It Works

**Traditional self-training** causes model collapse:
- Model generates narrow/biased synthetic data (mode-seeking)
- Fine-tuning on synthetic makes it even narrower
- Positive feedback loop → collapse

**Neon flips the script**:
- Synthetic training creates predictable degradation
- Negative extrapolation **reverses** that degradation
- Result: Better generalization than the base model!

## Performance Improvements

From the Neon paper (on various architectures):

| Model | Dataset | Base FID | Neon FID | Improvement |
|-------|---------|----------|----------|-------------|
| xAR-L | ImageNet-256 | 1.28 | **1.02** | -20% (SOTA!) |
| VAR-d16 | ImageNet-256 | 3.30 | **2.01** | -39% |
| EDM | CIFAR-10 | 1.78 | **1.38** | -22% |

## Prerequisites

### 1. Trained Base LoRA
You need a working LoRA trained on real data. This becomes your "reference" model.

### 2. Synthetic Data Generation
Use your base LoRA to generate synthetic images:
- **Quantity**: 1,000-10,000 images recommended
- **Diversity**: Use varied prompts from your training distribution
- **Tools**: Any SD/SDXL generator (ComfyUI, Auto1111, etc.)

### 3. Dataset Structure
Organize synthetic images like your original training data:
```
synthetic_dataset/
├── images/
│   ├── img_001.png
│   ├── img_002.png
│   └── ...
└── captions/  (or metadata)
    ├── img_001.txt
    ├── img_002.txt
    └── ...
```

## Step-by-Step Workflow

### Step 1: Train Base LoRA (Normal Training)

Train your LoRA as usual on **real data**:

```bash
accelerate launch sdxl_train_network.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --dataset_config="real_dataset.toml" \
  --output_dir="./output" \
  --output_name="my_lora_base" \
  --save_model_as=safetensors \
  --network_module="networks.lora" \
  --network_dim=32 \
  --network_alpha=16 \
  --learning_rate=1e-4 \
  --max_train_steps=2000 \
  [... other training args ...]
```

**Result**: `my_lora_base.safetensors` (your reference checkpoint)

### Step 2: Generate Synthetic Dataset

Use your base LoRA to generate synthetic images:

**Example with ComfyUI/Auto1111**:
1. Load base LoRA at full strength
2. Use diverse prompts from your training set
3. Generate 1k-10k images
4. Caption them (reuse original captions or auto-caption)

**Tips**:
- **Match training distribution**: Use similar prompts/styles as original data
- **Diverse sampling**: Don't use the same seeds/settings repeatedly
- **Quality**: Use reasonable inference settings (not too extreme CFG)

### Step 3: Fine-Tune on Synthetic Data

**Critical**: This step intentionally degrades the model (creates θ_aux)

```bash
accelerate launch sdxl_train_network.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --train_data_dir="./synthetic_dataset" \
  --resume="./output/my_lora_base.safetensors" \
  --output_dir="./output" \
  --output_name="my_lora_aux" \
  --save_model_as=safetensors \
  --network_module="networks.lora" \
  --network_dim=32 \
  --network_alpha=16 \
  --learning_rate=5e-5 \
  --max_train_steps=100 \
  [... match base training args ...]
```

**Key points**:
- Start from `--resume` (base LoRA)
- Train on **synthetic** data only
- **Short training**: 50-200 steps (typical: 100)
- **Lower LR**: ~50% of base LR to avoid overfitting

**Result**: `my_lora_aux.safetensors` (degraded from synthetic training)

### Step 4: Apply Neon Merge

Now merge base + aux using the Neon formula:

**Option A: Using the standalone merge tool**:

```bash
python sd-scripts/library/neon_merge.py \
  --base_checkpoint="./output/my_lora_base.safetensors" \
  --aux_checkpoint="./output/my_lora_aux.safetensors" \
  --output="./output/my_lora_neon.safetensors" \
  --extrapolation_weight=0.3
```

**Option B: Auto-merge during training** (add to Step 3 command):

```bash
  --neon_enable \
  --neon_base_checkpoint="./output/my_lora_base.safetensors" \
  --neon_extrapolation_weight=0.3 \
  --neon_output_name="my_lora_neon"
```

This automatically merges after training completes.

**Result**: `my_lora_neon.safetensors` (improved via negative extrapolation!)

## Parameters

### Core Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| **Extrapolation Weight (w)** | 0.3 | 0.1-0.5 | Higher = stronger extrapolation from synthetic model |
| **Synthetic Training Steps** | 100 | 50-200 | Steps to train on synthetic data (creates aux model) |
| **Synthetic Dataset Size** | 5000 | 1000-10000 | Number of synthetic images |

### Tuning Guide

**Extrapolation Weight (`w`)**:
- `w=0.0`: No Neon (just returns base model)
- `w=0.1-0.2`: Conservative extrapolation (safer, smaller improvement)
- `w=0.3`: **Recommended default** (good balance)
- `w=0.4-0.5`: Aggressive extrapolation (higher risk, higher reward)
- `w>0.5`: Not recommended (can become unstable)

**Synthetic Training Steps**:
- **Too few** (< 50): aux model not different enough from base → weak signal
- **Optimal** (50-200): Clear degradation signal → good extrapolation
- **Too many** (> 300): Severe overfitting → extrapolation may overcorrect

**Synthetic Dataset Size**:
- **Minimum** (1k): Works but may be noisy
- **Recommended** (5k-10k): Best balance
- **More** (> 10k): Diminishing returns (Neon works even with small synthetic sets!)

## GUI Usage

### GUI Workflow

1. **Train base LoRA**:
   - Configure training normally
   - Set Output Name: `my_lora_base`
   - Train as usual

2. **Generate synthetic data** (external tool)

3. **Train on synthetic**:
   - Load previous training config
   - Change dataset to synthetic folder
   - Set Resume: `my_lora_base.safetensors`
   - Reduce max steps: 100
   - Set Output Name: `my_lora_aux`

4. **Apply Neon**:
   - Expand "Neon Training (Advanced)"
   - Check ✅ "Enable Neon Merge"
   - Base Checkpoint: `/path/to/my_lora_base.safetensors`
   - Extrapolation Weight: 0.3
   - (Optional) Output Name: `my_lora_neon`
   - Click "Train" (aux training + auto-merge)

### GUI Parameters

![Neon GUI](assets/neon_gui.png)

- **Enable Neon Merge**: Turn on Neon post-processing
- **Base Checkpoint Path**: Your reference LoRA (before synthetic training)
- **Extrapolation Weight**: The `w` parameter (default: 0.3)
- **Output Name**: Custom name for merged result (auto-generated if empty)

## Advanced Usage

### Combining Neon with SRPO

Neon works great **after** SRPO training!

```bash
# 1. Train with SRPO (perceptual alignment)
--srpo_enable \
--srpo_reward_model="HPS" \
--output_name="my_lora_srpo"

# 2. Use SRPO result as base for Neon
# (Generate synthetic, train aux, then merge)

# 3. Result: SRPO quality + Neon generalization!
```

### Iterative Neon

You can iterate Neon multiple times:

```
Base → Neon1 → use Neon1 as new base → Neon2 → ...
```

**Caution**: Each iteration compounds, so use smaller `w` values (e.g., 0.2)

### Style-Specific Synthetic Data

For better results, generate synthetic data that **emphasizes** the aspects you want to improve:

- **Character consistency**: Generate same character in many poses
- **Style coherence**: Generate diverse scenes in target style
- **Lighting variety**: Vary lighting conditions in synthetic set

## Troubleshooting

### Issue: Neon Output Looks Worse

**Possible causes**:
1. **w too high**: Try reducing to 0.2 or 0.1
2. **Aux training too long**: Reduce synthetic training steps
3. **Poor synthetic data**: Regenerate with better diversity

### Issue: Neon Output Looks the Same as Base

**Possible causes**:
1. **w too low**: Try increasing to 0.4
2. **Aux training too short**: Increase synthetic training steps
3. **Synthetic data too similar to real**: This is actually good (means base was already well-calibrated)

### Issue: Merged Model Has Artifacts

**Solutions**:
1. Check that base and aux have **same architecture** (dim, alpha, etc.)
2. Ensure both models are **same format** (.safetensors or .pt)
3. Verify synthetic training didn't collapse completely (check aux outputs)

## Tips & Best Practices

### ✅ Do's

- **Start with defaults**: w=0.3, 100 synthetic steps
- **Match architectures**: Base and aux must have identical network config
- **Diverse synthetic data**: Avoid mode collapse in generation
- **Validate aux model**: Test it before merging (should be slightly worse than base)
- **Compare all three**: Base vs Aux vs Neon to see the improvement

### ❌ Don'ts

- **Don't use different network dimensions** for base vs aux
- **Don't overtrain on synthetic** (> 200 steps usually too much)
- **Don't use extreme w values** (< 0.05 or > 0.6)
- **Don't skip base checkpoint** (Neon needs reference model)
- **Don't use tiny synthetic sets** (< 500 images less reliable)

## FAQ

**Q: Do I need to train on real data first?**  
A: Yes! Neon requires a base model trained on real data. It's a post-training technique.

**Q: How much does Neon cost in compute?**  
A: Very little! Just 50-200 steps of training (~1-5% of original training time).

**Q: Can I use Neon with other training methods?**  
A: Yes! Works great with standard training, SRPO, DreamBooth, etc.

**Q: What if I don't have synthetic data yet?**  
A: Generate it using your base LoRA in any SD/SDXL UI tool. Use varied prompts.

**Q: Does Neon work for anime/illustration styles?**  
A: Yes! Neon is architecture/style-agnostic. Works for any image generation model.

**Q: Can I apply Neon to someone else's LoRA?**  
A: Only if you have the base checkpoint. Without it, you can't compute the merge.

**Q: What's the minimum synthetic dataset size?**  
A: Paper shows 1k works, but 5k-10k is recommended for stable results.

---

## Citation

If you use Neon in your work, please cite:

```bibtex
@article{alemohammadneon2025,
  title   = {Neon: Negative Extrapolation From Self-Training Improves Image Generation},
  author  = {Alemohammad, Sina and Wang, Zhangyang and Baraniuk, Richard G.},
  journal = {arXiv preprint arXiv:2510.03597},
  year    = {2025}
}
```

---

**Happy training! 🎨✨**
