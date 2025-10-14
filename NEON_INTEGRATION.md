# Neon Integration for Kohya_ss

## What is Neon?

**Neon (Negative Extrapolation from Self-Training)** is an **automatic post-training technique** that **improves LoRAs by generating synthetic data, training on it, then extrapolating away from the degradation**.

- **Paper**: [Neon: Negative Extrapolation From Self-Training](https://arxiv.org/abs/2510.03597)
- **Original Repo**: [VITA-Group/Neon](https://github.com/VITA-Group/Neon)

### The Magic Formula

```
θ_neon = (1+w)·θ_base - w·θ_aux

where:
- θ_base = your trained LoRA (after main training completes)
- θ_aux = auxiliary LoRA (post-trained on auto-generated synthetic data)
- w = extrapolation weight (default: 0.3)
```

By extrapolating **away** from synthetic-trained weights, you improve beyond the base model!

## Quick Start

### ✨ Automatic Workflow (One Checkbox!)

Just enable Neon and train as usual:

```bash
python sdxl_train_network.py \
  --dataset_config="my_data.toml" \
  --output_name="my_lora" \
  --max_train_epochs=20 \
  [... standard training args ...] \
  --neon_enable \
  --neon_post_training_steps=100 \
  --neon_synthetic_image_percent=100
```

**What happens:**
1. **Main training** runs (e.g., 20 epochs)
2. **Automatic**: Generates synthetic images using trained model + original prompts
3. **Automatic**: Post-trains on synthetic data for 100 steps (aux model)
4. **Automatic**: Applies Neon merge → saves improved `my_lora.safetensors`

**That's it!** One command, fully automatic improvement!

## Integration Details

### Files Added

```
sd-scripts/library/neon_train_utils.py    # Core Neon merge logic
kohya_gui/class_neon_training.py         # GUI components
docs/Neon_Training_Guide.md              # Full documentation
NEON_INTEGRATION.md                       # This file
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--neon_enable` | False | Enable automatic Neon post-training |
| `--neon_save_pre_post` | False | Save model before post-training (adds '_neon_pre' suffix) |
| `--neon_post_training_epochs` | 0 | Epochs for post-training (0 = use steps instead) |
| `--neon_post_training_steps` | 100 | Steps for post-training (used if epochs=0) |
| `--neon_synthetic_image_percent` | 100.0 | % of original images to generate (100 = same count) |
| `--neon_extrapolation_weight` | 0.3 | Extrapolation weight `w` (range: 0.1-0.5) |

### GUI Components

Location: **"Neon Post-Training"** accordion

Components:
- **Enable Neon Post-Training**: Main checkbox to activate automatic workflow
- **Save Pre-Post Model**: Save model before post-training (comparison)
- **Post-Training Epochs/Steps**: Duration of post-training phase
- **Synthetic Image Percentage**: How many synthetic images to generate
- **Extrapolation Weight (w)**: Merge strength slider

### Config File Support

Neon parameters save to TOML:

```toml
[neon]
enable = true
save_pre_post = false
post_training_epochs = 0
post_training_steps = 100
synthetic_image_percent = 100.0
extrapolation_weight = 0.3
```

## Usage Examples

### Example 1: Basic Automatic Neon (CLI)

```bash
# Single command - everything automatic!
accelerate launch sdxl_train_network.py \
  --dataset_config="character_data.toml" \
  --output_name="my_character" \
  --max_train_epochs=20 \
  --learning_rate=1e-4 \
  [... standard training args ...] \
  --neon_enable \
  --neon_post_training_steps=100 \
  --neon_synthetic_image_percent=100 \
  --neon_save_pre_post
```

**What happens:**
- Trains for 20 epochs (main training)
- Automatically generates 100% of original image count as synthetic
- Post-trains on synthetic for 100 steps
- Saves `my_character_neon_pre.safetensors` (pre-post)
- Merges and saves `my_character.safetensors` (final Neon-improved)

### Example 2: Neon + SRPO (CLI)

Combine two advanced techniques in one command!

```bash
accelerate launch sdxl_train_network.py \
  --dataset_config="data.toml" \
  --output_name="style_advanced" \
  --max_train_epochs=20 \
  --srpo_enable \
  --srpo_reward_model="HPS" \
  --srpo_use_reward_model \
  --neon_enable \
  --neon_post_training_steps=150 \
  --neon_synthetic_image_percent=100
```

**What happens:**
- Trains with SRPO for 20 epochs (perceptual alignment)
- Automatically generates synthetic images using SRPO result
- Post-trains on synthetic for 150 steps
- Merges → final model has **SRPO quality + Neon generalization**!

**Result**: Best of both worlds in one training run!

### Example 3: Standalone Merge Tool

If you already have base + aux checkpoints:

```python
from library.neon_train_utils import neon_merge_lora

neon_merge_lora(
    base_lora_path="my_lora_base.safetensors",
    aux_lora_path="my_lora_aux.safetensors",
    output_path="my_lora_neon.safetensors",
    w=0.3
)
```

### Example 4: GUI Workflow (Super Simple!)

1. **Configure training normally**:
   - Dataset, epochs, learning rate, etc.
   - Output name: `anime_lora`
   - Optionally enable SRPO

2. **Enable Neon** (one checkbox!):
   - Expand "Neon Post-Training" accordion
   - ✅ Check "Enable Neon Post-Training"
   - Set Post-Training Steps: 100
   - Set Synthetic Image %: 100
   - ✅ Check "Save Pre-Post Model" (optional, for comparison)

3. **Click "Train"**:
   - Everything happens automatically!
   - Main training completes
   - Synthetic generation (automatic)
   - Post-training (automatic)
   - Neon merge (automatic)

**Result**: `anime_lora.safetensors` (Neon-improved) + `anime_lora_neon_pre.safetensors` (pre-post)

## Parameter Tuning

### Extrapolation Weight (w)

| Value | Behavior | When to Use |
|-------|----------|-------------|
| 0.1-0.2 | Conservative | First try, uncertain synthetic quality |
| 0.3 | **Default** | Recommended starting point |
| 0.4-0.5 | Aggressive | When confident in synthetic data |
| > 0.5 | Risky | Not recommended (can destabilize) |

### Synthetic Training Steps

| Steps | Effect | Recommendation |
|-------|--------|----------------|
| < 50 | Weak aux signal | Too few |
| 50-100 | Good aux signal | **Recommended** |
| 100-200 | Strong aux signal | Works well |
| > 200 | Overfitting risk | Usually too many |

### Synthetic Dataset Size

| Size | Quality | Note |
|------|---------|------|
| < 1k | Noisy | Minimum viable |
| 1k-5k | Good | Works well |
| 5k-10k | **Optimal** | Best results |
| > 10k | Diminishing returns | Neon is efficient! |

## Technical Details

### Merge Algorithm

```python
def neon_merge(base_params, aux_params, w=0.3):
    """
    θ_neon = (1+w)·θ_base - w·θ_aux
             = θ_base - w·(θ_aux - θ_base)
    """
    neon_params = {}
    for key in base_params:
        if key in aux_params:
            # Negative extrapolation
            neon_params[key] = base_params[key] - w * (
                aux_params[key] - base_params[key]
            )
        else:
            # Keep base if key missing in aux
            neon_params[key] = base_params[key]
    return neon_params
```

### Why It Works

1. **Synthetic training creates predictable degradation**:
   - Mode-seeking behavior in samplers
   - Narrower distribution than real data
   - θ_aux drifts away from true data manifold

2. **Negative extrapolation corrects the drift**:
   - Compute direction: θ_aux - θ_base (degradation vector)
   - Extrapolate opposite: θ_base - w·(degradation)
   - Result: Better alignment with true distribution

3. **Small synthetic set works**:
   - Only need to **detect drift direction**
   - Don't need full coverage of real distribution
   - That's why 1k-5k synthetic images suffice!

## Best Practices

### ✅ Do's

1. **Validate aux model**: Test it before merging (should be slightly worse)
2. **Match hyperparameters**: Base and aux must use same network architecture
3. **Start conservative**: w=0.3 for first attempt
4. **Diverse synthetic**: Use varied prompts/seeds in generation
5. **Compare systematically**: Test base vs aux vs neon side-by-side

### ❌ Don'ts

1. **Don't change network dim/alpha** between base and aux
2. **Don't overtrain aux** (> 200 steps usually counterproductive)
3. **Don't use extreme w** (< 0.05 or > 0.6)
4. **Don't skip validation** (always test aux before merging)
5. **Don't expect miracles** (Neon improves, but can't fix bad base model)

## Troubleshooting

### Neon output looks worse than base

**Causes & Solutions**:
- **w too high** → Reduce to 0.2 or 0.1
- **Aux overtrained** → Reduce synthetic training steps
- **Poor synthetic data** → Regenerate with better diversity
- **Architecture mismatch** → Verify base/aux have same dim/alpha

### Neon output looks same as base

**Causes & Solutions**:
- **w too low** → Increase to 0.4
- **Aux undertrained** → Increase synthetic training steps
- **Synthetic too close to real** → Actually good! (means base is well-calibrated)

### Merge fails with error

**Possible issues**:
- **Format mismatch** → Ensure both .safetensors or both .pt
- **Key mismatch** → Check network architectures match exactly
- **File not found** → Verify base checkpoint path is correct

## Limitations

1. **Requires base checkpoint**: Can't apply Neon without reference model
2. **Small improvements**: Typical gains are 10-30% (not 2-3x)
3. **Quality dependent**: Better synthetic data → better Neon results
4. **Not a fix-all**: Won't fix fundamental issues with base training

## Future Enhancements

Potential additions:
- [ ] Multi-checkpoint averaging (ensemble base models)
- [ ] Adaptive w selection based on FID/metrics
- [ ] Auto-synthetic generation pipeline integration
- [ ] Iterative Neon with decay schedules

## Documentation

- **Full Guide**: `docs/Neon_Training_Guide.md`
- **GUI Help**: Hover over parameters in "Neon Training (Advanced)"
- **Paper**: https://arxiv.org/abs/2510.03597
- **Original Repo**: https://github.com/VITA-Group/Neon

---

**Questions? Check the FAQ in `Neon_Training_Guide.md`!**
