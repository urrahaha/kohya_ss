# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kohya_ss is a **Gradio-based GUI and CLI for training diffusion models**, specifically for fine-tuning Stable Diffusion models using various techniques like LoRA, DreamBooth, and Textual Inversion. It wraps Kohya's sd-scripts training library with an accessible interface.

**Core Purpose**: Provide a user-friendly interface for training custom diffusion models (LoRA networks, fine-tunes, etc.) with extensive configuration options.

## Repository Architecture

### High-Level Structure

```
kohya_ss/
├── kohya_gui.py              # Main GUI entry point (Gradio interface)
├── kohya_gui/                # GUI module directory
│   ├── *_gui.py             # Training mode tabs (lora, dreambooth, finetune, etc.)
│   ├── class_*.py           # Reusable UI component classes
│   └── utilities.py         # Shared utility functions
├── sd-scripts/              # Git submodule: Core training scripts from kohya-ss
│   ├── train_network.py    # SD 1.x/2.x LoRA training
│   ├── sdxl_train_network.py  # SDXL LoRA training
│   ├── flux_train_network.py  # Flux.1 LoRA training
│   ├── sd3_train_network.py   # SD3 LoRA training
│   └── library/            # Shared training utilities
├── reference_repos/         # Reference implementations (git submodules)
│   ├── NLoRA/
│   ├── SRPO/
│   ├── Neon/
│   └── diff2flow/
├── config.toml              # User configuration (default paths)
├── setup.sh / setup.bat     # Installation scripts
└── gui.sh / gui.bat         # GUI launcher scripts
```

### Key Design Patterns

**1. Training Script Orchestration**: The GUI constructs command-line arguments for underlying Python training scripts in `sd-scripts/`. It uses `accelerate launch` to execute training with proper multi-GPU support.

**2. Component-Based UI**: Training parameters are organized into class-based Gradio components:
- `SourceModel`: Model selection, paths, checkpoints
- `BasicTraining`: Core training params (LR, epochs, batch size)
- `AdvancedTraining`: Advanced options (gradient checkpointing, xformers, etc.)
- `Folders`, `SampleImages`, `HuggingFace`, `MetaData`: Specialized sections

**3. Config File Management**: All training configurations can be saved/loaded as JSON files. The GUI uses a dual-system:
- **JSON**: Save/load UI states for individual training runs
- **TOML**: Export final training configs (consumed by sd-scripts)

**4. Submodule Integration**: The `sd-scripts` directory is a git submodule pointing to kohya-ss/sd-scripts. New features from the upstream repo are integrated via submodule updates.

## Development Commands

### Setup and Installation

```bash
# Linux/macOS initial setup
./setup.sh

# Skip git updates during setup (for development)
./setup.sh --no-git-update

# macOS with homebrew Python
brew install python@3.11
./setup.sh

# Windows initial setup
.\setup.bat
```

### Running the GUI

```bash
# Linux/macOS
./gui.sh

# With custom listen address and port
./gui.sh --listen 0.0.0.0 --server_port 7860

# Headless mode (for servers)
./gui.sh --headless

# Windows
.\gui.bat

# Using uv (newer method, no setup required)
./gui-uv.sh  # Linux/macOS
.\gui-uv.bat # Windows
```

### Environment Management

```bash
# Activate virtual environment manually
source venv/bin/activate  # Linux/macOS
.\venv\Scripts\activate   # Windows

# Use conda (alternative)
conda activate kohya_ss

# Install/update dependencies
pip install -r requirements.txt  # Basic
pip install -r requirements_linux.txt  # Linux CUDA
pip install -r requirements_linux_rocm.txt  # AMD ROCm
```

### Testing

```bash
# No automated test suite currently exists
# Testing is primarily manual through the GUI

# To test a specific training script directly:
cd sd-scripts
accelerate launch train_network.py --help
```

### Working with sd-scripts Submodule

```bash
# Update sd-scripts to latest commit
git submodule update --remote sd-scripts
git add sd-scripts
git commit -m "chore: update sd-scripts submodule to latest commit"

# Initialize submodules after fresh clone
git submodule update --init --recursive
```

## Code Organization Principles

### Training Flow

1. **User configures training** via GUI (e.g., `kohya_gui/lora_gui.py`)
2. **GUI validates inputs** (paths, model existence, parameter ranges)
3. **Config TOML generated** dynamically from UI state
4. **Command constructed**: `accelerate launch <script> --config_file <toml>`
5. **Training executes** via `CommandExecutor` class (non-blocking subprocess)
6. **Outputs saved** to user-specified directory

### Adding New Training Parameters

When sd-scripts adds a new parameter:

1. **Add to class component** (e.g., `class_basic_training.py` or `class_advanced_training.py`)
2. **Wire to main tab** (e.g., `lora_gui.py`):
   - Add to `settings_list` for config save/load
   - Add to function signatures: `save_configuration()`, `open_configuration()`, `train_model()`
3. **Add to TOML generation** in `train_model()` function (the `config_toml_data` dict)
4. **Update config examples** if necessary (`config example.toml`)

**Important**: The order of parameters in function signatures must match exactly between save/load/train functions. The code includes runtime arity checking to catch mismatches.

### GUI Component Pattern

```python
# Standard component structure
class MyComponent:
    def __init__(self, headless=False, config=None):
        with gr.Accordion("My Section"):
            self.my_param = gr.Textbox(
                label="My Parameter",
                value=config.get("my_section.my_param", "default")
            )
```

## Special Integrations

### SRPO (Semantic Reward Policy Optimization)

Location: `kohya_gui/class_srpo_training.py`

Integration from Tencent's SRPO paper for reward-model-based training. Adds parameters like `--srpo_enable`, `--srpo_reward_model`, `--srpo_use_diff2flow`.

**Key Files**:
- `reference_repos/SRPO/` (reference implementation)
- `sd-scripts/library/srpo_utils.py` (integration code)

### Neon (Negative Extrapolation Post-Training)

Location: `kohya_gui/class_neon_training.py`

Automatic post-training workflow that generates synthetic data and applies negative extrapolation to improve LoRAs.

**Key Files**:
- `reference_repos/Neon/` (reference implementation)
- `sd-scripts/library/neon_train_utils.py` (merge logic)
- `NEON_INTEGRATION.md` (detailed usage guide)

### Diff2Flow

Integration for flow matching with diffusion models. Used in both SRPO and Neon workflows via `--use_diff2flow` and `--d2f_param` flags.

**Key Files**:
- `reference_repos/diff2flow/` (reference implementation)
- Used in SDXL flow matching training

### Aurora Spline LoRA

Recent addition: Aurora LoRA type with learnable spline gates for improved adaptation.

**Parameters**: `spline_gate`, `spline_scale`, `spline_centers`

## Configuration System

### config.toml Structure

```toml
# Default paths for GUI dropdowns
model_dir = "/path/to/models"
lora_model_dir = "/path/to/loras"
output_dir = "/path/to/outputs"
dataset_dir = "/path/to/datasets"

# Section-specific paths
[lora]
lc_model_dir = "/path/to/models"
lc_output_dir = "/path/to/outputs"
```

The GUI loads these defaults via `KohyaSSGUIConfig` class and pre-fills input fields.

### Training Config JSON

Saved training configurations include all UI state:
- Source model settings
- Training hyperparameters
- Advanced options
- Acceleration settings
- Metadata

**File naming**: `{output_name}_{timestamp}.json`

## Common Patterns

### Path Validation

```python
# Always validate paths before training
if not validate_file_path(dataset_config):
    return TRAIN_BUTTON_VISIBLE  # Abort with error message

if not validate_folder_path(output_dir, can_be_written_to=True, create_if_not_exists=True):
    return TRAIN_BUTTON_VISIBLE
```

### Command Construction

```python
# Start with accelerate launch
run_cmd = [accelerate_path, "launch"]

# Add accelerate config
run_cmd = AccelerateLaunch.run_cmd(
    run_cmd=run_cmd,
    num_processes=num_processes,
    mixed_precision=mixed_precision,
    # ...
)

# Add training script
run_cmd.append(rf"{scriptdir}/sd-scripts/train_network.py")

# Add config file
run_cmd.append("--config_file")
run_cmd.append(tmpfilename)
```

### Learning Rate Hierarchy

The training scripts support multiple learning rates with fallback logic:
1. **Main LR**: Base learning rate for optimizer
2. **Text Encoder LR**: Specific LR for CLIP/T5 (falls back to main LR if 0)
3. **T5XXL LR**: Override for T5XXL specifically (falls back to TE LR, then main LR)
4. **Unet LR**: Specific LR for diffusion model (falls back to main LR if 0)

## Gotchas and Known Issues

1. **Parameter Ordering**: Function signatures for `save_configuration()`, `open_configuration()`, and `train_model()` must have parameters in identical order. Runtime checks exist but adding params requires updating all three.

2. **Submodule Updates**: When updating `sd-scripts`, new parameters may appear. Always check the upstream CHANGELOG and add corresponding GUI controls.

3. **Shell Usage**: The `use_shell` flag controls whether subprocess commands use `shell=True`. This can cause issues on some systems. Controlled via `--do_not_use_shell` CLI flag.

4. **Headless Mode**: When running on servers without display, use `--headless` flag. This disables file picker buttons and auto-opens.

5. **TOML vs JSON**: Training runs export TOML (for sd-scripts), but UI state is saved as JSON. They are not interchangeable.

6. **Virtual Environment**: The GUI expects either a venv at `./venv` or an active conda environment. Check `gui.sh` for environment detection logic.

## Model Architecture Support

The codebase supports multiple diffusion model architectures:

- **SD 1.x/2.x**: Via `train_network.py`
- **SDXL**: Via `sdxl_train_network.py` (also supports flow matching with `sdxl_train_network_fm.py`)
- **Flux.1**: Via `flux_train_network.py` (newer architecture)
- **SD3**: Via `sd3_train_network.py`

Each has specific parameters and requirements. Use checkboxes in `SourceModel` class to switch architectures.

## LoRA Types

Supported LoRA variants (selected via dropdown):
- **Standard**: Basic LoRA
- **Kohya LoCon**: Adds convolutional layer adaptation
- **Kohya DyLoRA**: Dynamic rank LoRA
- **LoRA-FA**: Frozen-A variant
- **LyCORIS family**: BOFT, Diag-OFT, DyLoRA, GLoRA, iA3, LoCon, LoHa, LoKr, Native Fine-Tuning
- **Flux1**: Architecture-specific (double blocks, single blocks)
- **Flux1 OFT**: Orthogonal fine-tuning for Flux
- **LoFT**: Latest variant
- **NLoRA**: Negative LoRA (from NLoRA reference)
- **AuroRA**: Spline-based LoRA

Each type has specific network args constructed in `train_model()` function.

## Localization

The GUI supports multiple languages via JSON files in `localizations/`. The `localization_ext.py` module injects JavaScript to swap UI text based on `--language` flag.

## Best Practices

1. **Always read existing implementations** before adding new parameters
2. **Follow the component pattern** for UI organization
3. **Use runtime arity checks** - they catch signature mismatches early
4. **Validate all paths** before executing training commands
5. **Test with print mode** (`--print_only`) to verify command construction without running training
6. **Keep submodules in sync** with upstream when adding features
7. **Document new parameters** in GUI tooltips (`info=` parameter) and update `CLAUDE.md`

## Additional Resources

- **LoRA Training Guide**: `docs/LoRA/top_level.md`
- **SRPO Integration**: `SRPO_INTEGRATION.md`
- **Neon Integration**: `NEON_INTEGRATION.md`
- **Installation Guides**: `docs/Installation/`
- **Original sd-scripts README**: `sd-scripts/README.md`
