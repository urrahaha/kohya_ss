"""
GUI components for Neon (Negative Extrapolation from Self-Training)

Neon improves models by training on synthetic data then extrapolating away from degradation.
"""

import gradio as gr
from .custom_logging import setup_logging

log = setup_logging()


class NeonTraining:
    """
    GUI components for Neon training configuration
    """

    def __init__(
        self,
        headless: bool = False,
        config: dict = None,
    ):
        self.headless = headless
        self.config = config or {}

        with gr.Accordion("Neon Post-Training", open=False, elem_id="neon_training_accordion"):
            gr.Markdown(
                """
                ## 🔬 Neon: Automatic Post-Training Refinement
                
                **Automatically improves your LoRA after main training completes!**
                
                ### Automatic Workflow:
                1. **Main training** completes (e.g., 20 epochs with optional SRPO)
                2. **Auto-generate synthetic images** using trained model + original prompts
                3. **Post-training phase** on synthetic data (auxiliary model)
                4. **Neon merge**: `θ_neon = (1+w)θ_base - wθ_aux` → final improved LoRA
                
                ### What You Get:
                - ✅ Fully automatic (one click!)
                - ✅ Uses your original image prompts
                - ✅ Optional: save both pre-post and post-post models
                - ✅ ~5-10 minutes additional time
                
                **Best for:** Squeezing extra quality from any training (works great with SRPO!)
                """
            )
            
            with gr.Row():
                with gr.Column():
                    self.neon_enable = gr.Checkbox(
                        label="Enable Neon Post-Training",
                        value=self.config.get("neon.enable", False),
                        info="Automatically generate synthetic data + post-train + merge after main training"
                    )
                
                with gr.Column():
                    self.neon_save_pre_post = gr.Checkbox(
                        label="Save Pre-Post Model",
                        value=self.config.get("neon.save_pre_post", False),
                        info="Save model before Neon post-training (appends '_neon_pre' to filename)"
                    )
            
            with gr.Row():
                self.neon_synthetic_dataset_dir = gr.Textbox(
                    label="Synthetic Dataset Directory",
                    value=self.config.get("neon.synthetic_dataset_dir", ""),
                    placeholder="./synthetic_data (leave empty for auto: ./output/neon_synthetic)",
                    info="Where to save synthetic images. Images named with '_synthetic' suffix (e.g., img001_synthetic.png). Reuses existing synthetic images!"
                )
            
            with gr.Accordion("Neon Parameters", open=False):
                with gr.Row():
                    with gr.Column():
                        self.neon_post_training_epochs = gr.Number(
                            label="Post-Training Epochs",
                            value=self.config.get("neon.post_training_epochs", 0),
                            minimum=0,
                            precision=0,
                            info="Epochs for post-training (0 = use steps instead)"
                        )
                    
                    with gr.Column():
                        self.neon_post_training_steps = gr.Number(
                            label="Post-Training Steps (Override)",
                            value=self.config.get("neon.post_training_steps", 100),
                            minimum=0,
                            precision=0,
                            info="Steps for post-training (used if epochs=0, default: 100)"
                        )
                
                with gr.Row():
                    with gr.Column():
                        self.neon_synthetic_image_percent = gr.Slider(
                            label="Synthetic Image Percentage",
                            value=self.config.get("neon.synthetic_image_percent", 100),
                            minimum=10,
                            maximum=200,
                            step=10,
                            info="% of original images to generate (100% = same count, includes repeats)"
                        )
                    
                    with gr.Column():
                        self.neon_extrapolation_weight = gr.Slider(
                            label="Extrapolation Weight (w)",
                            value=self.config.get("neon.extrapolation_weight", 0.3),
                            minimum=0.0,
                            maximum=1.0,
                            step=0.05,
                            info="Merge strength. Typical: 0.1-0.5 (default: 0.3)"
                        )
                
                gr.Markdown(
                    """
                    ### 📖 Usage Guide:
                    
                    **Simple workflow:**
                    
                    1. **Configure main training** (standard or SRPO):
                       - Set dataset, epochs, learning rate, etc.
                       - Enable SRPO if desired
                    
                    2. **Enable Neon Post-Training**:
                       - ✅ Check "Enable Neon Post-Training"
                       - Set synthetic dataset dir (or leave empty for auto)
                       - Set post-training steps (default: 100)
                       - Set synthetic image % (default: 100%)
                       - Optionally check "Save Pre-Post Model"
                    
                    3. **Click Train**:
                       - Main training runs (e.g., 20 epochs)
                       - **Automatic**: Generates synthetic images (reuses existing!)
                       - **Automatic**: Post-trains on synthetic data
                       - **Automatic**: Applies Neon merge
                       - Saves final improved LoRA!
                    
                    **Tips:**
                    - **Synthetic image naming**: Images saved with `_synthetic` suffix (auto-detected!)
                    - **Incremental generation**: Only generates missing synthetic images
                    - Post-training steps: 50-200 typical (default 100)
                    - Synthetic %: 100% matches original count, 200% doubles it
                    - Save pre-post: Useful for comparison (adds '_neon_pre' suffix)
                    - Works seamlessly with SRPO training!
                    - Adds ~5-10 minutes to training time
                    
                    **Example workflow:**
                    - Main: 20 epochs on 100 images
                    - Neon: Checks directory, finds 30 `*_synthetic.png` images
                    - Generates: Only 70 more images needed (img031_synthetic.png - img100_synthetic.png)
                    - Post-train: 100 steps on all 100 synthetic images
                    - Merge: Automatic → `my_lora.safetensors` (Neon-improved!)
                    
                    **Image naming examples:**
                    - `img001_synthetic.png`, `img002_synthetic.jpg`
                    - Auto-detected and reused in subsequent runs!
                    """
                )
