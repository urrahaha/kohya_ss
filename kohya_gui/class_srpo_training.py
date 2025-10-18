import gradio as gr
from typing import Tuple


class SRPOTraining:
    """
    SRPO (Style Reward Preference Optimization) Training Parameters
    
    Implements direct preference alignment for diffusion models using reward models.
    Based on: https://github.com/Tencent-Hunyuan/SRPO
    
    Attributes:
        srpo_enable: Enable SRPO training mode
        srpo_reward_model: Reward model selection (HPS-v2.1, PickScore, CLIP)
        srpo_timestep_length: Number of timesteps for SRPO training
        srpo_discount_pos: Discount range for positive branch
        srpo_discount_inv: Discount range for inversion branch
        srpo_train_timestep: Timestep range for training
        srpo_groundtruth_ratio: Groundtruth ratio for image recovery
        srpo_guidance_scale: Guidance scale for SRPO sampling
        srpo_reward_threshold: Reward threshold for ReLU loss
    """
    
    def __init__(self, headless: bool = False, config: dict = {}):
        """
        Initializes the SRPO Training parameters.
        
        Parameters:
            headless (bool): Run in headless mode without GUI
            config (dict): Configuration dictionary with saved values
        """
        self.headless = headless
        self.config = config
        
        with gr.Accordion("SRPO Training (Style Reward Preference Optimization)", open=False):
            with gr.Row():
                gr.Markdown(
                    """
                    ### SRPO Training Mode
                    **SRPO** directly aligns diffusion models using reward models (HPS-v2.1, PickScore, CLIP).
                    
                    **Key Features:**
                    - Direct preference alignment without traditional RL
                    - Faster training (~10 minutes for visible improvements)
                    - Free of reward hacking with proper regularization
                    - Supports dynamic controllable text conditions
                    
                    **Requirements:**
                    - Reward model checkpoints must be downloaded
                    - Works best with online generation (disables latent caching)
                    - Requires more VRAM for online rollout
                    
                    **Recommended for:** SDXL LoRA training to improve perceptual quality
                    """
                )
            
            with gr.Row():
                with gr.Column():
                    self.srpo_enable = gr.Checkbox(
                        label="Enable SRPO Training",
                        value=self.config.get("srpo.enable", False),
                        info="Enable Self-Reinforced Preference Optimization"
                    )
                
                with gr.Column():
                    self.srpo_use_reward_model = gr.Checkbox(
                        label="Use Reward Model",
                        value=self.config.get("srpo.use_reward_model", True),
                        info="Enable reward-based guidance. ⚠️ DISABLE for anime/illustration - existing models are biased toward photorealism"
                    )

            with gr.Row():
                with gr.Column():
                    self.srpo_use_diff2flow = gr.Checkbox(
                        label="SRPO Diff2Flow bridge (v→ε)",
                        value=self.config.get("srpo.use_diff2flow", False),
                        info="If UNet is v-parameterized, convert v to epsilon before SRPO one-step update."
                    )
                with gr.Column():
                    self.srpo_d2f_param = gr.Dropdown(
                        label="SRPO parameterization",
                        choices=["auto", "v", "eps"],
                        value=self.config.get("srpo.d2f_param", "auto"),
                        info="Override UNet parameterization for Diff2Flow bridge (auto = infer)."
                    )
        
            with gr.Row():
                with gr.Column():
                    self.srpo_reward_model = gr.Dropdown(
                        label="Reward Model",
                        choices=["HPS", "PickScore", "CLIP"],
                        value=self.config.get("srpo.reward_model", "HPS"),
                        info="Reward model for preference alignment (only used if enabled above)"
                    )
            
            with gr.Accordion("SRPO Core Parameters", open=False):
                with gr.Row():
                    self.srpo_timestep_length = gr.Slider(
                        label="Timestep Length",
                        value=self.config.get("srpo.timestep_length", 100),
                        minimum=10,
                        maximum=1000,
                        step=10,
                        info="Number of timesteps for SRPO training (default: 100)"
                    )
                    
                    self.srpo_guidance_scale = gr.Slider(
                        label="Guidance Scale",
                        value=self.config.get("srpo.guidance_scale", 3.5),
                        minimum=1.0,
                        maximum=20.0,
                        step=0.5,
                        info="CFG scale for SRPO sampling (default: 3.5)"
                    )
                
                with gr.Row():
                    self.srpo_groundtruth_ratio = gr.Slider(
                        label="Groundtruth Ratio",
                        value=self.config.get("srpo.groundtruth_ratio", 0.9),
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        info="Ratio for image recovery (default: 0.9)"
                    )
                    
                    self.srpo_reward_threshold = gr.Slider(
                        label="Reward Threshold",
                        value=self.config.get("srpo.reward_threshold", 0.7),
                        minimum=0.0,
                        maximum=2.0,
                        step=0.1,
                        info="ReLU threshold to prevent reward hacking (default: 0.7)"
                    )
            
            with gr.Accordion("SRPO Advanced Parameters", open=False):
                gr.Markdown(
                    """
                    **Discount Parameters:** Control reward weighting across timesteps
                    - **Positive Branch:** Rewards realistic/detailed styles
                    - **Inversion Branch:** Penalizes flat/oily textures
                    
                    **Timestep Range:** Focus training on specific diffusion stages
                    - Too early (>0.99): may cause structural distortions
                    - Too late: encourages color-based reward hacking
                    """
                )
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Discount (Positive Branch)**")
                        self.srpo_discount_pos_start = gr.Slider(
                            label="Start",
                            value=self.config.get("srpo.discount_pos_start", 0.1),
                            minimum=0.0,
                            maximum=1.0,
                            step=0.05,
                            info="Start discount for positive branch (default: 0.1)"
                        )
                        self.srpo_discount_pos_end = gr.Slider(
                            label="End",
                            value=self.config.get("srpo.discount_pos_end", 0.25),
                            minimum=0.0,
                            maximum=1.0,
                            step=0.05,
                            info="End discount for positive branch (default: 0.25)"
                        )
                    
                    with gr.Column():
                        gr.Markdown("**Discount (Inversion Branch)**")
                        self.srpo_discount_inv_start = gr.Slider(
                            label="Start",
                            value=self.config.get("srpo.discount_inv_start", 0.3),
                            minimum=0.0,
                            maximum=1.0,
                            step=0.05,
                            info="Start discount for inversion branch (default: 0.3)"
                        )
                        self.srpo_discount_inv_end = gr.Slider(
                            label="End",
                            value=self.config.get("srpo.discount_inv_end", 0.01),
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            info="End discount for inversion branch (default: 0.01)"
                        )
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Training Timestep Window**")
                        default_mode = self.config.get("srpo.train_timestep_mode", "percentage")
                        self.srpo_train_timestep_mode = gr.Dropdown(
                            label="Timestep Mode",
                            choices=["absolute", "percentage"],
                            value=default_mode,
                            info="Choose absolute timesteps or percentage window so settings stay consistent when you change timestep length."
                        )
                        self.srpo_train_timestep_start = gr.Slider(
                            label="Start Timestep",
                            value=self.config.get("srpo.train_timestep_start", 5),
                            minimum=0,
                            maximum=50,
                            step=1,
                            info="Start of training timestep range (default: 5)",
                            visible=default_mode == "absolute"
                        )
                        self.srpo_train_timestep_end = gr.Slider(
                            label="End Timestep",
                            value=self.config.get("srpo.train_timestep_end", 25),
                            minimum=0,
                            maximum=100,
                            step=1,
                            info="End of training timestep range (default: 25)",
                            visible=default_mode == "absolute"
                        )
                        self.srpo_train_timestep_start_pct = gr.Slider(
                            label="Start (%)",
                            value=self.config.get("srpo.train_timestep_start_pct", 0.15),
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            info="Lower bound as fraction of total steps (default: 0.15)",
                            visible=default_mode != "absolute"
                        )
                        self.srpo_train_timestep_end_pct = gr.Slider(
                            label="End (%)",
                            value=self.config.get("srpo.train_timestep_end_pct", 0.6),
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            info="Upper bound as fraction of total steps (default: 0.6)",
                            visible=default_mode != "absolute"
                        )

                        def _set_srpo_timestep_visibility(mode):
                            show_absolute = mode == "absolute"
                            show_pct = mode == "percentage"
                            return (
                                gr.update(visible=show_absolute),
                                gr.update(visible=show_absolute),
                                gr.update(visible=show_pct),
                                gr.update(visible=show_pct),
                            )

                        self.srpo_train_timestep_mode.change(
                            _set_srpo_timestep_visibility,
                            inputs=[self.srpo_train_timestep_mode],
                            outputs=[
                                self.srpo_train_timestep_start,
                                self.srpo_train_timestep_end,
                                self.srpo_train_timestep_start_pct,
                                self.srpo_train_timestep_end_pct,
                            ],
                        )
            
            with gr.Accordion("Custom Control Words (Style Preference)", open=False):
                gr.Markdown(
                    """
                    **Custom Control Words** allow you to steer SRPO toward specific styles.
                    
                    - **Positive Controls**: Words describing desired qualities (e.g., for anime: "cel-shaded", "vibrant", "stylized")
                    - **Negative Controls**: Words describing undesired qualities (e.g., for anime: "photorealistic", "3D render", "hyperrealistic")
                    
                    Enter comma-separated words. The training will cycle through them randomly.
                    Leave blank to use defaults (optimized for photorealism).
                    
                    **Examples:**
                    - **Anime**: Positive: `cel-shaded, vibrant colors, stylized, sharp lineart` / Negative: `photorealistic, 3D, realistic lighting, depth of field`
                    - **Illustration**: Positive: `artistic, painterly, expressive brushwork` / Negative: `photograph, raw photo, candid shot`
                    - **Realism** (default): Positive: `detailed, natural-lighting, real, high resolution` / Negative: `flat, anime, painting, oily, cg artwork`
                    """
                )
                
                with gr.Row():
                    self.srpo_positive_controls = gr.Textbox(
                        label="Positive Control Words",
                        placeholder="e.g., detailed, natural-lighting, real, photorealistic (comma-separated)",
                        value=self.config.get("srpo.positive_controls", ""),
                        lines=2,
                        info="Words describing desired style qualities. Leave empty for defaults."
                    )
                
                with gr.Row():
                    self.srpo_negative_controls = gr.Textbox(
                        label="Negative Control Words",
                        placeholder="e.g., flat, anime, painting, cg artwork (comma-separated)",
                        value=self.config.get("srpo.negative_controls", ""),
                        lines=2,
                        info="Words describing undesired style qualities. Leave empty for defaults."
                    )
