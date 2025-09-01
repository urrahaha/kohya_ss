import gradio as gr
from .class_gui_config import KohyaSSGUIConfig
from .custom_logging import setup_logging

log = setup_logging()

class SDXLParameters:
    def __init__(
        self,
        sdxl_checkbox: gr.Checkbox,
        show_sdxl_cache_text_encoder_outputs: bool = True,
        config: KohyaSSGUIConfig = {},
        trainer: str = "",
    ):
        self.sdxl_checkbox = sdxl_checkbox
        self.show_sdxl_cache_text_encoder_outputs = show_sdxl_cache_text_encoder_outputs
        self.config = config
        self.trainer = trainer
        
        self.initialize_accordion()

    def initialize_accordion(self):
        with gr.Accordion(
            visible=False, open=True, label="SDXL Specific Parameters"
        ) as self.sdxl_row:
            with gr.Row():
                self.sdxl_cache_text_encoder_outputs = gr.Checkbox(
                    label="Cache text encoder outputs",
                    info="Cache the outputs of the text encoders. This option is useful to reduce the GPU memory usage. This option cannot be used with options for shuffling or dropping the captions.",
                    value=self.config.get("sdxl.sdxl_cache_text_encoder_outputs", False),
                    visible=self.show_sdxl_cache_text_encoder_outputs,
                )
                self.sdxl_no_half_vae = gr.Checkbox(
                    label="No half VAE",
                    info="Disable the half-precision (mixed-precision) VAE. VAE for SDXL seems to produce NaNs in some cases. This option is useful to avoid the NaNs.",
                    value=self.config.get("sdxl.sdxl_no_half_vae", False),
                )
                self.sdxl_flow_matching = gr.Checkbox(
                    label="Flow Matching",
                    info="Train SDXL using Flow Matching loss (uses sd-scripts/sdxl_train_network_fm.py or sd-scripts/sdxl_train_fm.py). Also works for LoRA, although it should be used on a base model that's already trained with Flow Matching.",
                    value=self.config.get("sdxl.sdxl_flow_matching", False),
                )
            
            # Flow Matching specific parameters
            with gr.Row():
                self.fm_shift = gr.Number(
                    label="FM shift",
                    info="FlowMatch scheduler shift value (recommended ~3.0)",
                    value=self.config.get("sdxl.fm_shift", 3.0),
                    step=0.1,
                )
                self.fm_logit_mean = gr.Number(
                    label="FM logit mean",
                    info="Logit-normal mean for timestep sampling",
                    value=self.config.get("sdxl.fm_logit_mean", 0.0),
                    step=0.1,
                )
                self.fm_logit_std = gr.Number(
                    label="FM logit std",
                    info="Logit-normal std for timestep sampling",
                    value=self.config.get("sdxl.fm_logit_std", 1.0),
                    step=0.1,
                )
                self.fm_mode_scale = gr.Number(
                    label="FM mode scale",
                    info="Mode scale used in logit-normal density",
                    value=self.config.get("sdxl.fm_mode_scale", 1.29),
                    step=0.01,
                )
                self.fused_backward_pass = gr.Checkbox(
                    label="Fused backward pass",
                    info="Enable fused backward pass. This option is useful to reduce the GPU memory usage. Can't be used if Fused optimizer groups is > 0. Only AdaFactor is supported",
                    value=self.config.get("sdxl.fused_backward_pass", False),
                    visible=self.trainer == "finetune" or self.trainer == "dreambooth",
                )
                self.fused_optimizer_groups = gr.Number(
                    label="Fused optimizer groups",
                    info="Number of optimizer groups to fuse. This option is useful to reduce the GPU memory usage. Can't be used if Fused backward pass is enabled. Since the effect is limited to a certain number, it is recommended to specify 4-10.",
                    value=self.config.get("sdxl.fused_optimizer_groups", 0),
                    minimum=0,
                    step=1,
                    visible=self.trainer == "finetune" or self.trainer == "dreambooth",
                )
                self.disable_mmap_load_safetensors = gr.Checkbox(
                    label="Disable mmap load safe tensors",
                    info="Disable memory mapping when loading the model's .safetensors in SDXL.",
                    value=self.config.get("sdxl.disable_mmap_load_safetensors", False),
                )
                
            # Profiler (TensorBoard) configuration for SDXL / Flow Matching
            with gr.Row():
                self.profile = gr.Checkbox(
                    label="Enable Profiler (TensorBoard)",
                    info="Enable torch.profiler to record TensorBoard traces during training. Traces will be written to the specified directory.",
                    value=self.config.get("sdxl.profile", False),
                )
                self.profile_dir = gr.Textbox(
                    label="Profiler log directory",
                    info="Directory to save profiler traces (TensorBoard format).",
                    value=self.config.get("sdxl.profile_dir", "profiler_logs"),
                    interactive=self.config.get("sdxl.profile", False),
                )
                self.profile_wait = gr.Number(
                    label="Profiler wait steps",
                    info="Number of wait steps before warmup in profiler schedule.",
                    value=self.config.get("sdxl.profile_wait", 1),
                    step=1,
                    minimum=0,
                    interactive=self.config.get("sdxl.profile", False),
                )
                self.profile_warmup = gr.Number(
                    label="Profiler warmup steps",
                    info="Number of warmup steps before active profiling.",
                    value=self.config.get("sdxl.profile_warmup", 1),
                    step=1,
                    minimum=0,
                    interactive=self.config.get("sdxl.profile", False),
                )
                self.profile_active = gr.Number(
                    label="Profiler active steps",
                    info="Number of active profiling steps to record.",
                    value=self.config.get("sdxl.profile_active", 5),
                    step=1,
                    minimum=1,
                    interactive=self.config.get("sdxl.profile", False),
                )

            with gr.Row():
                self.naive_fm = gr.Checkbox(
                    label="Naive Flow Matching",
                    info="Use naive Flow Matching instead of Diff2Flow (trains from scratch). By default, Diff2Flow is used for better knowledge transfer.",
                    value=self.config.get("sdxl.naive_fm", False),
                )
                self.diffusion_parameterization = gr.Dropdown(
                    label="Diffusion Parameterization",
                    info="Diffusion model parameterization for Diff2Flow. SDXL typically uses v-parameterization.",
                    choices=["v", "eps"],
                    value=self.config.get("sdxl.diffusion_parameterization", "v"),
                )

            with gr.Row():
                self.fused_backward_pass.change(
                    lambda fused_backward_pass: gr.Number(
                        interactive=not fused_backward_pass
                    ),
                    inputs=[self.fused_backward_pass],
                    outputs=[self.fused_optimizer_groups],
                )
                self.fused_optimizer_groups.change(
                    lambda fused_optimizer_groups: gr.Checkbox(
                        interactive=fused_optimizer_groups == 0
                    ),
                    inputs=[self.fused_optimizer_groups],
                    outputs=[self.fused_backward_pass],
                )


        self.sdxl_checkbox.change(
            lambda sdxl_checkbox: gr.Accordion(visible=sdxl_checkbox),
            inputs=[self.sdxl_checkbox],
            outputs=[self.sdxl_row],
        )

        # Toggle profiler fields interactivity based on 'Enable Profiler' and 'Flow Matching'
        def _prof_interactive(enable_profiler: bool, enable_fm: bool):
            enabled = bool(enable_profiler and enable_fm)
            return (
                gr.Textbox(interactive=enabled),
                gr.Number(interactive=enabled),
                gr.Number(interactive=enabled),
                gr.Number(interactive=enabled),
            )

        self.profile.change(
            _prof_interactive,
            inputs=[self.profile, self.sdxl_flow_matching],
            outputs=[
                self.profile_dir,
                self.profile_wait,
                self.profile_warmup,
                self.profile_active,
            ],
        )

        self.sdxl_flow_matching.change(
            _prof_interactive,
            inputs=[self.profile, self.sdxl_flow_matching],
            outputs=[
                self.profile_dir,
                self.profile_wait,
                self.profile_warmup,
                self.profile_active,
            ],
        )

        # Disable diffusion parameterization when Naive FM is enabled
        def _toggle_d2f_param(naive: bool):
            return gr.Dropdown(interactive=not naive)

        self.naive_fm.change(
            _toggle_d2f_param,
            inputs=[self.naive_fm],
            outputs=[self.diffusion_parameterization],
        )
