# Import pipeline_stable_video_diffusion
from diffusers import StableVideoDiffusionPipeline
import torch

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16", num_frames = 2
)
pipe.enable_model_cpu_offload()

