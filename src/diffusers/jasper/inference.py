# Import pipeline_stable_video_diffusion
from diffusers import StableVideoDiffusionPipeline
import torch
from diffusers.utils import load_image, export_to_video

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
)
pipe.enable_model_cpu_offload()



image = load_image("/home/wisley/diffusers/pca.png")
image = image.resize((64,64))


generator = torch.manual_seed(42)
frames = pipe(image, decode_chunk_size=8, generator=generator).frames[0]

export_to_video(frames, "car_scene.mp4", fps=7)
