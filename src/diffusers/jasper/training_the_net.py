import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path
from typing import Optional
import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
import shutil
import diffusers
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
import pdb
from diffusers import StableVideoDiffusionPipeline
from einops import rearrange
from diffusers.models.unets import UNetSpatioTemporalConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available, make_image_grid
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion_with_controlnet import StableVideoDiffusionPipelineWithControlNet,SpatioTemporalControlNet, CustomConditioningNet, SpatioTemporalControlNetOutput
import gc
from diffusers import DiffusionPipeline


from diffusers.image_processor import VaeImageProcessor


# accelerate config
# accelerate config default

from diffusers.utils import load_image, export_to_video

if is_wandb_available():
    import wandb

logger = get_logger(__name__)


def print_sum_of_weights_for_zero_initialized_layers(model):
    layers_of_interest = [model.controlnet_mid_block] + list(model.controlnet_down_blocks)
    for i, layer in enumerate(layers_of_interest):
        if isinstance(layer, nn.Conv2d):
            weight_sum = layer.weight.sum().item()
            print(f"Layer {i} weight sum: {weight_sum}")


def rand_log_normal(shape, loc=0., scale=1., device='cpu', dtype=torch.float32):
    """Draws samples from an lognormal distribution."""
    u = torch.rand(shape, dtype=dtype, device=device) * (1 - 2e-7) + 1e-7
    return torch.distributions.Normal(loc, scale).icdf(u).exp()

def tensor_to_vae_latent(t, vae):
    video_length = t.shape[1]

    t = rearrange(t, "b f c h w -> (b f) c h w")
    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
    latents = latents * vae.config.scaling_factor

    return latents


def _append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]

def validation_video(batch, pipe, control_net_trained, unet, tokenizer, text_encoder, step):
    with torch.no_grad():
        pipe_with_controlnet = StableVideoDiffusionPipelineWithControlNet(
        vae = pipe.vae,
        image_encoder = pipe.image_encoder,
        unet=unet,
        scheduler=pipe.scheduler,
        feature_extractor=pipe.feature_extractor,
        controlnet=control_net_trained,
        tokenizer = tokenizer,
        text_encoder = text_encoder
        )

        # Normal
        prompt = "A driving scene during the night, with rainy weather in boston-seaport"
        # prompt = batch['caption']
        pseudo_sample = batch['conditioning'].flatten(0,1)
        # Define a simple torch generator
        generator = torch.Generator().manual_seed(42)
        image = batch['reference_image']
        # Save the image
        image.save(f"/mnt/e/13_Jasper_diffused_samples/training/output/images/image_{step}.png")


        # Random sample
        random_sample = torch.ones_like(pseudo_sample, device=pseudo_sample.device, dtype=pseudo_sample.dtype)
        frames_random = pipe_with_controlnet( image = image,num_frames = 14, prompt=prompt, conditioning_image = random_sample,  decode_chunk_size=8, generator=generator).frames[0]
        frames = pipe_with_controlnet( image = image,num_frames = 14, prompt=prompt, conditioning_image = pseudo_sample,  decode_chunk_size=8, generator=generator).frames[0]

        export_to_video(frames, f"/mnt/e/13_Jasper_diffused_samples/training/output/vids/videojap_{step}.avi", fps=7)
        export_to_video(frames_random, f"/mnt/e/13_Jasper_diffused_samples/training/output/vids/videojap_random_{step}.avi", fps=7)
        return


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            # image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- controlnet
inference: true
---
    """
    model_card = f"""
# controlnet-{repo_id}

These are controlnet weights trained on {base_model} with new type of conditioning.
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json

class DiffusionDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.image_factor_x = 360 / 320    
        self.image_factor_y = 640 / 512
        self.transform = transforms.Compose([
            transforms.Resize((int(360/self.image_factor_y) , int(640/self.image_factor_x))),
            transforms.CenterCrop((320, 512)),
        ])
        self.image_processor = VaeImageProcessor(vae_scale_factor=8)

    def __len__(self):
        # Assuming each set of ground truths represents a separate sample
        return len(self.data['ground_truth'])

    def __getitem__(self, idx):
        # Processing ground truth images
        
        ground_truth_images = [self.transform(Image.open(path)) for path in self.data['ground_truth'][idx]]
        ground_truth_images = self.image_processor.preprocess(image = ground_truth_images, height = 320, width = 512)

        # prescan_images = [self.transform(Image.open(path)) for path in self.data['prescan_images'][idx]]
        # prescan_images = self.image_processor.preprocess(image = prescan_images, height = 320, width = 512)

        # Processing conditioning images set one (assuming RGB, 4 channels after conversion)
        conditioning_images_one = [self.transform(Image.open(path).convert("RGB")) for path in self.data['conditioning_images_one'][idx]]
        conditioning_images_one = self.image_processor.preprocess(image = conditioning_images_one, height = 320, width = 512)

        # Processing conditioning images set two (assuming grayscale, converted to RGB to match dimensions)
        # conditioning_images_two = [self.transform(Image.open(path)) for path in self.data['conditioning_images_two'][idx]]
        # conditioning_images_two = self.image_processor.preprocess(image = conditioning_images_two, height = 320, width = 512)
        
        # Concatenating condition one and two images along the channel dimension
        # conditioned_images = [torch.cat((img_one, img_two), dim=0) for img_one, img_two in zip(conditioning_images_one, conditioning_images_two)]

        # Processing reference images (single per scene, matched by index)
        # reference_image = self.transform(Image.open(self.data['ground_truth'][idx][0]))

        # Retrieving the corresponding caption
        caption = self.data['caption'][idx][0]
        reference_image = self.transform(Image.open(self.data['ground_truth'][idx][0]))

        

        return {
            "ground_truth": ground_truth_images,
            "conditioning": conditioning_images_one,
            "caption": caption,
            "reference_image": reference_image,
            # "prescan_images": prescan_images
        }

def collate_fn(batch):
    ground_truth = torch.stack([item['ground_truth'] for item in batch])
    conditioning = torch.stack([item['conditioning'] for item in batch])
    captions = [item['caption'] for item in batch]  # List of strings, no need to stack
    reference_images = [item['reference_image'] for item in batch]
    

    return {
        "ground_truth": ground_truth,
        "conditioning": conditioning,
        "caption": captions[0],
        "reference_image": reference_images[0],
    }

def encode_image(pixel_values, feature_extractor, image_encoder, accelerator, weight_dtype):
    # pixel: [-1, 1]
    pixel_values = _resize_with_antialiasing(pixel_values, (224, 224))
    # We unnormalize it after resizing.
    pixel_values = (pixel_values + 1.0) / 2.0

    # Normalize the image with for CLIP input
    pixel_values = feature_extractor(
        images=pixel_values,
        do_normalize=True,
        do_center_crop=False,
        do_resize=False,
        do_rescale=False,
        return_tensors="pt",
    ).pixel_values

    pixel_values = pixel_values.to(
        device=accelerator.device, dtype=weight_dtype)
    image_embeddings = image_encoder(pixel_values).image_embeds
    return image_embeddings

def _get_add_time_ids(
    fps,
    motion_bucket_id,
    noise_aug_strength,
    dtype,
    batch_size,
    unet
):
    add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

    # passed_add_embed_dim = unet.module.config.addition_time_embed_dim * \
    #     len(add_time_ids)
    # expected_add_embed_dim = unet.module.add_embedding.linear_1.in_features

    # if expected_add_embed_dim != passed_add_embed_dim:
    #     raise ValueError(
    #         f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
    #     )

    add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
    add_time_ids = add_time_ids.repeat(batch_size, 1)
    return add_time_ids

def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    # First, we have to determine sigma
    # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )

    # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
    # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
    # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    # Make sure it is odd
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(
        input, size=size, mode=interpolation, align_corners=align_corners)
    return output


def _compute_padding(kernel_size):
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def _filter2d(input, kernel):
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(
        device=input.device, dtype=input.dtype)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    padding_shape: list[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = torch.nn.functional.conv2d(
        input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    out = output.view(b, c, h, w)
    return out


def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]

    x = (torch.arange(window_size, device=sigma.device,
         dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)


def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out



def main(output_dir, logging_dir, gradient_accumulation_steps, mixed_precision, hub_model_id):
    train_unet = False 

    logging_dir = Path(output_dir, logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision="fp16",
        log_with= "wandb",
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if accelerator.is_main_process:
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        # Pushing to the hub
        if True:
            hub_token = "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"
            repo_id = create_repo(
                repo_id=hub_model_id or Path(output_dir).name, exist_ok=True, token=hub_token
            ).repo_id

    # Load the models
    weight_dtype = torch.float16
    # if accelerator.mixed_precision == "fp16":
    #     weight_dtype = torch.float16
    # elif accelerator.mixed_precision == "bf16":
    #     weight_dtype = torch.bfloat16

    # Importing the pipelines
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid", torch_dtype=torch.float16, variant="fp16"
    )
    pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2")


    # # Getting the models

    if train_unet:
        my_net =  UNetSpatioTemporalConditionModel()
        unet_weights = pipe.unet.state_dict()
        my_net.load_state_dict(unet_weights)

        # checkpoint = torch.load('/mnt/e/13_Jasper_diffused_samples/training/unet_uncoditional/test/model_checkpoint_239.ckpt') 
        # my_net.load_state_dict(checkpoint['unet_state_dict']) 
        
        
        control_net = SpatioTemporalControlNet.from_unet(my_net)
        control_net = control_net.half()   
    else:
        my_net =  UNetSpatioTemporalConditionModel()
        unet_weights = pipe.unet.state_dict()
        my_net.load_state_dict(unet_weights)

        control_net = SpatioTemporalControlNet.from_unet(my_net)    
        control_net = control_net.half()   
        
    # Make them f16
    my_net = my_net.half()

    vae = pipe.vae
    image_encoder = pipe.image_encoder
    unet=my_net
    scheduler=pipe.scheduler
    feature_extractor=pipe.feature_extractor
    controlnet=control_net
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder

    # Initialize a controlnet_pipeline so we can use al the functinos
    pipe_with_controlnet = StableVideoDiffusionPipelineWithControlNet(
    vae = pipe.vae,
    image_encoder = image_encoder,
    unet=my_net,
    scheduler=scheduler,
    feature_extractor= feature_extractor,
    controlnet=control_net,
    tokenizer = tokenizer,
    text_encoder = text_encoder
)
    
    # move the pipeline to the device
    pipe_with_controlnet= pipe_with_controlnet.to(device= accelerator.device, dtype=torch.float16)
 
    # set some variables
    max_train_steps = 50000 
    learning_rate = 1e-5
    lr_scheduler = "constant"
    lr_warmup_steps = 0
    num_train_epochs = 50
    train_batch_size = 1
    adam_epsilon = 1e-08
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_weight_decay = 1e-2
    max_grad_norm = 1.0

    # Taken from [Sayak Paul's Diffusers PR #6511](https://github.com/huggingface/diffusers/pull/6511/files)
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                i = len(weights) - 1

                while len(weights) > 0:
                    weights.pop()
                    model = models[i]

                    sub_dir = "controlnet"
                    model.save_pretrained(os.path.join(output_dir, sub_dir))

                    i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)
    
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if train_unet:
    
        
        unet.requires_grad_(True)
        unet.train()
        unet.enable_gradient_checkpointing()
    else:
        controlnet.train()
        unet.requires_grad_(False)
        controlnet.enable_gradient_checkpointing()


    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if True:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW
    # optimizer_class = torch.optim.AdamW

    # Optimizer creation
    if train_unet:
        params_to_optimize = unet.parameters()
    else:
        params_to_optimize = controlnet.parameters()

    optimizer = optimizer_class(
        params_to_optimize,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    train_dataset = DiffusionDataset(json_path='/home/wisley/custom_diffusers_library/src/diffusers/jasper/jappie_seg.json')

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=1,  # Or your preferred batch size
        num_workers=0,  # Adjust based on your setup
    )

    validation_dataset = DiffusionDataset(json_path='/home/wisley/custom_diffusers_library/src/diffusers/jasper/validation.json')

    validation_dataloader = DataLoader(
        validation_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=1,  # Or your preferred batch size
        num_workers=0,  # Adjust based on your setup
    )

    data_iter = iter(validation_dataloader)
    batch_validation = next(data_iter)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    print(f"accelerator num processes: {accelerator.num_processes}")
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
        num_cycles=1,
        power=1,
    )

    # Go train my boy
    if train_unet: 
        unet.requires_grad_(True)
        for param in controlnet.parameters():
            param.requires_grad = False


        for name, param in unet.named_parameters():
            if "temporal_transformer_block" not in name:
                # Set the desired attribute or action here. For example, to make the parameter trainable:
                param.requires_grad = False 
            else:
                param.requires_grad = True


        # Prepare everything with our `accelerator`.
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )
    else:
        controlnet.requires_grad_(True)
        for param in controlnet.parameters():
            param.requires_grad = True

        # Do not train the temporal transformer block
        for name, param in controlnet.named_parameters():
            if "temporal" not in name:
                # Set the desired attribute or action here. For example, to make the parameter trainable:
                param.requires_grad = True
            else:
                param.requires_grad = False

        for param in unet.parameters():
            param.requires_grad = False

        # Prepare everything with our `accelerator`.
        controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            controlnet, optimizer, train_dataloader, lr_scheduler
        )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    if True:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly")


    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    if overrode_max_train_steps:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        training_config = {
                "max_train_steps": 50000,
                "learning_rate": 2e-5,
                "lr_scheduler": "constant",
                "lr_warmup_steps": 0,
                "num_train_epochs": 50,
                "train_batch_size": 1,
                "adam_epsilon": 1e-08,
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "adam_weight_decay": 1e-2,
                "max_grad_norm": 1.0,
                # Add placeholders for any other arguments required for tracker initialization
                "tracker_project_name": "FinalVideo",
                "validation_prompt": None,  # Placeholder for the argument to be popped
                "validation_image": None    # Placeholder for the argument to be popped
            }
        tracker_config = training_config.copy()

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompt")
        tracker_config.pop("validation_image")

        accelerator.init_trackers(tracker_config["tracker_project_name"], config=tracker_config)

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if False:

        # Get the most recent checkpoint
        dirs = os.listdir(output_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None


        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(output_dir, path))
        global_step = int(path.split("-")[1])

        initial_global_step = global_step
        first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )



    scheduler.set_timesteps(25, device=accelerator.device)
    timesteps = scheduler.timesteps
    generator = torch.Generator(device=accelerator.device).manual_seed(42)

    guidance_scale = None
    image_logs = None
    for epoch in range(first_epoch, num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet if train_unet else controlnet):

                # # Get all the inputs
                inputs = pipe_with_controlnet.prepare_input_for_forward(batch['reference_image'], batch['caption'], batch['conditioning'], num_frames=14)


                pixel_values = batch["ground_truth"].to(weight_dtype).to(
                    accelerator.device, non_blocking=True
                )
                # pixel_values_conditional = batch["prescan_images"].to(weight_dtype).to(
                #     accelerator.device, non_blocking=True
                # )
                conditional_pixel_values = pixel_values[:, 0:1, :, :, :]
                

                latents = tensor_to_vae_latent(pixel_values, vae)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                cond_sigmas = rand_log_normal(shape=[bsz,], loc=-3.0, scale=0.5).to(latents)
                noise_aug_strength = cond_sigmas[0] # TODO: support batch > 1
                cond_sigmas = cond_sigmas[:, None, None, None, None]
                conditional_pixel_values = \
                    torch.randn_like(conditional_pixel_values) * cond_sigmas + conditional_pixel_values
                conditional_latents = tensor_to_vae_latent(conditional_pixel_values, vae)[:, 0, :, :, :]
                conditional_latents = conditional_latents / vae.config.scaling_factor

                # Sample a random timestep for each image
                # P_mean=0.7 P_std=1.6
                sigmas = rand_log_normal(shape=[bsz,], loc=0.7, scale=1.6).to(latents.device)
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                sigmas = sigmas[:, None, None, None, None]
                noisy_latents = latents + noise * sigmas
                

                timesteps = torch.Tensor(
                    [0.25 * sigma.log() for sigma in sigmas]).to(accelerator.device)
                timesteps = timesteps.to(device=accelerator.device, dtype=weight_dtype)

                inp_noisy_latents = noisy_latents / ((sigmas**2 + 1) ** 0.5)
                
                encoder_hidden_states = encode_image(
                    pixel_values[:, 0, :, :, :].float(), feature_extractor=feature_extractor, image_encoder=image_encoder, accelerator=accelerator, weight_dtype=weight_dtype)
                
                added_time_ids = _get_add_time_ids(
                    7, # fixed
                    127, # motion_bucket_id = 127, fixed
                    noise_aug_strength, # noise_aug_strength == cond_sigmas
                    encoder_hidden_states.dtype,
                    bsz,
                    unet
                )
                added_time_ids = added_time_ids.to(device=accelerator.device, dtype=weight_dtype)


                if True:
                    random_p = torch.rand(
                        bsz, device=torch.device("cuda"), generator=generator)
                    # Sample masks for the edit prompts.
                    prompt_mask = random_p < 2 * 0.05
                    prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                    # Final text conditioning.
                    null_conditioning = torch.zeros_like(encoder_hidden_states)
                    encoder_hidden_states = torch.where(
                        prompt_mask, null_conditioning.unsqueeze(1), encoder_hidden_states.unsqueeze(1))
                    # Sample masks for the original images.
                    image_mask_dtype = weight_dtype
                    image_mask = 1 - (
                        (random_p >= 0.05).to(
                            image_mask_dtype)
                        * (random_p < 3 * 0.05).to(image_mask_dtype)
                    )
                    image_mask = image_mask.reshape(bsz, 1, 1, 1)
                    # Final image conditioning.
                    conditional_latents = image_mask * conditional_latents

                conditional_latents = conditional_latents.unsqueeze(
                    1).repeat(1, noisy_latents.shape[1], 1, 1, 1)
                inp_noisy_latents = torch.cat(
                    [inp_noisy_latents, conditional_latents], dim=2)
                inp_noisy_latents = inp_noisy_latents.to(device=accelerator.device, dtype=weight_dtype)


                # Make sure encoder hidden states and the added time ids are on the same device
                encoder_hidden_states = encoder_hidden_states.to(device=accelerator.device, dtype=weight_dtype)

                if train_unet:
                    model_pred = unet(
                        inp_noisy_latents, timesteps,encoder_hidden_states=inputs["unet_encoder_hidden_states"][:1], added_time_ids = added_time_ids).sample
                                    
                    

                else:
                    down_block_res_samples, mid_block_res_sample = controlnet.forward(
                        inp_noisy_latents,
                        timesteps,
                        encoder_hidden_states= inputs["controlnet_encoder_hidden_states"][:1], 
                        added_time_ids= added_time_ids,
                        return_dict=False,
                        controlnet_condition = inputs['controlnet_condition'].flatten(0,1),
                        training = True
                    )
                # predict the noise residual


                
                    model_pred = unet(
                        inp_noisy_latents,
                        timesteps,
                        encoder_hidden_states= encoder_hidden_states,
                        added_time_ids= added_time_ids,
                        down_block_additional_residuals= down_block_res_samples,
                        mid_block_additional_residual = mid_block_res_sample,
                    ).sample


                
                target = latents

                # Denoise the latents
                c_out = -sigmas / ((sigmas**2 + 1)**0.5)
                c_skip = 1 / (sigmas**2 + 1)
                denoised_latents = model_pred * c_out + c_skip * noisy_latents
                weighing = (1 + sigmas ** 2) * (sigmas**-2.0)

                # MSE loss
                loss = torch.mean(
                    (weighing.float() * (denoised_latents.float() -
                     target.float()) ** 2).reshape(target.shape[0], -1),
                    dim=1,
                )
                loss = loss.mean()

                # loss = F.mse_loss(noise_pred.float(), noise_total.float(), reduction="mean")
                print(f"this is the loss: {loss}")

                accelerator.backward(loss)

                # Perform gradient clipping using accelerator's utility method if available
                # Note: This step might need adjustment based on the Accelerator's version and capabilities
                if accelerator.sync_gradients:
                    params_to_clip = controlnet.parameters() if not train_unet else unet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)

                
                # print the change of the params
                cumulative_grad_sum = 0.0

                for param in controlnet.parameters() if not train_unet else unet.parameters():
                    if param.grad is not None:
                        # Sum the squares of gradients
                        cumulative_grad_sum += param.grad.data.norm(2).item() ** 2

                # Take the square root to go back to the original scale
                cumulative_grad_sum = cumulative_grad_sum ** 0.5

                print(f"Cumulative Gradient Step (Norm): {cumulative_grad_sum}")


                optimizer.step()
                lr_scheduler.step()

                # Zero the parameter gradients
                optimizer.zero_grad(set_to_none=True)

                # Step the learning rate scheduler

             
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % 20 == 0 or global_step == 1:
                        try:
                            validation_video(batch_validation, pipe_with_controlnet, controlnet, unet, tokenizer, text_encoder, global_step) 
                        except Exception as e:
                            print(e)

                    if global_step % 200 == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if True :
                            checkpoints = os.listdir(output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                        # delete the oldest checkpoint if there are more than 4
                        try:
                            if len(checkpoints) > 4:
                                shutil.rmtree(f'{output_dir}/{checkpoints[0]}')
                        except Exception as e:
                            print(e)

                        checkpoint = {
                            'epoch': epoch,
                            'unet_state_dict': unet.state_dict(),
                            'model_state_dict': controlnet.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss,
                            # You can include more components as needed
                        }

                        # If there are more then 4 model_checkpoints_{step}.ckpt files in the folder delete the oldest  with the lowest value for step

                        # First get the amount of files in the dir
                        dir = f'{output_dir}/test'
                        files = os.listdir(dir)
                        files = [f for f in files if f.startswith("model_checkpoint")]
                        files = sorted(files, key=lambda x: int(x.split("_")[2].split(".")[0]))

                        try:
                            if len(files) > 4:
                                # delete the file with th lowest value for step
                                os.remove(f'{output_dir}/test/{files[0]}')
                        except Exception as e:
                            print(e)
                                            
                            
                     
                        torch.save(checkpoint, f'{output_dir}/test/model_checkpoint_{global_step}.ckpt')



            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet = unwrap_model(controlnet)
        controlnet.save_pretrained(output_dir)

        if True:
            # save_model_card(
            #     repo_id,
            #     image_logs=image_logs,
            #     base_model=pretrained_model_name_or_path,
            #     repo_folder=output_dir,
            # )
            upload_folder(
                repo_id=repo_id,
                folder_path=output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()

if __name__ == "__main__":
    import torch
    print(torch.__version__)
    print(torch.version.cuda)
    import xformers
    print(xformers.__version__)


    main(
        output_dir="/mnt/e/13_Jasper_diffused_samples/training/a",
        logging_dir="/mnt/e/13_Jasper_diffused_samples/training/logs",
        gradient_accumulation_steps=4,
        mixed_precision="fp16",
        hub_model_id="temporalControlNet",
    )