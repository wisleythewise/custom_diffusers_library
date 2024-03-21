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
from diffusers import StableVideoDiffusionPipeline

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
        
        prompt = "A driving scene during the night, with rainy weather in boston-seaport"
        # prompt = batch['caption']
        pseudo_sample = batch['conditioning'][:14]
        # Define a simple torch generator
        generator = torch.Generator().manual_seed(42)
        image = batch['reference_image']

        frames = pipe_with_controlnet( image = image,num_frames = 14, prompt=prompt, conditioning_image = pseudo_sample,  decode_chunk_size=8, generator=generator).frames[0]

        export_to_video(frames, f"/mnt/e/13_Jasper_diffused_samples/training/output/vids/videojap_{step}.avi", fps=7)
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
        self.transform = transforms.Compose([
            transforms.Resize((288, 512)),
            transforms.CenterCrop((288, 512)),
        ])
        self.image_processor = VaeImageProcessor(vae_scale_factor=8)

    def __len__(self):
        # Assuming each set of ground truths represents a separate sample
        return len(self.data['ground_truth'])

    def __getitem__(self, idx):
        # Processing ground truth images
        
        ground_truth_images = [self.transform(Image.open(path)) for path in self.data['ground_truth'][idx]]
        ground_truth_images = self.image_processor.preprocess(image = ground_truth_images, height = 288, width = 512)

        prescan_images = [self.transform(Image.open(path)) for path in self.data['prescan_images'][idx]]
        prescan_images = self.image_processor.preprocess(image = prescan_images, height = 288, width = 512)

        # Processing conditioning images set one (assuming RGB, 4 channels after conversion)
        conditioning_images_one = [self.transform(Image.open(path)) for path in self.data['conditioning_images_one'][idx]]
        conditioning_images_one = self.image_processor.preprocess(image = conditioning_images_one, height = 288, width = 512)

        # Processing conditioning images set two (assuming grayscale, converted to RGB to match dimensions)
        conditioning_images_two = [self.transform(Image.open(path)) for path in self.data['conditioning_images_two'][idx]]
        conditioning_images_two = self.image_processor.preprocess(image = conditioning_images_two, height = 288, width = 512)
        
        # Concatenating condition one and two images along the channel dimension
        conditioned_images = [torch.cat((img_one, img_two), dim=0) for img_one, img_two in zip(conditioning_images_one, conditioning_images_two)]

        # Processing reference images (single per scene, matched by index)
        reference_image = self.transform(Image.open(self.data['reference_image'][idx][0]))

        # Retrieving the corresponding caption
        caption = self.data['caption'][idx][0]

        

        return {
            "ground_truth": ground_truth_images,
            "conditioning": torch.stack(conditioned_images),
            "caption": caption,
            "reference_image": reference_image,
            "prescan_images": prescan_images
        }

def collate_fn(batch):
    ground_truth = torch.stack([item['ground_truth'] for item in batch])
    conditioning = torch.stack([item['conditioning'] for item in batch])
    prescan_images = torch.stack([item['prescan_images'] for item in batch])
    captions = [item['caption'] for item in batch]  # List of strings, no need to stack
    reference_images = [item['reference_image'] for item in batch]
    

    return {
        "ground_truth": ground_truth.flatten(0, 1),
        "conditioning": conditioning.flatten(0, 1),
        "caption": captions[0],
        "reference_image": reference_images[0],
        "prescan_images": prescan_images.flatten(0, 1)
    }

def _encode_vae_image(

        image: torch.Tensor,
        vae
    ):

        with torch.no_grad(): 

            # print(f"this is the shape of the image: {image.shape}")
            image = image.to(device=vae.device, dtype=vae.dtype)
            image_latents = vae.encode(image.to(device=vae.device)).latent_dist.sample()

            image_latents = torch.nn.functional.interpolate(image_latents, size=(36,64), mode="nearest")


            # duplicate image_latents for each generation per prompt, using mps friendly method
            image_latents = image_latents.repeat(1, 1, 1, 1)

            return image_latents

def encode_batch(images, vae ):
    outputs = []  # Initialize an empty list to store each output


    # Loop through each image in the pseudo_image tensor
    for i in range(images.shape[0]):
        output = _encode_vae_image(
            images[i].unsqueeze(0),  # Unsqueeze to add the batch dimension back
            vae=vae
        )
        outputs.append(output)  # Append the output to the list

    # Concatenate all outputs along the 0 dimension
    final_output = torch.cat(outputs, dim=0)
    final_output = final_output.unsqueeze(0)

    # if True:
    #     negative_image_latents = torch.zeros_like(final_output)

    #     # For classifier free guidance, we need to do two forward passes.
    #     # Here we concatenate the unconditional and text embeddings into a single batch
    #     # to avoid doing two forward passes
    #     final_output = torch.cat([negative_image_latents, final_output])

    return final_output


def main(output_dir, logging_dir, gradient_accumulation_steps, mixed_precision, hub_model_id):
    train_unet = True

    logging_dir = Path(output_dir, logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        # mixed_precision=mixed_precision,
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
        control_net = SpatioTemporalControlNet()
        checkpoint = torch.load('/mnt/e/13_Jasper_diffused_samples/training/output/test/model_checkpoint_399.ckpt')
        print(checkpoint['model_state_dict'].keys())     
        control_net.load_state_dict(checkpoint['model_state_dict']) 
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
    max_train_steps = 192000
    learning_rate = 1e-5
    lr_scheduler = "constant"
    lr_warmup_steps = 500
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

    train_dataset = DiffusionDataset(json_path='/home/wisley/custom_diffusers_library/src/diffusers/jasper/complete_data_paths.json')

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=1,  # Or your preferred batch size
        num_workers=0,  # Adjust based on your setup
    )

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
            if "up_blocks" in name:
                # Set the desired attribute or action here. For example, to make the parameter trainable:
                param.requires_grad = True
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
                "max_train_steps": 192000,
                "learning_rate": 1e-5,
                "lr_scheduler": "constant",
                "lr_warmup_steps": 500,
                "num_train_epochs": 50,
                "train_batch_size": 1,
                "adam_epsilon": 1e-08,
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "adam_weight_decay": 1e-2,
                "max_grad_norm": 1.0,
                # Add placeholders for any other arguments required for tracker initialization
                "tracker_project_name": "trainingTheUnetV6",
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

    guidance_scale = None
    image_logs = None
    for epoch in range(first_epoch, num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet if train_unet else controlnet):
                
                # Get the timestep
                random_idx = torch.randint(0, 25, (1,))
                timestep = timesteps[random_idx]

                # map the batch condition to decive and dtype
                batch['conditioning'] = batch['conditioning'].to(device=accelerator.device, dtype=weight_dtype)
                
                
                # Get all the inputs
                inputs = pipe_with_controlnet.prepare_input_for_forward(batch['reference_image'], batch['caption'], batch['conditioning'], num_frames=14)
                to_encode = batch["ground_truth"] if not train_unet else batch["prescan_images"]
                latents = vae.encode(to_encode.to(dtype=weight_dtype, device=accelerator.device)).latent_dist.sample() 
                
                latent_model_input = latents.to(device=accelerator.device, dtype=weight_dtype)
                latent_model_input = latent_model_input * vae.config.scaling_factor 
                latent_model_input = latent_model_input.unsqueeze(0)
                latent_model_input = torch.cat([latent_model_input] * 2) 

                # if guidance_scale is None:
                #     guidance_scale = torch.linspace(1.0, 3.0, 14).unsqueeze(0)
                #     guidance_scale = guidance_scale.to(accelerator.device, weight_dtype)
                #     guidance_scale = guidance_scale.repeat(1 * 1, 1)
                #     guidance_scale = _append_dims(guidance_scale, 5)

                            
                # Concatenate image_latents over channels dimention
                image_latents = inputs['image_latents']

                latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)


                noise_total = torch.randn_like(latent_model_input, device=accelerator.device)
                noisy_latents = scheduler.add_noise(latent_model_input, noise_total, timestep)
                noisy_latents = noisy_latents.to(device = accelerator.device, dtype = weight_dtype)

                if train_unet:
                                    
                    noise_pred = unet.forward(
                        noisy_latents,
                        timestep,
                        encoder_hidden_states= inputs["unet_encoder_hidden_states"],
                        added_time_ids= inputs['unet_added_time_ids'],
                        # down_block_additional_residuals= down_block_res_samples,
                        # mid_block_additional_residual = mid_block_res_sample,
                        return_dict=False,
                    )[0]
                    

                else:
                    down_block_res_samples, mid_block_res_sample = controlnet.forward(
                        noisy_latents.to(device = accelerator.device, dtype = weight_dtype),
                        timestep,
                        encoder_hidden_states= inputs["controlnet_encoder_hidden_states"], 
                        added_time_ids= inputs['controlnet_added_time_ids'],
                        return_dict=False,
                        controlnet_condition = inputs['controlnet_condition']
                    )
                # predict the noise residual


                
                    noise_pred = unet.forward(
                        noisy_latents,
                        timestep,
                        encoder_hidden_states= inputs["unet_encoder_hidden_states"],
                        added_time_ids= inputs['unet_added_time_ids'],
                        down_block_additional_residuals= down_block_res_samples,
                        mid_block_additional_residual = mid_block_res_sample,
                        return_dict=False,
                    )[0]
                    

                target = noise_total[:1,:,:4,:,:]

                # noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                # noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            

                loss = F.mse_loss(noise_pred[:1].float(), target.float(), reduction="mean")
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

                # Zero the parameter gradients
                optimizer.zero_grad(set_to_none=True)

                # Step the learning rate scheduler
                lr_scheduler.step()

             
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % 20 == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if True :
                            checkpoints = os.listdir(output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                        try:
                            validation_video(batch, pipe_with_controlnet, controlnet, unet, tokenizer, text_encoder, step) 
                        except Exception as e:
                            print(e)
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
                                            
                            
                     
                        torch.save(checkpoint, f'{output_dir}/test/model_checkpoint_{step}.ckpt')



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
        output_dir="/mnt/e/13_Jasper_diffused_samples/training/unet_batch",
        logging_dir="/mnt/e/13_Jasper_diffused_samples/training/logs",
        gradient_accumulation_steps=4,
        mixed_precision="fp16",
        hub_model_id="temporalControlNet",
    )