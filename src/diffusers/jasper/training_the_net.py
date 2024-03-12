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


# We must login into wandb and huggingface
# huggingface-cli login
# huggingface-cli api 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

# wandb login 
# wandb api e22ca6359f013b2748d82d61417b843a3b9e74be

# accelerate config
# accelerate config default

if is_wandb_available():
    import wandb

logger = get_logger(__name__)


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
            image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
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
            transforms.Resize((576, 1024)),
            transforms.CenterCrop((576, 1024)),
        ])

    def __len__(self):
        # Assuming each set of ground truths represents a separate sample
        return len(self.data['ground_truth'])

    def __getitem__(self, idx):
        # Processing ground truth images
        to_tensor = transforms.ToTensor()
        ground_truth_images = [to_tensor(self.transform(Image.open(path))) for path in self.data['ground_truth'][idx]]

        # Processing conditioning images set one (assuming RGB, 4 channels after conversion)
        conditioning_images_one = [to_tensor(self.transform(Image.open(path))) for path in self.data['conditioning_images_one'][idx]]

        # Processing conditioning images set two (assuming grayscale, converted to RGB to match dimensions)
        conditioning_images_two = [to_tensor(self.transform(Image.open(path))) for path in self.data['conditioning_images_two'][idx]]
        
        # Concatenating condition one and two images along the channel dimension
        conditioned_images = [torch.cat((img_one, img_two), dim=0) for img_one, img_two in zip(conditioning_images_one, conditioning_images_two)]

        # Processing reference images (single per scene, matched by index)
        reference_image = self.transform(Image.open(self.data['reference_image'][idx][0]))

        # Retrieving the corresponding caption
        caption = self.data['caption'][idx][0]

        

        return {
            "ground_truth": torch.stack(ground_truth_images),
            "conditioning": torch.stack(conditioned_images),
            "caption": caption,
            "reference_image": reference_image
        }

def collate_fn(batch):
    ground_truth = torch.stack([item['ground_truth'] for item in batch])
    conditioning = torch.stack([item['conditioning'] for item in batch])
    captions = [item['caption'] for item in batch]  # List of strings, no need to stack
    reference_images = [item['reference_image'] for item in batch]
    

    return {
        "ground_truth": ground_truth.flatten(0, 1),
        "conditioning": conditioning.flatten(0, 1),
        "caption": captions[0],
        "reference_image": reference_images[0],
    }

def _encode_vae_image(

        image: torch.Tensor,
        vae
    ):

        with torch.no_grad(): 

            # print(f"this is the shape of the image: {image.shape}")
            image = image.to(device=vae.device, dtype=vae.dtype)
            image_latents = vae.encode(image.to(device=vae.device)).latent_dist.mode()



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

    if True:
        negative_image_latents = torch.zeros_like(final_output)

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        final_output = torch.cat([negative_image_latents, final_output])

    return final_output


def main(output_dir, logging_dir, gradient_accumulation_steps, mixed_precision, hub_model_id):
    logging_dir = Path(output_dir, logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
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
        "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
    )
    pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2")


    # Getting the models
    unet_weights = pipe.unet.state_dict()
    my_net = UNetSpatioTemporalConditionModel()
    my_net.load_state_dict(unet_weights)


    control_net = SpatioTemporalControlNet.from_unet(my_net) 
    
    # Make them f16
    my_net = my_net.half()
    control_net = control_net.half()   

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
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.train()

    # Allow for memory save attention
    # if True:
    #     if is_xformers_available():
    #         import xformers

    #         xformers_version = version.parse(xformers.__version__)
    #         if xformers_version == version.parse("0.0.16"):
    #             logger.warn(
    #                 "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
    #             )
    #         unet.enable_xformers_memory_efficient_attention()
    #         controlnet.enable_xformers_memory_efficient_attention()
    #     else:
    #         raise ValueError("xformers is not available. Make sure it is installed correctly")

    if True:
        controlnet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    # if unwrap_model(controlnet).dtype != torch.float32:
    #     raise ValueError(
    #         f"Controlnet loaded as datatype {unwrap_model(controlnet).dtype}. {low_precision_error_string}"
    #     )

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
    params_to_optimize = controlnet.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    train_dataset = DiffusionDataset(json_path='/mnt/e/13_Jasper_diffused_samples/complete_data_paths.json')

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

    # Prepare everything with our `accelerator`.
    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
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
                "tracker_project_name": "temporalControlNet",
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
    # if args.resume_from_checkpoint:
    #     if args.resume_from_checkpoint != "latest":
    #         path = os.path.basename(args.resume_from_checkpoint)
    #     else:
    #         # Get the most recent checkpoint
    #         dirs = os.listdir(args.output_dir)
    #         dirs = [d for d in dirs if d.startswith("checkpoint")]
    #         dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
    #         path = dirs[-1] if len(dirs) > 0 else None

    #     if path is None:
    #         accelerator.print(
    #             f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
    #         )
    #         args.resume_from_checkpoint = None
    #         initial_global_step = 0
    #     else:
    #         accelerator.print(f"Resuming from checkpoint {path}")
    #         accelerator.load_state(os.path.join(args.output_dir, path))
    #         global_step = int(path.split("-")[1])

    #         initial_global_step = global_step
    #         first_epoch = global_step // num_update_steps_per_epoch
    # else:
    initial_global_step = 0

    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )


# {
#             "latents": latents,
#             "unet_encoder_hidden_states": image_embeddings,
#             "unet_added_time_ids": added_time_ids,
#             "controlnet_encoder_hidden_states": prompt_embeds if prompt is not None else image_embeddings,
#             "controlnet_added_time_ids": added_time_ids,
#             "controlnet_condition": conditioning_image,
#             "image_latents": image_latents,
            
#         }

    scheduler.set_timesteps(25, device=accelerator.device)
    timesteps = scheduler.timesteps



    image_logs = None
    for epoch in range(first_epoch, num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):
                
                # Get the timestep
                random_idx = torch.randint(0, 25, (1,))
                timestep = timesteps[random_idx]
                
                # Get all the inputs
                inputs = pipe_with_controlnet.prepare_input_for_forward(batch['reference_image'], batch['caption'], batch['conditioning'])
                
                # Convert images to latent space
                latents = encode_batch(batch["ground_truth"], vae)

                # make sure the latents are on the correct device and dtype
                latents = latents.to(device=accelerator.device, dtype=weight_dtype)
                
                # latent_model_input = torch.cat([latents] * 2) 
                latent_model_input = scheduler.scale_model_input(latents, timestep.item())

                noise_for_video = torch.randn_like(latent_model_input, device=accelerator.device)

                # enable grad for the noise
                noise_for_video.requires_grad = True
                noise_for_image = torch.zeros_like(inputs['image_latents'], device=accelerator.device)
                noise_total = torch.cat([noise_for_video, noise_for_image], dim=2)
                
                # Concatenate image_latents over channels dimention
                latent_model_input = torch.cat([latent_model_input, inputs['image_latents']], dim=2)

                # Sample noise that we'll add to the latents
                
                # Sample a random timestep for each image
                

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = scheduler.add_noise(latent_model_input, noise_total, timestep)
                noisy_latents = noisy_latents.to(device = accelerator.device, dtype = weight_dtype)


                sample_control = noisy_latents.reshape(50,8,72,128)
                sample_downsampled = torch.nn.functional.interpolate(sample_control, scale_factor=0.5, mode='nearest')

                # movce back
                sample_downsampled = sample_downsampled.reshape(2,25,8,36,64)

                down_block_res_samples, mid_block_res_sample = controlnet.forward(
                    sample_downsampled.to(device = accelerator.device, dtype = weight_dtype),
                    timestep,
                    encoder_hidden_states= inputs["controlnet_encoder_hidden_states"], 
                    added_time_ids= inputs['controlnet_added_time_ids'],
                    return_dict=False,
                    controlnet_condition = inputs['controlnet_condition']
                )

                # predict the noise residual


                with torch.no_grad():
                    model_pred = unet(
                        noisy_latents,
                        timestep,
                        encoder_hidden_states= inputs["unet_encoder_hidden_states"],
                        added_time_ids= inputs['unet_added_time_ids'],
                        down_block_additional_residuals= down_block_res_samples,
                        mid_block_additional_residual = mid_block_res_sample,
                        return_dict=False,
                    )[0]

                target = noise_for_video

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")


                # accelerator.backward(loss)
                # if accelerator.sync_gradients:
                #     params_to_clip = controlnet.parameters()
                #     accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
                
                # optimizer.step()
        
                # lr_scheduler.step()

                scaled_loss = accelerator.scaler.scale(loss)
                accelerator.backward(scaled_loss)
                if accelerator.sync_gradients:
                    params_to_clip = controlnet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
                accelerator.scaler.step(optimizer)
                accelerator.scaler.update()
                lr_scheduler.step()

                # ISSUE
                # optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % 10 == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if True :
                            checkpoints = os.listdir(output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")



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
        output_dir="/mnt/e/13_Jasper_diffused_samples/training/output",
        logging_dir="/mnt/e/13_Jasper_diffused_samples/training/logs",
        gradient_accumulation_steps=4,
        mixed_precision="fp16",
        hub_model_id="temporalControlNet",
    )