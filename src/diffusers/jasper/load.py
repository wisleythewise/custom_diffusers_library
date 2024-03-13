import torch
import shutil
import os
# checkpoint = torch.load('/mnt/e/13_Jasper_diffused_samples/training/output/test/model_checkpoint_39.ckpt')


# from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion_with_controlnet import StableVideoDiffusionPipelineWithControlNet,SpatioTemporalControlNet, CustomConditioningNet, SpatioTemporalControlNetOutput
# model = SpatioTemporalControlNet()
# # print(checkpoint['model_state_dict'].keys())     
# model.load_state_dict(checkpoint['model_state_dict'])


output_dir="/mnt/e/13_Jasper_diffused_samples/training/output"
checkpoints = os.listdir(output_dir)
checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
print(checkpoints )
shutil.rmtree(f'{output_dir}/{checkpoints[0]}')