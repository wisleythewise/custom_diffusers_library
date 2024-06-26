import argparse

import torch

from diffusers import MotionAdapter


def convert_motion_module(original_state_dict):
    converted_state_dict = {}
    for k, v in original_state_dict.items():
        if "pos_encoder" in k:
            continue

        else:
            converted_state_dict[
                k.replace(".norms.0", ".norm1")
                .replace(".norms.1", ".norm2")
                .replace(".ff_norm", ".norm3")
                .replace(".attention_blocks.0", ".attn1")
                .replace(".attention_blocks.1", ".attn2")
                .replace(".temporal_transformer", "")
            ] = v

    return converted_state_dict


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--use_motion_mid_block", action="store_true")
    parser.add_argument("--motion_max_seq_length", type=int, default=32)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    state_dict = torch.load(args.ckpt_path, map_location="cpu")
    if "state_dict" in state_dict.keys():
        state_dict = state_dict["state_dict"]

    conv_state_dict = convert_motion_module(state_dict)
    adapter = MotionAdapter(
        use_motion_mid_block=args.use_motion_mid_block, motion_max_seq_length=args.motion_max_seq_length
    )
    # skip loading position embeddings
    adapter.load_state_dict(conv_state_dict, strict=False)
    adapter.save_pretrained(args.output_path)
    adapter.save_pretrained(args.output_path, variant="fp16", torch_dtype=torch.float16)
