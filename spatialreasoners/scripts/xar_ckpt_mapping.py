import argparse
import os
from pathlib import Path

import torch


IGNORE_WEIGHTS = {
    "mask",
    "rope.freqs"
}


KEY_MAPPING = {
    "ada_lin": "adaLN_modulation",
    "time_embed": "t_emb",
    "class_emb": "class_embedding.emb",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()

    output_path = Path(args.output_path)
    assert output_path.suffix == ".pt"
    ckpt = torch.load(args.input_path, weights_only=True)
    state_dict = {}

    for weight_key in ckpt.keys():
        if any(k in weight_key for k in IGNORE_WEIGHTS):
            continue
        updated_key: str = weight_key
        for k, v in KEY_MAPPING.items():
            if k in updated_key:
                updated_key = updated_key.replace(k, v)
                break
        state_dict[f"denoising_model.denoiser.{updated_key}"] = ckpt[weight_key]
    
    output_path = Path(args.output_path)
    os.makedirs(output_path.parent, exist_ok=True)
    torch.save(state_dict, args.output_path)
