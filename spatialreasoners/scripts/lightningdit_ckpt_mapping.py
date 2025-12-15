import argparse
import os
from pathlib import Path

import torch


IGNORE_WEIGHTS = {
    "pos_embed",
    "feat_rope"
}


KEY_MAPPING = {
    "t_embedder": "t_emb",
    "y_embedder.embedding_table": "class_embedding.emb"
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("-rmp", "--remove_prefix", type=str, default="module.")
    parser.add_argument("-modk", "--model_keys", type=str, default="model")
    parser.add_argument("-emak", "--ema_keys", type=str, default="ema")
    args = parser.parse_args()

    output_path = Path(args.output_path)
    assert output_path.suffix == ".pt"
    ckpt = torch.load(args.input_path, weights_only=True)

    model_state_dict = ckpt
    if args.model_keys is not None:
        model_keys = list(map(str.strip, args.model_keys.split(",")))
        for k in model_keys:
            model_state_dict = model_state_dict[k]
    
    if args.ema_keys is not None:
        ema_state_dict = ckpt
        ema_keys = list(map(str.strip, args.ema_keys.split(",")))
        for k in ema_keys:
            ema_state_dict = ema_state_dict[k]
    else:
        ema_state_dict = model_state_dict

    out_state_dict = {}

    weight_key: str
    for prefix, state_dict in (
        ("denoising_model.denoiser", model_state_dict), 
        ("denoising_model.ema_denoiser.module", ema_state_dict)
    ):
        weight: torch.Tensor
        for weight_key, weight in state_dict.items():
            if args.remove_prefix is not None and weight_key.startswith(args.remove_prefix):
                weight_key = weight_key[len(args.remove_prefix):]

            if weight_key.startswith("final_layer"):
                if weight_key.startswith("final_layer.linear"):
                    """
                    LightningDiT uses rectified flows with swapped weight functions of data and noise
                    such that their models are effectively trained to predict the negative of the flow in our framework
                    """
                    weight = -weight
                out_state_dict[weight_key.replace("final_layer", f"{prefix}.out_layer")] = weight
            elif weight_key.startswith("x_embedder"):
                if weight_key.startswith("x_embedder.proj.weight"):
                    """pySpaRe processes already patched tokens --> linear layer instead of 2D convolution"""
                    weight = weight.flatten(1)
                out_state_dict[weight_key.replace("x_embedder", f"{prefix}.patch_emb")] = weight
            elif not any(k in weight_key for k in IGNORE_WEIGHTS):
                for k, v in KEY_MAPPING.items():
                    if weight_key == k or weight_key.startswith(f"{k}."):
                        weight_key = weight_key.replace(k, v)
                        break
                out_state_dict[f"{prefix}.{weight_key}"] = weight

    # add n_averaged > 0 to avoid overwriting EMA weights with directly optimized weights
    out_state_dict["denoising_model.ema_denoiser.n_averaged"] = torch.tensor(1, dtype=torch.long)

    output_path = Path(args.output_path)
    os.makedirs(output_path.parent, exist_ok=True)
    torch.save(out_state_dict, args.output_path)
