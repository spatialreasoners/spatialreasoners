import argparse
import math
import os
from pathlib import Path

import torch
from torch.nn import init


DROP_WEIGHTS = {
    "diffloss.net.final_layer", # NOTE treated separately
    "fake_latent",      # NOTE "no class" embedding for CFG, treated separately
    "class_emb.weight"  # NOTE treated separately
}


NO_MODEL_SET = {
    "class_embedding",
    "time_embedding"
}


KEY_MAPPING = {
    "diffloss.net.time_embed": "time_embedding",
    "encoder_pos_embed_learned": "encoder.pos_emb",
    "mask_token": "decoder.mask_token",
    "decoder_pos_embed_learned": "decoder.pos_emb",
    "diffusion_pos_embed_learned": "mlp_pos_emb",
    "z_proj": "encoder.z_proj",
    "z_proj_ln": "encoder.z_proj_norm",
    "encoder_blocks": "encoder.blocks",
    "encoder_norm": "encoder.out_norm",
    "decoder_embed": "decoder.in_proj",
    "decoder_blocks": "decoder.blocks",
    "decoder_norm": "decoder.out_norm",
    "diffloss.net.cond_embed": "denoiser_mlp.c_proj",
    "diffloss.net.input_proj": "denoiser_mlp.in_layer",
    "diffloss.net.res_blocks": "denoiser_mlp.blocks"
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ip", "--input_path", type=str, required=True)
    parser.add_argument("-op", "--output_path", type=str, required=True)
    parser.add_argument("-kol", "--keep_output_layer", action=argparse.BooleanOptionalAction, default=True)
    # NOTE this is the result of 32 channels * [1, 1] patch size
    # NOTE ignored of keep_output_layer == False
    parser.add_argument("-kov", "--keep_output_variance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("-naoc", "--num_additional_output_channels", type=int, default=0)
    args = parser.parse_args()

    output_path = Path(args.output_path)
    assert output_path.suffix == ".pt"
    ckpt = torch.load(args.input_path, weights_only=True)
    state_dict = {}

    for model_key, prefix in zip(("model", "model_ema"), ("denoiser", "ema_denoiser.module")):
        model_weights: dict[str, torch.Tensor] = ckpt[model_key]
        for weight_key in model_weights.keys():
            if any(k in weight_key for k in DROP_WEIGHTS):
                continue
            updated_key: str = weight_key
            for k, v in KEY_MAPPING.items():
                if updated_key == k or updated_key.startswith(f"{k}."):
                    updated_key = updated_key.replace(k, v)
                    break
            if not any(updated_key.startswith(k) for k in NO_MODEL_SET):
                updated_key = f"model.{updated_key}"
            state_dict[f"{prefix}.{updated_key}"] = model_weights[weight_key]

        if args.keep_output_layer:
            for k, v in model_weights.items():
                if k.startswith("diffloss.net.final_layer"):
                    if k.startswith("diffloss.net.final_layer.linear"):
                        # NOTE there are 16 channels for mean + var and [1, 1] patch size
                        v = v.reshape(32, 1, 1, *v.shape[1:])
                        if not args.keep_output_variance:
                            v = v[:16]
                        v = v.flatten(0, 2)
                        if args.num_additional_output_channels > 0:
                            additional = torch.empty((args.num_additional_output_channels, *v.shape[1:]), dtype=v.dtype)
                            if k.endswith("weight"):
                                init.kaiming_uniform_(additional, a=math.sqrt(5))
                            elif k.endswith("bias"):
                                fan_in = model_weights["diffloss.net.final_layer.linear.weight"].shape[1]
                                bound = 1 / math.sqrt(fan_in)
                                init.uniform_(additional, -bound, bound)
                            else:
                                raise ValueError(f"Unexpected key {k}")
                            v = torch.cat((v, additional), dim=0)
                    state_dict[k.replace("diffloss.net.final_layer", f"{prefix}.model.denoiser_mlp.out_layer")] = v

        # Concatenate class embeddings and "no class" embedding (fake_latent)
        class_emb = torch.cat((model_weights["class_emb.weight"], model_weights["fake_latent"]), dim=0)
        state_dict[f"{prefix}.class_embedding.emb.weight"] = class_emb

    # add n_averaged > 0 to avoid overwriting EMA weights with directly optimized weights
    state_dict["ema_denoiser.n_averaged"] = torch.tensor(1, dtype=torch.long)

    output_path = Path(args.output_path)
    os.makedirs(output_path.parent, exist_ok=True)
    torch.save(state_dict, args.output_path)
