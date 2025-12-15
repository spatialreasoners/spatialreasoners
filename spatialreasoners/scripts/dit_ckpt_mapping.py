import argparse
import os
from pathlib import Path

import torch


IGNORE_WEIGHTS = {
    "final_layer", # NOTE treated separately
    "pos_embed"
}


NO_MODEL_SET = {
    "class_embedding",
    "time_embedding"
}


KEY_MAPPING = {
    "t_embedder": "time_embedding",
    "x_embedder": "patch_emb",
    "y_embedder.embedding_table": "class_embedding.emb",
    "final_layer": "out_layer"
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ip", "--input_path", type=str, required=True)
    parser.add_argument("-op", "--output_path", type=str, required=True)
    parser.add_argument("-kol", "--keep_output_layer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("-rmvar", "--remove_var", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    output_path = Path(args.output_path)
    assert output_path.suffix == ".pt"
    ckpt = torch.load(args.input_path, weights_only=True)
    state_dict = {}

    for prefix in ("denoiser", "ema_denoiser.module"):
        for weight_key in ckpt.keys():
            if any(k in weight_key for k in IGNORE_WEIGHTS):
                continue
            updated_key: str = weight_key
            for k, v in KEY_MAPPING.items():
                if updated_key == k or updated_key.startswith(f"{k}."):
                    updated_key = updated_key.replace(k, v)
                    break
            if not any(updated_key.startswith(k) for k in NO_MODEL_SET):
                updated_key = f"model.{updated_key}"
            state_dict[f"{prefix}.{updated_key}"] = ckpt[weight_key]

        if args.keep_output_layer:
            for k, v in ckpt.items():
                if k.startswith("final_layer"):
                    if k.startswith("final_layer.linear") and args.remove_var:
                        # Throw out the variance channels
                        # NOTE there are 8 channels for mean + var and [2, 2] patch size
                        v = v.reshape(2, 2, 8, *v.shape[1:])[:, :, :4].flatten(0, 2)
                    state_dict[k.replace("final_layer", f"{prefix}.model.out_layer")] = v

    # add n_averaged > 0 to avoid overwriting EMA weights with directly optimized weights
    state_dict["ema_denoiser.n_averaged"] = torch.tensor(1, dtype=torch.long)

    output_path = Path(args.output_path)
    os.makedirs(output_path.parent, exist_ok=True)
    torch.save(state_dict, args.output_path)
