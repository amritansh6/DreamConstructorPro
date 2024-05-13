import argparse
import os
import time
import os.path as osp

import torch

from MeshTextureOptimizer import MeshTextureOptimizer
from SDS import CLIP
from Score_Distillation_Sampling import SDS
from utils import seed_everything

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="A chair and a table with a toy dinosaur on it")
    parser.add_argument("--seed", type=int, default=42)  # 42
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument(
        "--postfix",
        type=str,
        default="_test",
        help="postfix for the output directory to differentiate multiple runs")

    parser.add_argument(
        "-m",
        "--mesh_paths",
        type=list,
        # default=["data/10213_Armed_ Dinner_Chair_v2_iterations-2.obj", "data/10241_Outdoor_table_w_hole_for_umbrella_v1_L3.obj", "data/11678_dinosaur_v1_L3.obj"],
        default=["data/dg/chair_sai.obj", "data/dg/table_sai.obj", "data/dg/toy_dinosaur_sai.obj"],
        help="Path to the input image",
    )
    parser.add_argument(
        "--dist",
        type=int,
        default=4
    )
    parser.add_argument(
        "--views_per_iter",
        type=int,
        default=1  # viewpoints sampled from calculated the loss in a single iteration
    )
    parser.add_argument(
        "--use_sds",
        type=int,
        default=0  # use SDS loss when != 0
    )
    parser.add_argument(
        "--use_clip",
        type=int,
        default=1  # use CLIP loss when != 0
    )
    parser.add_argument(
        "--use_rand_init",
        type=int,
        default=0  # use sampling-base initialization for the initial positions of meshes
    )
    args = parser.parse_args()

    #### rotation for meshes to face forward (edit it for the meshes downloaded from Internet) ####

    # args.mesh_init_orientations = [(90, 180, 0), (90, 0, 0), (90, -90, 0)]
    args.mesh_init_orientations = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]  # meshes from DreamGaussian need not to be rotated

    ##########################

    #### configs from LLM ####

    # manually adjustthe initial positions of meshes (could use the suggestion from LLM instead)

    args.mesh_configs = [{
        "transition": (-1, 0, 0),
        "rotation": (0, 0, 0),
        "scale": 1  # chair
    },
        {
            "transition": (0, 0, 0),
            "rotation": (0, 0, 0),
            "scale": 1  # table
        },
        {
            "transition": (0, 1, 0),
            "rotation": (0, 0, 0),
            "scale": 0.5  # dinosaur
        }]

    seed_everything(args.seed)

    # create output directory
    args.output_dir = osp.join(args.output_dir, "mesh")
    output_dir = os.path.join(
        args.output_dir, args.prompt.replace(" ", "_") + args.postfix + ("_sds" if args.use_sds else "") + (
            "_clip" if args.use_clip else "") + f"_{args.views_per_iter}view"
    )
    os.makedirs(output_dir, exist_ok=True)

    # initialize SDS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sds = SDS(sd_version="2.1", device=device, output_dir=output_dir) if args.use_sds else None
    clip = CLIP(device=device, output_dir=output_dir) if args.use_clip else None

    # optimize the texture map of a mesh
    start_time = time.time()
    assert (
            args.mesh_paths is not None
    ), "mesh_path should be provided for optimizing the texture map for a mesh"

    neg_prompt = [""]  # ["", "distortion", "blur"]

    optimizer=MeshTextureOptimizer(sds,clip,mesh_paths=args.mesh_paths, output_dir=output_dir, prompt=args.prompt, neg_prompt=neg_prompt,
        device=device,total_iter=10, args=args)
    optimizer.optimize()
    print(f"Optimization took {time.time() - start_time:.2f} seconds")