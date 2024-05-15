import math

import imageio
import numpy as np
import torch
from pytorch3d.io import load_obj, load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    HardPhongShader,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    look_at_view_transform,
    TexturesUV,
    TexturesVertex
)
from skimage import img_as_ubyte
from torch.optim.lr_scheduler import LambdaLR
from pytorch3d.structures import Meshes, packed_to_list
import open3d as o3d
from pytorch3d.transforms import RotateAxisAngle
from pytorch3d.structures import (
    join_meshes_as_batch,
    join_meshes_as_scene
)
import random
from heapq import heappush, heappop, heappushpop

from rendered_utils import clone_mesh


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, -1)


# calculate the text embs.
@torch.no_grad()
def prepare_embeddings(sds, prompt, neg_prompt="", view_dependent=False):
    # text embeddings (stable-diffusion)
    if isinstance(prompt, str):
        prompt = [prompt]
    if isinstance(neg_prompt, str):
        neg_prompt = [neg_prompt]
    embeddings = {}
    embeddings["default"] = sds.get_text_embeddings(prompt)  # shape [1, 77, 1024]
    embeddings["uncond"] = sds.get_text_embeddings(neg_prompt)  # shape [1, 77, 1024]
    if view_dependent:
        for d in ["front", "side", "back"]:
            embeddings[d] = sds.get_text_embeddings([f"{prompt[0]}, {d} view"])
    return embeddings


# calculate the text embs.
@torch.no_grad()
def prepare_clip_embeddings(clip, prompt, neg_prompt=""):
    # text embeddings (openclip)
    if isinstance(prompt, str):
        prompt = [prompt]
    if isinstance(neg_prompt, str):
        neg_prompt = [neg_prompt]
    embeddings = {}
    embeddings["default"] = clip.get_text_embeddings(prompt)  # shape [1, 77, 1024]
    embeddings["uncond"] = clip.get_text_embeddings(neg_prompt)  # shape [1, 77, 1024]

    return embeddings


def linear_to_srgb(linear: torch, eps: float = None):
    """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
    if eps is None:
        eps = torch.tensor(torch.finfo(torch.float32).eps)
    srgb0 = 323 / 25 * linear
    srgb1 = (211 * torch.maximum(eps, linear) ** (5 / 12) - 11) / 200
    return torch.where(linear <= 0.0031308, srgb0, srgb1)


def normalize_mesh_longest_axis(mesh: Meshes, unit_length=1.0, rotation_degrees: tuple = (0, 0, 0)) -> Meshes:
    """
    Normalize a PyTorch3D mesh so that its longest axis is of length 1.
    
    Parameters:
        mesh (Meshes): The input PyTorch3D Meshes object.
    
    Returns:
        Meshes: The normalized mesh.
    """
    # Ensure there is only one mesh in the Meshes object for simplicity.
    if len(mesh) != 1:
        raise ValueError(
            "This function is designed for a single mesh. Please provide a Meshes object with only one mesh.")

    # Compute the bounding box
    verts = mesh.verts_list()[0]  # Assuming there is only one mesh
    min_vals, _ = torch.min(verts, dim=0)
    max_vals, _ = torch.max(verts, dim=0)

    # Calculate lengths along each axis and find the longest axis
    lengths = max_vals - min_vals
    scale_factor = unit_length / torch.max(lengths)

    # Calculate the centroid of the bounding box
    centroid = (min_vals + max_vals) / 2

    # Scale and then shift vertices to center the mesh
    normalized_verts = (verts - centroid) * scale_factor

    # To center along all axes after normalization, we calculate the new centroid
    new_min_vals, _ = torch.min(normalized_verts, dim=0)
    new_max_vals, _ = torch.max(normalized_verts, dim=0)
    new_centroid = (new_min_vals + new_max_vals) / 2

    # Shift the normalized vertices to the middle
    centered_verts = normalized_verts - new_centroid

    degs = [torch.tensor(deg).to(torch.float32) for deg in rotation_degrees]
    # Apply rotation
    rotation_matrix = RotateAxisAngle(degs[0], 'X').get_matrix() @ \
                      RotateAxisAngle(degs[2], 'Y').get_matrix() @ \
                      RotateAxisAngle(degs[1], 'Z').get_matrix()
    rotation_matrix = rotation_matrix[0, :3, :3].to(centered_verts.device)

    rotated_verts = torch.matmul(centered_verts, rotation_matrix.T)
    # Create a new mesh with the normalized and centered vertices
    normalized_centered_rotated_mesh = mesh.update_padded(new_verts_padded=rotated_verts.unsqueeze(0))

    return normalized_centered_rotated_mesh


@torch.no_grad()
def random_mesh_initiailization(args, mesh_list: [Meshes], renderer, clip, text_embeddings, n_iters=50, n_stages=6) -> [
    Meshes]:
    best_mesh_list = mesh_list
    best_score = 0.

    for n in range(n_stages):
        # choose the best mesh positions at the previous stage
        mesh_list = best_mesh_list

        for i in range(n_iters):
            temp_mesh_list = []
            for mesh in mesh_list:
                mesh = clone_mesh(mesh, (np.random.randn(3) * 0.5 / np.sqrt(n + 1)).astype('float32'),
                                  np.random.uniform(low=0.8, high=1.2))
                temp_mesh_list.append(mesh)

            mesh = join_meshes_as_scene(temp_mesh_list)
            R, T = look_at_view_transform(dist=args.dist, elev=random.choice([0, 10, 20, 30]),
                                          azim=random.choices(np.linspace(-180, 180, 12, endpoint=False), k=3))
            sample_cameras = FoVPerspectiveCameras(R=R, T=T, device=mesh.device)
            rend = torch.permute(
                renderer(join_meshes_as_batch([mesh] * len(sample_cameras)), cameras=sample_cameras)[..., :3],
                (0, 3, 1, 2))

            score = clip.clip_score(rend, text_embeddings)
            print(score)

            if score > best_score:
                best_mesh_list = temp_mesh_list
                best_score = score

        print(f"Stage {n}: best score {best_score.item()}")

    print(f"Best score: {best_score.item()}")

    return best_mesh_list


@torch.no_grad()
def random_mesh_initiailization_queue(args, mesh_list: [Meshes], renderer, clip, text_embeddings, rand_scale=0.5,
                                      n_iters=15, n_stages=10, n_queue=5) -> [Meshes]:
    mesh_heap = [(0, mesh_list)]

    for n in range(n_stages):
        mesh_list = random.choice(mesh_heap)[1]

        for i in range(n_iters):
            temp_mesh_list = []
            for mesh in mesh_list:
                mesh = clone_mesh(mesh, (np.random.randn(3) * rand_scale / np.sqrt(n + 1)).astype('float32'),
                                  np.random.uniform(low=0.8, high=1.2))
                temp_mesh_list.append(mesh)

            mesh = join_meshes_as_scene(temp_mesh_list)
            R, T = look_at_view_transform(dist=args.dist, elev=10, azim=[0, -120, 120])
            sample_cameras = FoVPerspectiveCameras(R=R, T=T, device=mesh.device)
            rend = torch.permute(
                renderer(join_meshes_as_batch([mesh] * len(sample_cameras)), cameras=sample_cameras)[..., :3],
                (0, 3, 1, 2))

            score = clip.clip_score(rend, text_embeddings).item()
            print(score)

            if len(mesh_heap) < n_queue:
                heappush(mesh_heap, (score, temp_mesh_list))
            else:
                heappushpop(mesh_heap, (score, temp_mesh_list))

        best_score = 0
        scores = []
        for score, m_list in mesh_heap:
            best_score = max(best_score, score)
            scores.append(score)

        print(f"Stage {n}: best score {best_score} in scores {scores}")

    best_score = 0
    best_mesh_list = []
    scores = []
    for score, m_list in mesh_heap:
        if score > best_score:
            best_score = score
            best_mesh_list = m_list
            scores.append(score)

    print(f"Best score {best_score} in scores {scores}")

    return best_mesh_list


def compute_centroid(mesh: Meshes):
    if len(mesh) != 1:
        raise ValueError("Expected a single mesh, but received a batch.")
    vertices = mesh.verts_packed()
    centroid = torch.mean(vertices, dim=0)
    return centroid


def euclidean_distance(point1: torch.Tensor, point2: torch.Tensor):
    return torch.norm(point1 - point2)


def is_over(object1_centroid: torch.Tensor, object2_centroid: torch.Tensor):
    return object1_centroid[2] > object2_centroid[2]


def describe_relation(mesh1: Meshes, mesh2: Meshes):
    centroid1 = compute_centroid(mesh1)
    centroid2 = compute_centroid(mesh2)
    distance = euclidean_distance(centroid1, centroid2)
    over_status = is_over(centroid1, centroid2)
    return {
        "distance": distance.item(),
        "is_over": over_status,
        "object1_above_object2": over_status,
        "object2_above_object1": not over_status and centroid2[2] > centroid1[2]
    }
