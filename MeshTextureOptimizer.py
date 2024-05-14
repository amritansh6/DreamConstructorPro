import argparse
import os
import os.path as osp
import time
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import numpy as np
import random
from pytorch3d.renderer import FoVPerspectiveCameras, look_at_view_transform, PointLights

from CameraCreator import CameraCreator
from rendered_utils import init_mesh, clone_mesh, save_mesh_as_ply, render_360_views, get_mesh_renderer_soft
from utils import (prepare_embeddings, prepare_clip_embeddings, normalize_mesh_longest_axis, random_mesh_initiailization_queue,
                   seed_everything, get_cosine_schedule_with_warmup)
from Score_Distillation_Sampling import SDS
from pytorch3d.structures import (
    join_meshes_as_batch,
    join_meshes_as_scene
)
from SDS import CLIP
from differentiable_object import DifferentiableObject


class MeshTextureOptimizer:
    def __init__(self, sds, clip, mesh_paths, output_dir, prompt, neg_prompt, device, total_iter, args,
                 log_interval=100, save_mesh=True):
        self.sds = sds
        self.clip = clip
        self.mesh_paths = mesh_paths
        self.output_dir = output_dir
        self.prompt = prompt
        self.neg_prompt = neg_prompt
        self.device = device
        self.total_iter = total_iter
        self.args = args
        self.renderer = get_mesh_renderer_soft(image_size=128, device=device)
        self.renderer.shader.lights = PointLights(location=[[0, 0, -3]], device=device)
        self.mesh_list = self.load_meshes()
        self.log_interval = log_interval
        self.sds_embeddings = prepare_embeddings(sds, prompt, neg_prompt) if args.use_sds else None
        self.clip_embeddings = prepare_clip_embeddings(clip, prompt,neg_prompt) if args.use_clip else None
        self.query_cameras, self.testing_cameras = self.create_cameras()
        self.loss_dict = {}
        self.save_mesh = save_mesh

    def load_meshes(self):
        mesh_list = [self.initialize_mesh(path, orientation)
                     for path, orientation in zip(self.mesh_paths, self.args.mesh_init_orientations)]
        mesh_list = [self.transform_mesh(mesh, config)
                     for mesh, config in zip(mesh_list, self.args.mesh_configs)]
        if self.args.use_rand_init:
            mesh_list = self.apply_random_initialization(mesh_list)

        return mesh_list

    def initialize_mesh(self, mesh_path, mesh_init_orientation):
        mesh, _, _, _ = init_mesh(mesh_path, device=self.device)
        mesh = normalize_mesh_longest_axis(mesh, rotation_degrees=mesh_init_orientation)
        return mesh.to(self.device)

    def transform_mesh(self, mesh, config):
        return clone_mesh(mesh, shift=config["transition"], scale=config["scale"])

    def apply_random_initialization(self, mesh_list):
        return random_mesh_initiailization_queue(self.args, mesh_list, self.renderer,
                                                 self.clip, self.clip_embeddings["default"], rand_scale=0.3)

    def create_cameras(self):
        camera_Creator=CameraCreator(device=self.device,dist=self.args.dist)
        query_cameras, testing_cameras = camera_Creator.create_cameras()
        return query_cameras, testing_cameras

    def optimize(self):
        diff_objects = DifferentiableObject(join_meshes_as_batch(self.mesh_list), self.device)
        optimizer = torch.optim.AdamW([
            {'params': [diff_objects.scale], 'lr': 1e-3},
            {'params': [diff_objects.rotation], 'lr': 1e-2},
            {'params': [diff_objects.transition], 'lr': 1e-2}
        ], lr=1e-4, weight_decay=0)
        scheduler = get_cosine_schedule_with_warmup(optimizer, 100, int(self.total_iter * 1.5))

        for i in tqdm(range(self.total_iter)):
            optimizer.zero_grad()
            mesh = join_meshes_as_scene(diff_objects())
            sampled_cameras = self.query_cameras[
                random.choices(range(len(self.query_cameras)), k=self.args.views_per_iter)]
            rend = torch.permute(self.renderer(mesh, cameras=sampled_cameras)[..., :3], (0, 3, 1, 2))
            loss = self.compute_loss(rend, i)
            print(f"Iter {i}, Loss: {loss.item()}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diff_objects.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if i % self.log_interval == 0 or i == self.total_iter - 1:
                self.log_outputs(mesh, i, loss, sampled_cameras[0])

    def compute_loss(self, rend, iter):
        loss = 0.0
        if self.args.use_sds:
            if rend.shape[-2:] != (512, 512):
                rend = F.interpolate(rend, size=(512, 512), mode='bilinear', align_corners=False)
            latents = self.sds.encode_imgs(rend)
            loss += self.sds.batch_sds_loss(latents, self.sds_embeddings["default"], self.sds_embeddings["uncond"])
        if self.args.use_clip:
            loss += self.clip.clip_loss(rend, self.clip_embeddings["default"], self.clip_embeddings["uncond"])
        return loss

    def log_outputs(self, mesh, iter, loss, camera):
        with torch.no_grad():
            mesh = join_meshes_as_scene(mesh)
            self.loss_dict[iter] = loss.item()
            img = self.renderer(mesh, cameras=camera)[0, ..., :3]
            img = (img.clamp(0, 1) * 255).round().cpu().numpy()
            output_im = Image.fromarray(img.astype("uint8"))
            output_path = os.path.join(self.output_dir, f"output_{self.prompt[0].replace(' ', '_')}_iter_{iter}.png")
            output_im.save(output_path)
            mesh_path = os.path.join(self.output_dir, f"mesh_iter_{iter}.ply")
            save_mesh_as_ply(mesh, mesh_path)

            log_path = os.path.join(self.output_dir, "logs.txt")
            self.write_log(log_path, f"Iter {iter}, Loss: {loss.item()}")
            self.perform_validation(mesh, iter, log_path)

            if self.save_mesh:
                render_360_views(mesh.detach(), self.renderer, dist=self.args.dist,
                                 device=self.device, output_path=os.path.join(self.output_dir, f"final_mesh.gif"))
                save_mesh_as_ply(mesh.detach(), os.path.join(self.output_dir, f"final_mesh.ply"))

    def write_log(self, log_path, message):
        with open(log_path, 'a') as log_file:
            log_file.write(message + '\n')

    def perform_validation(self, mesh, iter, log_path):
        loss = 0.0
        for camera in self.testing_cameras:
            rend = torch.permute(self.renderer(mesh, cameras=camera)[..., :3], (0, 3, 1, 2))
            if self.args.use_sds:
                if rend.shape[-2:] != (512, 512):
                    rend = F.interpolate(rend, size=(512, 512), mode='bilinear', align_corners=False)
                latents = self.sds.encode_imgs(rend)
                loss -= self.sds.batch_sds_loss(latents, self.sds_embeddings["default"], self.sds_embeddings["uncond"])
            if self.args.use_clip:
                loss -= self.clip.clip_loss(rend, self.clip_embeddings["default"])
        self.write_log(log_path, f"Iter {iter}, Validation Score: {loss.item() / len(self.testing_cameras)}")
