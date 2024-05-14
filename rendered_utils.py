import numpy as np
import torch
from pytorch3d.io import load_obj, load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, HardPhongShader, MeshRasterizer, MeshRenderer,
    PointLights, RasterizationSettings, look_at_view_transform, TexturesUV, TexturesVertex
)
from pytorch3d.structures import Meshes, packed_to_list
from skimage import img_as_ubyte
import imageio
import open3d as o3d


def get_mesh_renderer(image_size=512, lights=None, device=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
    raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1)
    return MeshRenderer(rasterizer=MeshRasterizer(raster_settings=raster_settings),
                        shader=HardPhongShader(device=device, lights=lights))


def get_mesh_renderer_soft(image_size=512, lights=None, device=None, sigma=1e-4):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
    raster_settings_soft = RasterizationSettings(
        image_size=image_size, blur_radius=np.log(1.0 / 1e-4 - 1.0) * sigma,
        faces_per_pixel=50, perspective_correct=False, bin_size=0)
    return MeshRenderer(rasterizer=MeshRasterizer(raster_settings=raster_settings_soft),
                        shader=HardPhongShader(device=device, lights=lights))


def render_360_views(mesh, renderer, device, dist=3, elev=0, output_path=None):
    images = []
    for azim in range(0, 360, 10):
        R, T = look_at_view_transform(dist, elev, azim)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        lights = PointLights(location=[[0, 0, -3]], device=device)
        rend = renderer(mesh, cameras=cameras, lights=lights).cpu().numpy()[0, ..., :3]
        images.append(img_as_ubyte(np.clip(rend, -1, 1)))
    imageio.mimsave(output_path, images, duration=0.1)


def init_mesh( model_path,device="cpu"):
    print("=> loading target mesh...")
    verts, faces, aux = load_obj(
        model_path, device=device, load_textures=True, create_texture_atlas=True
    )
    mesh = load_objs_as_meshes([model_path], device=device)
    faces = faces.verts_idx
    return mesh, verts, faces, aux


def clone_mesh(mesh,shift = [0., 0., 0.],scale = 1.0):
    shift = torch.tensor(shift, device=mesh.device)
    return Meshes(verts=[mesh.verts_packed()*scale + shift], faces=[mesh.faces_packed()], textures=mesh.textures).to(mesh.device)


def save_mesh_as_ply(mesh, path):
    textures = convert_to_textureVertex(mesh.textures, mesh)
    o3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(mesh.verts_packed().cpu().numpy()),
        triangles=o3d.utility.Vector3iVector(mesh.faces_packed().cpu().numpy()))
    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(textures.verts_features_packed().cpu().numpy())
    o3d.io.write_triangle_mesh(path, o3d_mesh)


def convert_to_textureVertex(textures_uv: TexturesUV, meshes: Meshes):
    verts_colors_packed = torch.zeros_like(meshes.verts_packed())
    verts_colors_packed[meshes.faces_packed()] = textures_uv.faces_verts_textures_packed()
    return TexturesVertex(packed_to_list(verts_colors_packed, meshes.num_verts_per_mesh()))
