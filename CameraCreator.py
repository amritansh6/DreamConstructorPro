import numpy as np
import torch
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras


class CameraCreator:
    def __init__(self, device, dist):
        self.device = device
        self.dist = dist

    def generate_cameras(self, elevations, azimuths, elev_step, azim_step, endpoint=False):
        Rs, Ts = [], []
        for elev in np.linspace(elevations[0], elevations[1], elev_step, endpoint=endpoint):
            R, T = look_at_view_transform(dist=self.dist, elev=elev, azim=np.linspace(azimuths[0], azimuths[1], azim_step, endpoint=endpoint))
            Rs.append(R)
            Ts.append(T)
        return FoVPerspectiveCameras(R=torch.cat(Rs), T=torch.cat(Ts), device=self.device)

    def create_cameras(self):
        query_cameras = self.generate_cameras((0, 15), (-180, 180), 5, 18)
        testing_cameras = self.generate_cameras((0, 0), (-180, 180), 1, 12)
        return query_cameras, testing_cameras