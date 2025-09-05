import torch
from torch import nn

from render.render import render_ray_bundle
from render.ray import RayBundle, RayBundleFan, RayBundleLinear, RayBundleTUESREC
from nerf.model import NerualRadianceField

class EmulatedUSDevice(nn.Module):

    def __init__(self, trained_model_path=None, subsample_slice=None) -> None:
        super().__init__()
        self.nerf = NerualRadianceField()
        self.sample_parameters = None
        if trained_model_path is not None:
            self.nerf.load_state_dict(torch.load(trained_model_path, weights_only=True))
        self.subsample_slice = subsample_slice

    def pose_to_ray_bundle(self, pose) -> RayBundle:
        raise NotImplementedError

    def forward(self, pose):
        if self.sample_parameters is None:
            raise ValueError('This instance was not initialized properly since self.sample_parameters is None.')
        rb = self.pose_to_ray_bundle(pose)
        rb.sample(*self.sample_parameters)
        if self.subsample_slice is not None:
            rb.subsample(self.subsample_slice)
        return render_ray_bundle(rb, self.nerf)


class ClariusC3HD3(EmulatedUSDevice):

    def __init__(self, trained_model_path=None, subsample_slice=(slice(None, None, None), slice(None, None, None))) -> None:
        super().__init__(trained_model_path, subsample_slice)
        near, depth = 0.045, 0.070
        h, w = 912, 192
        self.sample_parameters = (near, near + depth, h, w)
        self.central_angle = 73 * torch.pi / 180

    def pose_to_ray_bundle(self, pose) -> RayBundleFan:
        origin = pose[:3, -1] # (3, 1)
        rot_mat = pose[:3, :3] # (3, 3) 
        d = torch.linalg.det(rot_mat)
        if not torch.allclose(d, torch.ones_like(d), rtol=0.001): # check if the rotation matrix is orthogonal
            raise ValueError(f'Invalid pose, determinant of the rotation matrix is {d}.')
        # change direction and plane_normal if coordinate system doesnot match
        direction = rot_mat @ torch.tensor([0, 0, 1], dtype=torch.float).reshape(-1, 1) # (3, 1) 
        plane_normal = rot_mat @ torch.tensor([0, -1, 0], dtype=torch.float).reshape(-1, 1) # (3, 1) 
        
        return RayBundleFan(origin, direction, plane_normal, self.central_angle)


class SyentheticLiverProbe(EmulatedUSDevice):

    def __init__(self, trained_model_path=None, subsample_slice=None) -> None:
        super().__init__(trained_model_path, subsample_slice)
        near, far, width = 0, 0.14, 0.08
        h, w = 512, 256 # image height, width 
        self.sample_parameters = (near, far, width, h, w)
        self.offset = torch.tensor([ 
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
        ], dtype=torch.float) # 基底转换矩阵 用于将探头坐标系的x/y/z轴方向与世界坐标系的x/y/z轴方向调一致

    def pose_to_ray_bundle(self, pose):
        origin = pose[:3, -1] + self.offset[:3, -1] # ray origin (3,)
        rot_mat = pose[:3, :3] @ self.offset[:3, :3] # rotation matrix (3, 3)
        d = torch.linalg.det(rot_mat)
        if not (torch.allclose(d, torch.ones_like(d), rtol=0.001) or torch.allclose(d, -1 * torch.ones_like(d), rtol=0.001)):
            raise ValueError(f'Invalid pose, determinant of the rotation matrix is {d}.')
        # 声波传播方向 [0, 0, 1]说明厂商规定的是z方向 通过offset转换为世界坐标系的y方向 再叠加rot_mat得到这个探头的独特方向
        direction = rot_mat @ torch.tensor([0, 0, 1], dtype=torch.float).reshape(-1, 1) # ray direction (3, 1)
        # 成像平面法向量 [1, 0, 0]说明厂商规定的是x方向 通过offset转换为世界坐标系的z方向 再叠加rot_mat得到这个探头的独特方向
        plane_normal = rot_mat @ torch.tensor([1, 0, 0], dtype=torch.float).reshape(-1, 1) # plane normal (3, 1)
        # 总结 转换后 在世界坐标系下 声波沿y轴向上传播 声波的起点在x轴上排列 成像平面法向量与z方向一致 
        return RayBundleLinear(origin, direction, plane_normal)
    

class TUESRECProbe(EmulatedUSDevice):

    def __init__(self, trained_model_path=None, subsample_slice=None) -> None:
        super().__init__(trained_model_path, subsample_slice)
        self.sample_parameters = []
        
    def pose_to_ray_bundle(self, pose):
        return RayBundleTUESREC(pose)
