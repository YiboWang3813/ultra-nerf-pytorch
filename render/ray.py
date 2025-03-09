import torch
from torch import nn
from torch.nn.functional import normalize

class RayBundle:

    def __init__(self) -> None:
        self.points = None
        self.distances_to_origin = None
        self.directions = None

    def sample(self, *args, **kwargs):
        raise NotImplementedError
    
    def subsample(self, slice: tuple[slice, slice]):
        if self.points is None or self.distances_to_origin is None or self.directions is None:
            raise RuntimeError('subsample must be called after sample')
        self.points = self.points[slice]
        self.distances_to_origin = self.distances_to_origin[slice]
        self.directions = self.directions[slice]

class RayBundleFan(RayBundle):

    def __init__(self, origin, direction, plane_normal, central_angle):
        super().__init__()
        # Circular sector shaped ray bundle
        self.origin = origin.reshape(1, 3)                # origin of the circle
        self.central_angle = min(abs(central_angle), torch.pi)              # central angle in rad within [0, pi], defines how wide rays spread
        self.direction = normalize(direction.reshape(1, 3), dim=1)          # the center ray direction direction must be perpendicular to plane_normal 
        self.plane_normal = normalize(plane_normal.reshape(1, 3), dim=1)    # normal vector of the plane where the rays inhabit

    def sample(self, near, far, num_points_per_ray, num_rays, noisy=False, standard_deviation=5e-5):
        # sample points in a fan-shape area
        # near: distance between top most pixel to origin
        # far:  distance between bottom most pixel to origin
        # num_rays: width
        # num_points_per_ray: height
        distances_to_origin = torch.linspace(near, far, num_points_per_ray)
        points = self.origin.reshape(1, 1, 3) + distances_to_origin.reshape(-1, 1, 1) * self._get_ray_directions(num_rays).reshape(1, -1, 3)
        if noisy:
            points += torch.normal(0, standard_deviation, points.shape, device=torch.get_default_device())
        self.points = points
        self.directions = normalize(self.origin.reshape(1, 1, 3) - self.points, dim=-1)
        self.distances_to_origin = distances_to_origin.unsqueeze(-1).broadcast_to(points.shape[:2])

    def _get_ray_directions(self, num_rays):
        ray_angles = torch.linspace(-self.central_angle / 2, self.central_angle / 2, num_rays).reshape(1, -1)
        alphas = ray_angles[:, -(num_rays // 2):]
        sin_alphas = torch.sin(alphas).reshape(-1, 1)
        cos_alphas = torch.cos(alphas).reshape(-1, 1)
        cross = torch.linalg.cross(self.direction, self.plane_normal)
        dirs1 = sin_alphas * cross + cos_alphas * self.direction
        dirs2 = sin_alphas * -cross + cos_alphas * self.direction
        if num_rays & 1 == 0: # even
            directions = torch.concat([dirs1.flip([0]), dirs2], 0)
        else: # odd
            directions = torch.concat([dirs1.flip([0]), self.direction, dirs2], 0)
        return normalize(directions, dim=1)

class RayBundleLinear(RayBundle):

    def __init__(self, origin, direction, plane_normal):
        super().__init__()
        # Circular sector shaped ray bundle
        self.origin = origin.reshape(1, 3)                # origin of 
        self.direction = normalize(direction.reshape(1, 3), dim=1)          # the center ray direction direction must be perpendicular to plane_normal 
        self.plane_normal = normalize(plane_normal.reshape(1, 3), dim=1)    # normal vector of the plane where the rays inhabit
        if not torch.allclose(torch.dot(self.direction.flatten(), self.plane_normal.flatten()), torch.zeros(1), atol=5e-2):
            raise ValueError(f'Direction vector and plane normal vector must be perpendicular to each other, got {self.direction} and {self.plane_normal} whose dot product is {torch.dot(self.direction.flatten(), self.plane_normal.flatten())}')

    def sample(self, near, far, width, num_points_per_ray, num_rays):
        # sample points in a fan-shape area
        # near: distance between top most pixel to origin
        # far:  distance between bottom most pixel to origin
        # width: distance between left most ray and right most ray
        # num_rays: image width
        # num_points_per_ray: image height
        distances_to_origin = torch.linspace(near, far, num_points_per_ray)
        points = self._get_ray_origins(num_rays, width).reshape(1, -1, 3) + (distances_to_origin.reshape(-1, 1) * self.direction.reshape(1, 3)).reshape(-1, 1, 3)
        self.points = points
        self.directions = normalize(self.origin.reshape(1, 1, 3) - self.points, dim=-1)
        self.distances_to_origin = distances_to_origin.unsqueeze(-1).broadcast_to(points.shape[:2])
        return self.points, self.distances_to_origin

    def _get_ray_origins(self, num_rays, width):
        distances = torch.linspace(-width / 2, width / 2, num_rays).reshape(-1, 1)
        line_of_origins = torch.linalg.cross(self.direction, self.plane_normal).reshape(1, 3)
        return self.origin + distances * line_of_origins

class RayBundleTUESREC(RayBundle):

    def __init__(self, pose):
        super().__init__()
        self.tool_T_img = torch.tensor([
            [ 0.231064671309448, -0.218052035041340,  0.948189025293473, -70.7413291931152],
            [-0.190847036221471, -0.965787272832032, -0.175591436013112, -80.6505661010742],
            [ 0.954036962825936, -0.140386087807861, -0.264773903381482, -46.1766223907471],
            [ 0,                  0,                  0,                   1              ],
        ])
        self.mm_T_pixel = torch.tensor([
            [0.229389190673828, 0,                 0, 0],
            [0,                 0.220979690551758, 0, 0],
            [0,                 0,                 1, 0],
            [0,                 0,                 0, 1],
        ])
        self.height = 480
        self.width = 640
        # [160 : 480 , 70 : 400]
        self.pose = pose


    def sample(self):
        pixels = torch.empty((4, self.height * self.width), dtype=torch.float32) # 4 for homognous coordinate
        x, y = torch.arange(0, self.height, dtype=torch.float32), torch.arange(0, self.width, dtype=torch.float32)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        pixels[0,:] = grid_x.reshape(-1)
        pixels[1,:] = grid_y.reshape(-1)
        pixels[2,:] = 0
        pixels[3,:] = 1
        world = self.pose @ self.tool_T_img @ self.mm_T_pixel @ pixels
        world = world.reshape(4, len(x), len(y)).permute(1, 2, 0)
        self.points = world[:, :, :3] * 1e-3
        distances_to_origin = torch.linspace(0, self.mm_T_pixel[1, 1] * (self.height - 1), self.height)
        self.distances_to_origin =  distances_to_origin.unsqueeze(-1).broadcast_to(self.points.shape[:2])
        self.directions = torch.empty_like(self.points) # place holder should not be accessed
        return self.points, self.distances_to_origin