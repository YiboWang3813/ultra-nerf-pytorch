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
        """ Sample points along the rays (Get all points' spatial coordinates). 
        
        Args:
            near (int): distance between top most pixel to origin (unit: mm) 
            far (int): distance between bottom most pixel to origin (unit: mm) 
            num_points_per_ray (int): height 
            num_rays (int): width 
            noisy (bool): whether to add noise to the sampled points 
            standard_deviation (float): standard deviation of the noise added to the sampled points
        
        Returns:
            points (torch.Tensor): sampled points' spatial coordinates (shape: [num_points_per_ray, num_rays, 3]) 
            directions (torch.Tensor): the unit direction vectors from all sampled points to the center (shape: [num_points_per_ray, num_rays, 3]) 
            distances_to_origin (torch.Tensor): the distance from all sampled points to the origin (shape: [num_points_per_ray, num_rays])"""
        distances_to_origin = torch.linspace(near, far, num_points_per_ray) # (num_points_per_ray,)
        # follow the formula: r = o + td, where o is origin, r is ray, t is distance to origin, d is direction 
        points = self.origin.reshape(1, 1, 3) + distances_to_origin.reshape(-1, 1, 1) * self._get_ray_directions(num_rays).reshape(1, -1, 3) # (num_points_per_ray, num_rays, 3)
        if noisy:
            points += torch.normal(0, standard_deviation, points.shape, device=torch.get_default_device())
        self.points = points
        # directions from all points to the origin 
        self.directions = normalize(self.origin.reshape(1, 1, 3) - self.points, dim=-1) # (num_points_per_ray, num_rays, 3)
        # distances from all points to the origin 
        self.distances_to_origin = distances_to_origin.unsqueeze(-1).broadcast_to(points.shape[:2]) # (num_points_per_ray, num_rays)

    def _get_ray_directions(self, num_rays):
        """ Get all rays' directions.
        
        Args:
            num_rays (int): number of rays 
        
        Returns:
            directions (torch.Tensor): directions of all rays, shape (num_rays, 3) """
        ray_angles = torch.linspace(-self.central_angle / 2, self.central_angle / 2, num_rays).reshape(1, -1) # (1, num_rays)
        alphas = ray_angles[:, -(num_rays // 2):]
        sin_alphas = torch.sin(alphas).reshape(-1, 1) # (num_rays // 2, 1)
        cos_alphas = torch.cos(alphas).reshape(-1, 1) # (num_rays // 2, 1) 
        # calculate the cross product of the center ray's direction and the plane normal to get the positive ray spread direction
        cross = torch.linalg.cross(self.direction, self.plane_normal) # (1, 3) 
        # compose all rays' directions in the positive axis 
        dirs1 = sin_alphas * cross + cos_alphas * self.direction # (num_rays // 2, 3) 
        # compose all rays' directions in the negative axis 
        dirs2 = sin_alphas * -cross + cos_alphas * self.direction # (num_rays // 2, 3) 
        if num_rays & 1 == 0: # even
            directions = torch.concat([dirs1.flip([0]), dirs2], 0)
        else: # odd
            directions = torch.concat([dirs1.flip([0]), self.direction, dirs2], 0)
        return normalize(directions, dim=1) # (num_rays, 3) 

class RayBundleLinear(RayBundle):

    def __init__(self, origin, direction, plane_normal):
        super().__init__()
        """ 
        Initialize the ray bundle of lieanr ultrasound probe. 

        Args: 
            origin (Tensor): the origin of ultrasound probe, shape=(3,).
            direction (Tensor): the propagating direction of ultrasound wave, shape=(3, 1).
            plane_normal (Tensor): the normal vector of imaging plane where ultrasound waves lie in, shape=(3, 1).
        """
        self.origin = origin.reshape(1, 3)                 
        self.direction = normalize(direction.reshape(1, 3), dim=1)           
        self.plane_normal = normalize(plane_normal.reshape(1, 3), dim=1)  
        if not torch.allclose(torch.dot(self.direction.flatten(), self.plane_normal.flatten()), torch.zeros(1), atol=5e-2):
            raise ValueError(f'Direction vector and plane normal vector must be perpendicular to each other, got {self.direction} and {self.plane_normal} whose dot product is {torch.dot(self.direction.flatten(), self.plane_normal.flatten())}')

    def sample(self, near, far, width, num_points_per_ray, num_rays):
        """ 
        Sample points in a rectangle area. 

        Args: 
            near (float): the near plane of ultrasound imaging depth, unit: m. 
            far (float): the far plane of ultrasound imaging depth, unit: m.
            width (float): the imaging width of this ultrasound probe, unit: m. 
            num_points_per_ray (int): equals to image height, unit: pixel. 
            num_rays (int): equals to image width, unit: pixel.
        
        Returns: 
            self.points (Tensor): positions of all sampling points, shape=(num_points_per_ray, num_rays, 3)
            self.directions (Tensor): directions from all sampling points to the origin of ultrasound probe, shape=(num_points_per_ray, num_rays, 3)
            self.distances_to_origin (Tensor): distances from all sampling points to its own origin, shape=(num_points_per_ray, num_rays)
        """
        distances_to_origin = torch.linspace(near, far, num_points_per_ray) # (num_points_per_ray,)
        # 通过函数 得到ray_origins (1, num_rays, 3)
        # 通过和directions相乘 赋予distances_to_origin在世界坐标系中的实际位置 (num_points_per_ray, 1, 3)
        # 广播二者 得到稠密的点坐标 (num_points_per_ray, num_rays, 3)
        points = self._get_ray_origins(num_rays, width).reshape(1, -1, 3) + \
                (distances_to_origin.reshape(-1, 1) * self.direction.reshape(1, 3)).reshape(-1, 1, 3)
        self.points = points
        # 从全部points到probe origin的连线方向 (num_points_per_ray, num_rays, 3)
        self.directions = normalize(self.origin.reshape(1, 1, 3) - self.points, dim=-1) 
        # 对每个origin都复制一份声波传播距离distances (num_points_per_ray, num_rays)
        self.distances_to_origin = distances_to_origin.unsqueeze(-1).broadcast_to(points.shape[:2])
        return self.points, self.distances_to_origin

    def _get_ray_origins(self, num_rays, width):
        """ 
        Get the origins of all rays.

        Args: 
            num_rays (int): number of rays which equals to image width.
            width (float): imaging width of this ultrasound probe. 
        
        Returns: 
            ray_origins (Tensor): the origins of all rays, shape=(num_rays, 3).
        """
        # 以探头为中心 均分成像宽度 得到num_rays个点
        distances = torch.linspace(-width / 2, width / 2, num_rays).reshape(-1, 1) # (num_rays, 1)
        # 根据右手定则 计算全部ray origins所在的x轴的方向
        line_of_origins = torch.linalg.cross(self.direction, self.plane_normal).reshape(1, 3) # (1, 3)
        # 方向和点距离相乘赋予distances在世界坐标系中的实际位置 再加探头origin把这些点挪到以探头为中心的两边
        return self.origin + distances * line_of_origins # (num_rays, 3)

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