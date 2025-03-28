
import torch 

num_points_per_ray = 10 
num_rays = 5 

distances_to_origin = torch.randn((num_points_per_ray, num_rays)) 
print(distances_to_origin) 

distances_between_points = torch.abs(distances_to_origin[..., 1:, :] - distances_to_origin[..., :-1, :])
print(distances_between_points.shape) 

distances_between_points_new = torch.abs(distances_to_origin[1:, :] - distances_to_origin[:-1, :])
print(distances_between_points_new.shape) 

print(distances_to_origin[..., 1:, :] == distances_to_origin[1:, :])