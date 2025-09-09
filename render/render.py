import torch
from torch.nn.functional import conv2d

from nerf.model import RenderParameter, NerualRadianceField
from render.ray import RayBundle
from utils.utils import add_a_leading_one, repeat_last_element, sample_bernoulli


def render_ray_bundle(ray_bundle: RayBundle, nerf_model: NerualRadianceField):
    """ Render a ray bundle to get an intensity map with the nerf model. 
    
    Args:
        ray_bundle (RayBundle): The ray bundle as the source, providing points to be rendered.
        nerf_model (NerualRadianceField): The nerf model to generate render parameters.

    Returns:
        intensity_map (torch.Tensor): The intensity map of the target image. """

    gaussian_kernal_1d = torch.tensor([0.2790, 0.4420, 0.2790]) # sigma = 1.0
    gaussian_kernal_2d = gaussian_kernal_1d.reshape(-1, 1) * gaussian_kernal_1d.reshape(1, -1) # (3, 3)

    if ray_bundle.points is None or ray_bundle.distances_to_origin is None:
        raise ValueError('Please sample points before render.')

    # nerf_query = torch.concatenate((ray_bundle.directions, ray_bundle.points), -1)
    nerf_query = ray_bundle.points # (num_points_per_ray, num_rays, 3) 
    render_parameter = RenderParameter(nerf_model(nerf_query))
    distances_to_origin = ray_bundle.distances_to_origin # (num_points_per_ray, num_rays) 

    # calculate the distances between points, it should be the same value: (far - near) / num_points_per_ray 
    # distances_between_points[..., 1:, :] = distances_between_points[1:, :] 
    # 这里的[...]没有必要 相当于就是[1:, :] - [:-1, :]
    distances_between_points = torch.abs(distances_to_origin[..., 1:, :] - distances_to_origin[..., :-1, :]) # (num_points_per_ray-1, num_rays)
    distances_between_points = repeat_last_element(distances_between_points, dim=-2) # (num_points_per_ray, num_rays) 
    # calculate the attenuation in each point, then cumulative product all points along a ray to get the transmission
    attenuation = torch.exp(-render_parameter.attenuation_coefficient * distances_between_points)
    attenuation_transmission = torch.cumprod(attenuation, dim=-2) # (num_points_per_ray, num_rays) 

    border_indicator = sample_bernoulli(render_parameter.border_probability) # (num_points_per_ray, num_rays) 
    # calculate the reflection transmission 
    reflection_transmission = 1 - render_parameter.reflection_coefficient * border_indicator # (num_points_per_ray, num_rays)
    # 和上面一样 这里的[...]没有必要
    reflection_transmission = add_a_leading_one(torch.cumprod(reflection_transmission[..., :-1, :], dim=-2), dim=-2) # (num_points_per_ray, num_rays)
    # convolute the border indicator with a gaussian kernel to get the border convolution
    border_indicator = border_indicator.unsqueeze(0).unsqueeze(0) # (1, 1, num_points_per_ray, num_rays) 
    kernel = gaussian_kernal_2d.unsqueeze(0).unsqueeze(0) # (1, 1, 3, 3) 
    border_convolution = conv2d(border_indicator, kernel, padding='same')
    border_convolution = border_convolution.squeeze() # (num_points_per_ray, num_rays) 
    
    # calcuate the scatter map by producting the scattering density coefficient and amplitude 
    scatterers = sample_bernoulli(render_parameter.scattering_density_coefficient) # (num_points_per_ray, num_rays) 
    scatterers_map = scatterers * render_parameter.scattering_amplitude # (num_points_per_ray, num_rays) 
    scatterers_map = scatterers_map.unsqueeze(0).unsqueeze(0) # (1, 1, num_points_per_ray, num_rays) 
    # convolute the scatter map with a gaussian kernel to get the psf scatter
    psf_scatter = conv2d(scatterers_map, kernel, padding='same').squeeze() # (num_points_per_ray, num_rays) 

    transmission = attenuation_transmission * reflection_transmission # (num_points_per_ray, num_rays) 
    b = transmission * psf_scatter # scatter item 
    r = transmission * render_parameter.reflection_coefficient * border_convolution # reflection item 
    intensity_map = b + r

    return intensity_map # (num_points_per_ray, num_rays) 