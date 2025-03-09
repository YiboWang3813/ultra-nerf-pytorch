import torch
from torch.nn.functional import conv2d

from ..nerf.model import RenderParameter, NerualRadianceField
from .ray import RayBundle
from ..utils.utils import add_a_leading_one, repeat_last_element, sample_bernoulli


def render_ray_bundle(ray_bundle: RayBundle, nerf_model: NerualRadianceField):
    # renders a ray bundle 
    gaussian_kernal_1d = torch.tensor([0.2790,	0.4420,	0.2790]) # sigma = 1.0
    gaussian_kernal_2d = gaussian_kernal_1d.reshape(-1, 1) * gaussian_kernal_1d.reshape(1, -1)

    if ray_bundle.points is None or ray_bundle.distances_to_origin is None:
        raise ValueError('Please sample points before render.')

    # nerf_query = torch.concatenate((ray_bundle.directions, ray_bundle.points), -1)
    nerf_query = ray_bundle.points
    render_parameter = RenderParameter(nerf_model(nerf_query))
    distances_to_origin = ray_bundle.distances_to_origin

    distances_between_points = torch.abs(distances_to_origin[..., 1:, :] - distances_to_origin[..., :-1, :])
    distances_between_points = repeat_last_element(distances_between_points, dim=-2)
    attenuation = torch.exp(-render_parameter.attenuation_coefficient * distances_between_points)
    attenuation_transmission = torch.cumprod(attenuation, dim=-2)

    border_indicator = sample_bernoulli(render_parameter.border_probability)
    reflection_transmission = 1 - render_parameter.reflection_coefficient * border_indicator
    reflection_transmission = add_a_leading_one(torch.cumprod(reflection_transmission[..., :-1, :], dim=-2), dim=-2)
    border_indicator = border_indicator.unsqueeze(0).unsqueeze(0)
    kernel = gaussian_kernal_2d.unsqueeze(0).unsqueeze(0)
    border_convolution = conv2d(border_indicator, kernel, padding='same')
    border_convolution = border_convolution.squeeze()
    
    scatterers = sample_bernoulli(render_parameter.scattering_density_coefficient)
    scatterers_map = scatterers * render_parameter.scattering_amplitude
    scatterers_map = scatterers_map.unsqueeze(0).unsqueeze(0)
    psf_scatter = conv2d(scatterers_map, kernel, padding='same').squeeze()

    transmission = attenuation_transmission * reflection_transmission
    b = transmission * psf_scatter
    r = transmission * render_parameter.reflection_coefficient * border_convolution
    intensity_map = b + r

    return intensity_map