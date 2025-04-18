import torch
from torch import nn
from torchvision.ops import MLP
from torchvision import transforms

from utils.utils import cyclically_shift_dims_left

class NerualRadianceField(nn.Module):

    def __init__(self, input_transform=transforms.Lambda(lambda x: x)):
        super().__init__()
        self.input_transform = input_transform
        self.positional_encoding_dim = 20
        self.query_dim  = 3 # 3 for xyz in 3D space
        # self.query_dim  = 6 # 3 for xyz in 3D space and 3 for ray direction
        self.output_dim = 5 # 5 element parameter vector for physics-inspired rendering, see: "Ultra-NeRF: Neural Radiance Fields for Ultrasound Imaging"
        self.width = 256
        self.depth = 8
        self.positional_encoding = PositionalEncoding(self.positional_encoding_dim)
        self.mlp_input_dim = self.positional_encoding_dim * self.query_dim + self.query_dim # positional encoding is done separately on each query dimension
        self.mlp1 = MLP(
             self.mlp_input_dim, 
            [self.width for _ in range(self.depth // 2)]
        )
        self.mlp2 = MLP(
             self.mlp_input_dim + self.width, # skip connection in the middle
            [self.width for _ in range(self.depth // 2 - 1)] + [self.output_dim]
        )

    def forward(self, query):
        query_transformed = self.input_transform(query)
        pe = self.positional_encoding(query_transformed).flatten(start_dim=-2, end_dim=-1)
        query_transformed = torch.concat([query_transformed, pe], dim=-1)
        mlp_output = self.mlp2( torch.concat([query_transformed, self.mlp1(query_transformed)], dim=-1) )
        parameter_vector = torch.empty_like(mlp_output)
        parameter_vector[..., 0] = torch.abs(mlp_output[..., 0])
        parameter_vector[..., 1:] = torch.sigmoid(mlp_output[..., 1:])
        return parameter_vector

class PositionalEncoding:
    # see formula (4) of paper: "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis", output_dim == 2 * L

    def __init__(self, output_dim: int): 
        self.two_exponents = torch.exp2(torch.arange(output_dim // 2)) # half sine and half cosine

    def __call__(self, p: torch.Tensor):
        if not torch.is_tensor(p):
            p = torch.tensor(p, dtype=torch.float)

        if len(p.shape) == 0:
            # scalar input
            intermediate = torch.pi * p * self.two_exponents
            return torch.concat( [torch.sin(intermediate), torch.cos(intermediate)] ) # the sin and cos are not interlaced
        else:
            # when dealing with batched input a new dimension of size output_dim will be appended
            intermediate_shape = (len(self.two_exponents), ) + p.shape
            intermediate = torch.pi * (self.two_exponents.reshape(-1, 1) @ p.reshape(1, -1)).reshape(intermediate_shape)
            return cyclically_shift_dims_left(torch.concat( [torch.sin(intermediate), torch.cos(intermediate)] ))

class RenderParameter:
    
    def __init__(self, parameter_vector):

        # unpack the outputs of nerf model, split each to a physical parameter 
        self.attenuation_coefficient = parameter_vector[..., 0]
        self.reflection_coefficient = parameter_vector[..., 1] 
        self.border_probability = parameter_vector[..., 2]
        self.scattering_density_coefficient = parameter_vector[..., 3]
        self.scattering_amplitude = parameter_vector[..., 4] # (num_points_per_ray, num_rays)
