# from PIL import Image 
# import numpy as np 

# img = Image.open('/raid/liujie/code_recon/data/ultrasound/synthetic_liver/l1/images/9.png') 
# img = np.array(img) 
# print(img.shape) 


import torch 
import torch.nn.functional as F 

# x = torch.Tensor([1, 2, 3]) 
# x_norm = F.normalize(x.reshape(1, 3), dim=1) 

# print(x, x_norm)

x = torch.Tensor([1, 2, 3, 4, 5])
x1 = x[1:] 
x2 = x[:-1]

diff = x1 - x2 

print(diff) 

