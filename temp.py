from PIL import Image 
import numpy as np 

img = Image.open('/raid/liujie/code_recon/data/ultrasound/synthetic_liver/l1/images/9.png') 
img = np.array(img) 
print(img.shape) 