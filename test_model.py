import torch
from torch.optim import lr_scheduler
from torch.nn import MSELoss
from torch.optim.lr_scheduler import LRScheduler
from pytorch_msssim import SSIM 
import numpy as np
from tqdm import tqdm, trange
import imageio

from .utils.utils import set_torch_default_device_to_cuda_if_available, save_side_by_side_image
from .utils.utils import TORCH_DEVICE as device
from .data.dataset import ClariusC3HD3Dataset, SyentheticDataset, TUESRECDataset
from .render.device import ClariusC3HD3, SyentheticLiverProbe, TUESRECProbe

def test(us_device, dataset):
    with torch.no_grad():
        for idx, (gt, pose) in enumerate(dataset):
            pred = us_device(pose)
            img1 = gt[us_device.subsample_slice].cpu().numpy()
            img2 = pred.cpu().numpy()
            save_side_by_side_image(img1, img2, f'/home/mirmi/Documents/UltraAssistant/refactor/test_result/{idx}.png')

def main():
    set_torch_default_device_to_cuda_if_available()
    test_dataset = ClariusC3HD3Dataset('/home/mirmi/Documents/UltraAssistant/model/dataset/Dec 16/test.h5')
    us_device = ClariusC3HD3('/home/mirmi/Documents/UltraAssistant/refactor/nerf_30.pt', subsample_slice=(slice(None, None, None), slice(None, None, None)))
    # test_dataset = SyentheticDataset('/home/mirmi/Documents/UltraAssistant/model/ultra-nerf/data/synthetic_testing/l2')
    # us_device = SyentheticLiverProbe('/home/mirmi/Documents/UltraAssistant/refactor/nerf_50.pt')
    # test_dataset = TUESRECDataset('/home/mirmi/Documents/UltraAssistant/open datasets/TUES-REC/000/LH_Par_C_DtP.h5')
    # us_device = TUESRECProbe('/home/mirmi/Documents/UltraAssistant/refactor/trained models/tuesrec.pt', subsample_slice=(slice(70, 400, None), slice(160, 480, None)))
    test(us_device, test_dataset)

if __name__ == '__main__':
    main()