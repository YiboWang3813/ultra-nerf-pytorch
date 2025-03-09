import torch
from torch.optim import lr_scheduler
from torch.utils.data import random_split
from torch.nn import MSELoss
from torch.optim.lr_scheduler import LRScheduler
from pytorch_msssim import SSIM
import numpy as np
from tqdm import tqdm, trange

from .utils.utils import set_torch_default_device_to_cuda_if_available
from .utils.utils import TORCH_DEVICE as device
from .utils.inerf_helpers import camera_transf, camera_transf_translation_only
from .data.dataset import ClariusC3HD3Dataset, SyentheticDataset, TUESRECDataset
from .render.device import ClariusC3HD3, SyentheticLiverProbe, TUESRECProbe
from .utils.utils import loss_function

def inerf(image: torch.Tensor, initial_guess, us_device):
    tf = camera_transf_translation_only().to(device)
    epochs = 50
    start_lr, end_lr = 1e-3, 1e-4
    adam = torch.optim.Adam(tf.parameters(), lr=start_lr)
    scheduler = lr_scheduler.ExponentialLR(adam, gamma=pow(end_lr / start_lr, 1 / epochs))

    for _ in range(epochs):
        pred = us_device(tf(initial_guess))
        loss, _, _ = loss_function(pred, image[us_device.subsample_slice])
        adam.zero_grad()
        loss.backward()
        adam.step()
        scheduler.step()

    return tf(torch.eye(4))

def get_delta_pose(image: torch.Tensor, target: torch.Tensor, us_device, pose: torch.Tensor=None, initial_guess: torch.Tensor=None):
    if pose is None and initial_guess is None:
        raise ValueError('One of pose_1 and initial_guess_1 must be provided.')
    if pose is None:
        pose = inerf(image, initial_guess, us_device).detach() @ initial_guess
    delta_pose = inerf(target, pose, us_device)
    return delta_pose

def estimate_trajectory(images: torch.Tensor, start_pose, us_device, max_num_images=1000):
    trajectory = torch.zeros((len(images), 4, 4))
    trajectory[0] = start_pose
    for i in trange(1, min(len(trajectory), max_num_images)):
        delta = get_delta_pose(images[i - 1], images[i], us_device, initial_guess=trajectory[i - 1])
        with torch.no_grad():
            trajectory[i] = delta @ trajectory[i - 1]
    return trajectory

def estimate_trajectory_cheaty(images: torch.Tensor, poses, us_device, max_num_images=1000):
    trajectory = torch.zeros((len(images), 4, 4))
    trajectory[0] = poses[0]
    for i in trange(1, min(len(trajectory), max_num_images)):
        delta = get_delta_pose(images[i - 1], images[i], us_device, pose=poses[i - 1])
        with torch.no_grad():
            trajectory[i] = delta @ trajectory[i - 1]
    return trajectory

def test_region_of_convergence(image, pose, us_device, num_points=1000):
    std = torch.zeros_like(pose)
    std[:3, -1] = 1e-2
    initial_errors = initial_errors = np.empty((num_points, 3))
    success = np.empty(num_points, dtype=np.int8)
    for i in trange(num_points):
        error = torch.normal(torch.zeros_like(pose), std)
        initial_errors[i] = error[:3, -1].cpu().numpy()
        initial_guess = pose + error
        pose_diff = inerf(image, initial_guess, us_device)
        pose_estimated = pose_diff @ initial_guess
        estimation_error = (pose_estimated[:3, -1] - pose[:3, -1]).clone().detach().cpu().numpy()
        if np.linalg.norm(estimation_error) < 1e-3:
            success[i] = 1
        else:
            success[i] = 0
    print(initial_errors.shape)
    print(success.shape)
    print(initial_errors[:3])
    print(success[:3])
    np.save('initial erros.npy', initial_errors)
    np.save('success.npy', success)

def main():
    torch.autograd.set_detect_anomaly(True)
    set_torch_default_device_to_cuda_if_available()
    # test_dataset = ClariusC3HD3Dataset('/home/mirmi/Documents/UltraAssistant/model/dataset/Jan 16/test.h5')
    # us_device = ClariusC3HD3('/home/mirmi/Documents/UltraAssistant/refactor/trained models/phantom_jan_16.pt', subsample_slice=(slice(50, None, None), slice(None, None, None)))
    test_dataset = SyentheticDataset('/home/mirmi/Documents/UltraAssistant/model/ultra-nerf/data/synthetic_testing/l2')
    us_device = SyentheticLiverProbe('/home/mirmi/Documents/UltraAssistant/refactor/trained models/synthetic_nosubsample.pt', subsample_slice=(slice(None, None, None), slice(None, None, None)))
    # test_dataset = TUESRECDataset('/home/mirmi/Documents/UltraAssistant/open datasets/TUES-REC/000/LH_Par_C_DtP.h5')
    # us_device = TUESRECProbe('/home/mirmi/Documents/UltraAssistant/refactor/trained models/tuesrec.pt', subsample_slice=(slice(70, 400, None), slice(160, 480, None)))
    
    for step in [5, 10]:
        images = torch.from_numpy(test_dataset.images[::step]).cuda()
        trajectory_true = torch.from_numpy(test_dataset.poses[::step]).cuda()
        trajectory_estimated = estimate_trajectory(images, trajectory_true[0], us_device)
        # trajectory_estimated = estimate_trajectory_cheaty(images, trajectory_true, us_device)
        np.save(f'trajectory_true_{step}.npy', trajectory_true.cpu().numpy())
        np.save(f'trajectory_estimated_{step}.npy', trajectory_estimated.cpu().numpy())
    
    # image = torch.from_numpy(test_dataset.images[100]).cuda()
    # pose = torch.from_numpy(test_dataset.poses[100]).cuda()
    # test_region_of_convergence(image, pose, us_device)

if __name__ == '__main__':
    main()