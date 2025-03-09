import torch
from torch.optim import lr_scheduler
from torch.utils.data import random_split

from torch.optim.lr_scheduler import LRScheduler
import numpy as np
from tqdm import tqdm, trange
import wandb
import imageio

from .utils.utils import set_torch_default_device_to_cuda_if_available, loss_function
from .utils.utils import TORCH_DEVICE as device
from .data.dataset import ClariusC3HD3Dataset, SyentheticDataset, TUESRECDataset
from .render.device import ClariusC3HD3, SyentheticLiverProbe, TUESRECProbe

def log(items: dict[str, float]):
    use_wandb = True
    if not use_wandb:
        return
    global wandb_run
    if not 'wandb_run' in globals():
        wandb_run = wandb.init(project='ultra-nerf-pytorch-curveshape')
    wandb_run.log(items)

def train_one_epoch(us_device, dataset, optimizer: torch.optim.Optimizer):
    for gt, pose in tqdm(dataset):
        pred = us_device(pose)
        loss, _, _ = loss_function(pred, gt[us_device.subsample_slice])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def validate(us_device, dataset) -> tuple[float, float, float]:
    sum_loss, sum_ssim, sum_l2 = 0., 0., 0.
    with torch.no_grad():
        for gt, pose in dataset:
            pred = us_device(pose)
            loss, loss_ssim, loss_l2 = loss_function(pred, gt[us_device.subsample_slice])
            sum_loss += loss.item()
            sum_ssim += loss_ssim.item()
            sum_l2 += loss_l2.item()
    return (sum_loss / len(dataset), sum_ssim / len(dataset), sum_l2 / len(dataset))

def save_check_point(model, optimizer, epoch):  
    torch.save(model.state_dict(), f'nerf_{epoch}.pt')
    torch.save(optimizer.state_dict(), f'optimizer_{epoch}.pt')

def train(us_device, dataset):
    epochs = 30
    start_lr, end_lr = 1e-3, 1e-4
    adam = torch.optim.Adam(us_device.parameters(), lr=start_lr)
    scheduler = lr_scheduler.ExponentialLR(adam, gamma=pow(end_lr / start_lr, 1 / epochs))
    for epoch in trange(1, epochs + 1):
        train_set, val_set = random_split(dataset, [0.95, 0.05], torch.Generator(device=device))
        train_one_epoch(us_device, train_set, adam)
        validataion_losses = validate(us_device, val_set)
        log_items = {
            'loss'  : validataion_losses[0],
            'ssim'  : validataion_losses[1],
            'l2'    : validataion_losses[2],
            'lr'    : scheduler.get_last_lr()[0],
        }
        log(log_items)
        scheduler.step()

        if epoch % 10 == 0 or epoch == epochs:
            save_check_point(us_device.nerf, adam, epoch)

        # us_device.subsample_slice = (slice(None, None, None), slice(None, None, None))

def main():
    set_torch_default_device_to_cuda_if_available()
    training_dataset = ClariusC3HD3Dataset('/home/mirmi/Documents/UltraAssistant/model/dataset/Jan 16/train.h5')
    us_device = ClariusC3HD3(subsample_slice=(slice(50, None, None), slice(None, None, None)))
    # training_dataset = SyentheticDataset('/home/mirmi/Documents/UltraAssistant/model/ultra-nerf/data/synthetic_liver/complete_volume')
    # us_device = SyentheticLiverProbe(subsample_slice=(slice(50, None, None), slice(None, None, None))) # Note: important to ignore first several rows in the first epoch for synthetic liver dataset
    # training_dataset = TUESRECDataset('/home/mirmi/Documents/UltraAssistant/open datasets/TUES-REC/000/LH_Par_C_DtP.h5')
    # us_device = TUESRECProbe(subsample_slice=(slice(70, 400, None), slice(160, 480, None)))
    train(us_device, training_dataset)

if __name__ == '__main__':
    main()