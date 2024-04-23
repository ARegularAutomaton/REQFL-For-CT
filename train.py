import torch

from rei.rei import REI

from dataset.cvdb import CVDB_CVPR

from physics.ct import CT

from transforms.shift import Shift
from transforms.rotate import Rotate

def main():
    device=f'cuda:0'

    pretrained = None
    lr_cos = False
    save_ckp = True
    report_psnr = True

    n_views = 90 # number of views
    lb = 45 # lb for angle range
    ub = 135 # ub for angle range
    tau = 10 # SURE

    epochs = 5000
    ckp_interval = 10
    schedule = [5000]

    batch_size = 1
    lr = {'G': 5e-4, 'WD': 1e-8}
    alpha = {'req': 1e3, 'sure': 1e-5, 'mc': 1, 'eq': 1}

    I0 = 1e5
    noise_sigma = 0.1
    noise_model = {'noise_type': 'g', # Gaussian
                    'sigma': noise_sigma}

    img_width = 256
    img_height = img_width

    dataloader = CVDB_CVPR(dataset_name='CT10', mode='train', batch_size=batch_size,
                            shuffle=True, crop_size=(img_width, img_height), resize=False)

    transform = Rotate(n_trans=1)

    physics = CT(img_width, n_views, circle=False, device=device, I0=I0,
                  noise_model=noise_model, ub=ub, lb=lb)

    rei = REI(in_channels=1, out_channels=1, img_width=img_width, img_height=img_height,
              dtype=torch.float, device=device)

    rei.train_rei(dataloader, physics, transform, epochs, lr, alpha, ckp_interval,
                  schedule, pretrained, lr_cos, save_ckp, tau, report_psnr)

if __name__ == '__main__':
    main()