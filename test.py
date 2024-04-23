import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.unet import UNet

from dataset.cvdb import CVDB_CVPR

from physics.ct import CT

from utils.metric import cal_psnr, cal_ssim

def test_ct(net_name, net_ckp, device):
    radon_view = 90
    I0 = 1e5
    sigma = 0.1
    image_size = 128

    noise_model = {'noise_type': 'mpg',
                   'sigma': sigma}

    unet = UNet(in_channels=1, out_channels=1, compact=4, residual=True,
                circular_padding=True, cat=True).to(device)

    dataloader = CVDB_CVPR(dataset_name='CT10', mode='test', batch_size=1,
                               shuffle=False, crop_size=(image_size, image_size), resize=False)

    radon_view = radon_view
    forw = CT(image_size, radon_view, circle=False, device=device, I0=I0, noise_model=noise_model)

    # normalize the input
    f = lambda fbp: unet((fbp - forw.MIN) / (forw.MAX - forw.MIN)) \
                    * (forw.MAX - forw.MIN) + forw.MIN

    psnr_fbp, psnr_net = [],[]
    ssim_fbp, ssim_net = [],[]

    for _, x in enumerate(dataloader):
        simulation = 'radon'
        if simulation == 'radon':
            x = x[0] if isinstance(x, list) else x
            if len(x.shape) == 3:
                x = x.unsqueeze(1)
            x = x.type(torch.float).to(device)

            x = (x * (forw.MAX - forw.MIN) + forw.MIN)
            y = forw.A(x, add_noise=True)
            fbp = forw.iradon(torch.log(forw.I0 / y))

            psnr_fbp.append(cal_psnr(fbp, x))
            ssim_fbp.append(cal_ssim(fbp, x))


            checkpoint = torch.load(net_ckp, map_location=device)
            unet.load_state_dict(checkpoint['state_dict'])
            unet.to(device).eval()
            x_net = f(fbp)

            psnr_net.append(cal_psnr(x_net, x))
            ssim_net.append(cal_ssim(x_net, x))

        elif simulation == 'astra':
            # normalize the input
            norm = lambda fbp: (fbp - torch.min(fbp))/(torch.max(fbp) - torch.min(fbp)) # normalize fbp
            f = lambda fbp: unet(norm(fbp)) * (torch.max(fbp) - torch.min(fbp)) + torch.min(fbp)

            x = x[0] if isinstance(x, list) else x
            if len(x.shape) == 3:
                x = x.unsqueeze(1)
            x = x.type(torch.float).to(device)

            y = forw.A(x, add_noise=True)
            fbp = forw.pseudo(y)
            fbp = norm(fbp)

            psnr_fbp.append(cal_psnr(fbp, x))

            checkpoint = torch.load(net_ckp, map_location=device)
            unet.load_state_dict(checkpoint['state_dict'])
            unet.to(device).eval()
            x_net = f(fbp)

            psnr_net.append(cal_psnr(x_net, x))
        show(x, fbp, x_net)

    print('AVG-PSNR (views={}\tI0={}\tsigma={})\t FBP={:.3f} + {:.3f}\t{}={:.3f} + {:.3f}'.format(
        radon_view,I0,sigma, np.mean(psnr_fbp),np.std(psnr_fbp), net_name, np.mean(psnr_net), np.std(psnr_net)))
    print('AVG-SSIM (views={}\tI0={}\tsigma={})\t FBP={:.3f} + {:.3f}\t{}={:.3f} + {:.3f}'.format(
        radon_view,I0,sigma, np.mean(ssim_fbp),np.std(ssim_fbp), net_name, np.mean(ssim_net), np.std(ssim_net)))

def show(x, fbp, x_net):
    if get_display_metric(x_net, x, 'psnr').astype(np.float64) > 20 and get_display_metric(x_net, x, 0).astype(np.float64) > 0.7:
        print(torch.max(x), torch.max(fbp), torch.max(x_net))
        plt.subplot(1,3,1)
        plt.axis('off')
        plt.imshow(fbp.detach().cpu().numpy().squeeze(), cmap='gray')
        plt.title('A')
        plt.text(126, 13, f'PSNR: {get_display_metric(fbp, x)}\n SSIM: {get_display_metric(fbp, x, 0)}', color='white', fontsize=12, ha='right')
        
        plt.subplot(1,3,2)
        plt.axis('off')
        plt.imshow(x_net.detach().cpu().numpy().squeeze(), cmap='gray')
        plt.title('B')
        plt.text(126, 13, f'PSNR: {get_display_metric(x_net, x)}\n SSIM: {get_display_metric(x_net, x, 0)}', color='white', fontsize=12, ha='right')

        plt.subplot(1,3,3)
        plt.axis('off')
        plt.imshow(x.detach().cpu().numpy().squeeze(), cmap='gray')
        plt.title('C')
        plt.text(126, 13, f'', color='white', fontsize=12, ha='right')

        plt.show()

def get_display_metric(bp, x, name='psnr'):
    if name == 'psnr':
        return np.around(cal_psnr(bp, x), decimals=2).astype(str)
    else:
        return np.around(cal_ssim(bp, x), decimals=2).astype(str)

def plot(path):
    files = list_files(path)
    headers = ["epoch","mc loss","eq loss","Total loss","PSNR","MSE","SSIM","GPU Memory"]
    # print(plt.style.available)
    plt.style.use('seaborn-v0_8')
    for f in range(len(files)):
        for c in range(8):
            if is_csv_file(files[f]) and c == 4:
                arr = np.genfromtxt(files[f], delimiter=",", skip_header=1, usecols=(0,c))
                arr = arr.swapaxes(0,1)
                plt.plot(arr[0], arr[1], label=headers[c])
    plt.xlabel("epochs")
    plt.ylabel("value")
    plt.title("PSNR during training")
    plt.legend()
    plt.show()
    plt.style.use('default')

def list_files(path):
    files = []
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files

def is_csv_file(file_path):
    _, file_extension = os.path.splitext(file_path)
    return file_extension.lower() == '.csv'

if __name__ == '__main__':
    device = 'cuda:0'

    # plot metrics
    path = 'ckp/'
    plot(path)
    
    # test network
    net_ckp_ct = 'ckp/'
    test_ct(net_name='rei',net_ckp=net_ckp_ct, device=device)