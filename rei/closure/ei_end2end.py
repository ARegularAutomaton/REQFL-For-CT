import torch
import numpy as np
from utils.metric import cal_psnr, cal_mse, cal_psnr_complex


def closure_ei_end2end(net, dataloader, physics, transform, optimizer,
                       criterion, alpha, dtype, device, report_psnr):
    loss_mc_seq, loss_eq_seq, loss_seq, psnr_seq, mse_seq = [], [], [], [], []

    assert physics.name == 'ct'

    norm = lambda x: (x - physics.MIN) / (physics.MAX - physics.MIN)
    f = lambda fbp: net(norm(fbp)) * (physics.MAX - physics.MIN) + physics.MIN

    for i, x in enumerate(dataloader):
        x = x[0] if isinstance(x, list) else x
        if len(x.shape)==5:
            N,n_crops,C,H,W =x.shape
            x = x.view(N*n_crops, C,H,W)
        if len(x.shape)==3:
            x = x.unsqueeze(1)
        x = x.type(dtype).to(device) # GT

        simulation = 'radon'
        if simulation == 'radon':
            x = x * (physics.MAX - physics.MIN) + physics.MIN # normalize data

            meas0 = physics.A(x, add_noise=True)

            s_g = torch.log(physics.I0 / meas0)
            fbp_g = physics.pseudo(s_g)
            x1 = f(fbp_g)
            meas1 = physics.A(x1)

            loss_mc = alpha['mc'] * criterion(meas1, meas0)

            # EI: x2, x3
            x2 = transform.apply(x1)
            meas2 = physics.A(x2)
            fbp_2 = physics.pseudo(torch.log(physics.I0 / meas2))
            x3 = f(fbp_2)
            
            loss_eq = alpha['eq'] * criterion(norm(x3), norm(x2))
        elif simulation == 'astra':
            norm = lambda fbp: (fbp - torch.min(fbp))/(torch.max(fbp) - torch.min(fbp)) # normalize fbp
            f = lambda fbp: net(norm(fbp)) * (torch.max(fbp) - torch.min(fbp)) + torch.min(fbp)
            f_net = lambda fbp: net(norm(fbp))

            meas0 = physics.A(x, add_noise=True)

            fbp_g = physics.pseudo(meas0)
            x1 = f(fbp_g)
            meas1 = physics.A(x1)
            loss_mc = alpha['mc'] * criterion(meas1, meas0)

            x1_net = f_net(x1)
            x2 = transform.apply(x1_net)
            meas_x2 = physics.A(x2)
            fbp_x2 = physics.pseudo(meas_x2)
            x3 = f_net(fbp_x2)
            loss_eq = alpha['eq'] * criterion(x3, x2)
        loss = loss_mc + loss_eq

        loss_mc_seq.append(loss_mc.item())
        loss_eq_seq.append(loss_eq.item())
        loss_seq.append(loss.item())

        if report_psnr:
            psnr_seq.append(cal_psnr(x1, x))
            mse_seq.append(cal_mse(x1, x))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_closure = [np.mean(loss_mc_seq), np.mean(loss_eq_seq), np.mean(loss_seq)]
    if report_psnr:
        loss_closure.append(np.mean(psnr_seq))
        loss_closure.append(np.mean(mse_seq))

    return loss_closure