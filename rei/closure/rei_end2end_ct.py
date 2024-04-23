import torch
import numpy as np
from utils.metric import cal_psnr, cal_mse

def rei_closure(net, dataloader, physics, transform, optimizer,
                        criterion, alpha, tau, dtype, device, reportpsnr=False,):
    loss_sure_seq, loss_req_seq, loss_seq, psnr_seq, mse_seq = [], [], [], [], []

    assert physics.name=='ct'

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
            norm = lambda x: (x - physics.MIN) / (physics.MAX - physics.MIN)
            f = lambda fbp: net(norm(fbp)) * (physics.MAX - physics.MIN) + physics.MIN

            x = x * (physics.MAX - physics.MIN) + physics.MIN # normalize data

            meas0 = physics.A(x, add_noise=True)

            s_g = torch.log(physics.I0 / meas0)
            fbp_g = physics.pseudo(s_g)
            x1 = f(fbp_g)
            meas1 = physics.A(x1)

            # SURE-based unbiased estimator to the clean measurement consistency loss
            assert physics.noise_model['noise_type'] == 'g'

            sigma2 = physics.noise_model['sigma'] ** 2
            b1 = torch.randn_like(meas0)

            fbp_2 = physics.pseudo(torch.log(physics.I0 / (meas0 + tau * b1)))

            meas2 = physics.A(f(fbp_2))

            K = meas0.shape[0]  # batch size
            m = meas0.shape[-1] * meas0.shape[-2] * meas0.shape[-3] # dimension of y

            loss_sure = alpha['sure'] * torch.sum((meas1 - meas0).pow(2)) / (K * m) - sigma2 \
                        + (2 * sigma2 / (tau *m * K)) * (b1 * (meas2 - meas1)).sum()

            x2 = transform.apply(x1)
            meas_x2 = physics.A(x2, add_noise=True)
            fbp_x2 = physics.pseudo(torch.log(physics.I0 / meas_x2))
            x3 = f(fbp_x2)
            loss_req = alpha['req'] * criterion(norm(x3), norm(x2))
        elif simulation == 'astra':
            norm = lambda fbp: (fbp - (torch.min(fbp)))/(torch.max(fbp) - torch.min(fbp)) # normalize
            f = lambda fbp: net(norm(fbp)) * (torch.max(fbp) - torch.min(fbp)) + torch.min(fbp) # invert and unnormalise
            f_net = lambda fbp: net(norm(fbp)) # invert normalised
            
            meas0 = physics.A(x, add_noise=True)

            fbp_g = physics.pseudo(meas0)
            x1 = f(fbp_g)
            meas1 = physics.A(x1)

            # SURE-based unbiased estimator to the clean measurement consistency loss
            assert physics.noise_model['noise_type'] == 'g'
            sigma2 = physics.noise_model['sigma'] ** 2
            b1 = torch.randn_like(meas0)

            fbp_2 = physics.pseudo(meas0 + tau * b1)

            meas2 = physics.A(f(fbp_2))

            K = meas0.shape[0]  # batch size
            m = meas0.shape[-1] * meas0.shape[-2] * meas0.shape[-3] # dimension of y

            loss_sure = torch.sum((meas1 - meas0).pow(2)) / (K * m) - sigma2 \
                        + (2 * sigma2 / (tau *m * K)) * (b1 * (meas2 - meas1)).sum()

            x1_net = f_net(fbp_g)
            x2 = transform.apply(x1_net)
            meas_x2 = physics.A(x2, add_noise=True)
            fbp_x2 = physics.pseudo(meas_x2)

            x3 = f_net(fbp_x2)
            loss_req = criterion(x3, x2)

        loss = loss_sure + loss_req
        loss_sure_seq.append(loss_sure.item())
        loss_req_seq.append(loss_req.item())
        loss_seq.append(loss.item())

        if reportpsnr:
            psnr_seq.append(cal_psnr(x1, x))
            mse_seq.append(cal_mse(x1, x))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_closure = [np.mean(loss_sure_seq), np.mean(loss_req_seq), np.mean(loss_seq)]

    if reportpsnr:
        loss_closure.append(np.mean(psnr_seq))
        loss_closure.append(np.mean(mse_seq))

    return loss_closure