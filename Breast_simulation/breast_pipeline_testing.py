import h5py
import os
import torch
import torch.fft as fft
from scipy.interpolate import pchip_interpolate
import numpy as np
import scipy.io as sio
import torch.nn as nn
import torch.optim as optim
import math
from numpy.fft import fftshift, ifftshift, fft2
# from sigpy import linop
# import sigpy as sp

from dce import DCE
import dce


# TEST - ksp to image space , matching jo's code
def fft2c(x, axes=(-2, -1), norm='ortho'):
    x = torch.fft.fftshift(x, dim=axes)
    x = fft.ifft2(x, dim=axes, norm=norm)

    # center the kspace
    x = torch.fft.fftshift(x, dim=axes)
    return x

# TEST - image to kspace
def ifft2c(x, axes=(-2, -1), norm='ortho'):
    x = torch.fft.fftshift(x, dim=axes)
    x = fft.fft2(x, dim=axes, norm=norm)

    # center the kspace
    x = torch.fft.fftshift(x, dim=axes)
    return x


class DCE(nn.Module):
    def __init__(self,
                 ishape,
                 sample_time,
                 sig_baseline,
                 R1,
                 Cp,
                 M0=5.0,
                 R1CA=4.30,
                 FA=15.,
                 TR=0.006,
                 x_iscomplex=True,
                 device=torch.device('cuda')):
        super(DCE, self).__init__()

        if x_iscomplex:
            self.ishape = list(ishape[:-1])
        else:
            self.ishape = list(ishape)

        self.sample_time = torch.tensor(np.squeeze(sample_time), dtype=torch.float32, device=device)
        self.sig_baseline = sig_baseline

        self.R1 = torch.tensor(np.array(R1), dtype=torch.float32, device=device)
        self.M0 = torch.tensor(np.array(M0), dtype=torch.float32, device=device)
        self.R1CA = torch.tensor(np.array(R1CA), dtype=torch.float32, device=device)
        self.FA = torch.tensor(np.array(FA), dtype=torch.float32, device=device)
        self.TR = torch.tensor(np.array(TR), dtype=torch.float32, device=device)

        self.FA_radian = self.FA * np.pi / 180.
        E1 = torch.exp(-self.TR * self.R1)

        topM0 = self.sig_baseline * (1 - torch.cos(self.FA_radian) * E1)
        bottomM0 = torch.sin(self.FA_radian) * (1 - E1)
        self.M0 = topM0 / bottomM0
        self.M0_trans = self.M0 * torch.sin(self.FA_radian)
        self.M_steady = self.M0_trans * (1 - E1) / (1 - E1 * torch.cos(self.FA_radian))

        # Cp_DCE = dce.arterial_input_function(sample_time)

        # read in breast sim aif, resize for shape [numpoints, ]
        self.Cp = torch.tensor(Cp, dtype=torch.float32, device=device)
        self.Cp = self.Cp.transpose(0, 1).view(-1)

        self.device = device

    def _check_ishape(self, input):
        for i1, i2 in zip(input.shape, self.ishape):
            if i1 != i2:
                raise ValueError(
                    'input shape mismatch for {s}, got {input_shape}'.format(s=self, input_shape=input.shape))

    def matmul_complex(self, torch1, t2):
        return torch.view_as_complex(
            torch.stack((torch1.real @ t2.real - torch1.imag @ t2.imag, torch1.real @ t2.imag + torch1.imag @ t2.real),
                        dim=2))

    # TEST - ksp to image space , matching jo's code
    def fft2c(self, x, axes=(-2, -1), norm='ortho'):
        x = torch.fft.fftshift(x, dim=axes)
        x = fft.ifft2(x, dim=axes, norm=norm)

        # center the kspace
        x = torch.fft.fftshift(x, dim=axes)
        return x

    # TEST - image to kspace
    def ifft2c(self, x, axes=(-2, -1), norm='ortho'):
        x = torch.fft.fftshift(x, dim=axes)
        x = fft.fft2(x, dim=axes, norm=norm)

        # center the kspace
        x = torch.fft.fftshift(x, dim=axes)
        return x

    def convolve(self, x, y):
        # Using PyTorch's conv1d with appropriate padding and kernel size
        return torch.conv1d(x.unsqueeze(0), y.unsqueeze(0), padding=(len(y) - 1) // 2)


    # equation 1 in the paper
    def _param_to_conc_patlak(self, x):
        # t1_idx = torch.nonzero(self.sample_time)
        # t1 = self.sample_time[t1_idx]

        # convert sample time to every 1 second
        t_end = self.sample_time[-1]
        step_size = 0.1
        t_step = torch.arange(0, t_end + step_size, step=step_size, dtype=torch.float32)

        dt = torch.diff(t_step, dim=0)
        K_time = torch.cumsum(self.Cp, dim=0) * dt[-1]  # convolution - size: [22]

        mult = torch.stack((K_time, self.Cp), 1)  # [22, 2]

        xr = torch.reshape(x, (self.ishape[0], np.prod(self.ishape[1:])))  # [2, 102400 (320*320)]
        yr = torch.matmul(mult, xr)  # [22, 102400]

        oshape = [len(self.sample_time)] + self.ishape[1:]  # [22, 1, 320, 320]
        yr = torch.reshape(yr, tuple(oshape))  # reshape yr to [22, 1, 320, 320 ] --> concentration curve
        return yr

    def _param_to_conc_tcm(self, x):
        # sample time to every 1 second with step_size
        t_end = self.sample_time[-1]  # 150 s --> 2.5 min
        step_size = 1  # s

        # multiply the aif by all the tissues- assume same aif
        t_samp = np.arange(0, t_end + step_size, step_size)
        aifci = pchip_interpolate(self.sample_time, self.Cp, t_samp)
        aifci_tens = torch.tensor(aifci, dtype=torch.float32)

        aif_map = aifci_tens.unsqueeze(-1).unsqueeze(-1).repeat(1, 320, 320)
        aif_map = torch.unsqueeze(aif_map, dim=1)

        # # initialize x with 1E-8 so not zeros, make it no grad so the operation is not differentiable
        x_check = x.clone()
        with torch.no_grad():
            if torch.all(x_check == 0):
                x = x_check.fill_(1E-8)

        # x should have size [4, 1, x, y]
        ve = x[0, ...]
        vp = x[1, ...]
        fp = x[2, ...]
        PS = x[3, ...]

        Ce = torch.zeros_like(aif_map)  # [t_sampling_points, 1, x, y]
        cp = torch.zeros_like(aif_map)

        for i in range(1, len(t_samp)):
            dt = t_samp[i] - t_samp[i - 1]

            Ce_prev = Ce[i - 1].clone()
            cp_prev = cp[i - 1].clone()

            d_cp = fp * aifci_tens[i - 1] + (PS * Ce_prev) - (fp + PS) * cp_prev
            d_ce = PS * cp_prev - PS * Ce_prev

            dcp_dt = d_cp * dt
            dce_dt = d_ce * dt

            # set break here if ve < 1E-8 then set ce[i] = ce_prev, same with vp
            if torch.all(vp < 1E-8):
                cp[i] = cp_prev

            if torch.all(ve < 1E-8):
                Ce[i] = Ce_prev

            if torch.any(vp > 1E-8):
                der_vp = torch.div(dcp_dt, vp)
                cp[i] = cp_prev + der_vp

            if torch.any(ve > 1E-8):
                der_ve = torch.div(dce_dt, ve)
                Ce[i] = Ce_prev + der_ve

        conc = vp.repeat(len(t_samp), 1, 1, 1) * cp + ve.repeat(len(t_samp), 1, 1, 1) * Ce

        # bool log indices where sampling occurred
        logIdx = np.zeros(len(t_samp))
        start_idx = 0
        for i in range(len(self.sample_time)):
            for j in range(start_idx, len(t_samp)):
                if self.sample_time[i] <= t_samp[j]:
                    logIdx[j] = 1
                    start_idx = j
                    break
        logIdx = logIdx.astype(bool)

        conc = conc[logIdx, ...]  # back to shape [22, 320, 320]

        # # Check for NaN values in the final concentration tensor and replace them with zeros
        conc = torch.where(torch.isnan(conc), torch.zeros_like(conc), conc)

        return conc

    def _param_to_conc_tcm_biexponent(self, x):
        # This one is most promising - just need to find the right convolve function
        t = self.sample_time

        # # initialize x with 1E-8 so not zeros, make it no grad so the operation is not differentiable
        x_check = x.clone()
        with torch.no_grad():
            if torch.all(x_check == 0):
                x = x_check.fill_(1E-8)

        # x should have size [4, 1, x, y]
        ve = x[0, ...]
        vp = x[1, ...]
        fp = x[2, ...]
        ps = x[3, ...]

        E = ps / (ps + fp)
        e = ve / (vp + ve)
        Ee = E * e

        # tau_mult = (E - Ee + e) / (2. * E)
        # tau_quad = 1 - 4 * (Ee * (1 - E) * (1 - e)) / (E - Ee + e) ** 2
        #
        # tau_plus = tau_mult * (1 + torch.sqrt(tau_quad))
        # tau_minus = tau_mult * (1 - torch.sqrt(tau_quad))
        #
        # F_plus = fp * ((tau_plus - 1.) / (tau_plus - tau_minus))
        # F_minus = -fp * ((tau_minus - 1.) / (tau_plus - tau_minus))
        #
        # K_plus = fp / ((vp + ve) * tau_minus)
        # K_minus = fp / ((vp + ve) * tau_plus)
        #
        # R_t = F_plus * torch.exp(-t*K_plus) + F_minus * torch.exp(-t*K_minus)

        tau_pluss = (E - Ee + e) / (2. * E) * (1 + torch.sqrt(1 - 4 * (Ee * (1 - E) * (1 - e)) / (E - Ee + e) ** 2))
        tau_minus = (E - Ee + e) / (2. * E) * (1 - torch.sqrt(1 - 4 * (Ee * (1 - E) * (1 - e)) / (E - Ee + e) ** 2))

        F_pluss = fp * (tau_pluss - 1.) / (tau_pluss - tau_minus)
        F_minus = -fp * (tau_minus - 1.) / (tau_pluss - tau_minus)

        K_pluss = fp / ((vp + ve) * tau_minus)
        K_minus = fp / ((vp + ve) * tau_pluss)

        two_compartment_model = F_pluss * torch.exp(-t * K_pluss) + F_minus * torch.exp(-t * K_minus)
        #tcm = math.NP.convolve(R_t, self.Cp, self.sample_time) - the original line of code
        tcm = self.convolve(R_t, self.Cp)
        return tcm

    def _param_to_conc_tcm_sys_of_eqns(self, x):
        # this method may not work because of the system of equations - not a simple matrix equation like you thought
        # sample time to every 1 second with step_size
        t_end = self.sample_time[-1]  # 150 s --> 2.5 min
        step_size = 1  # s

        # multiply the aif by all the tissues- assume same aif
        t_samp = np.arange(0, t_end + step_size, step_size)
        aifci = pchip_interpolate(self.sample_time, self.Cp, t_samp)
        aifci_tens = torch.tensor(aifci, dtype=torch.float32)

        aif_map = aifci_tens.unsqueeze(-1).unsqueeze(-1).repeat(1, 320, 320)
        aif_map = torch.unsqueeze(aif_map, dim=1)

        x_check = x.clone()
        with torch.no_grad():
            if torch.all(x_check == 0):
                x = x_check.fill_(1E-8)

        # x should have size [4, 1, x, y]
        ve = x[0, ...]
        vp = x[1, ...]
        Fp = x[2, ...]
        PS = x[3, ...]

        alpha = (PS * Fp) / (ve * vp)
        beta = ((Fp * ve) / ((ve + vp) * PS)) + 1 / (ve + vp)
        gamma = ((ve + vp) * PS * Fp) / (ve * vp)

        B = torch.stack((alpha, beta, gamma, Fp), dim = 0) # [4, 1, 320, 320]
        C = aifci

        A = torch.linalg.solve(B, C)

        # vp = (torch.square(Fp)) / (beta * gamma - alpha)
        # ve = (gamma / alpha) - vp
        # PS = (alpha * ve * vp) / Fp

        return A


    # shape of x should be [2, c, y, x]
    def forward(self, x):
        self._check_ishape(x)

        # parameters (k_trans, v_p) to concentration
        #CA = self._param_to_conc_tcm(x)
        #A = self._param_to_conc_tcm_sys_of_eqns(x)
        C = self._param_to_conc_tcm_biexponent(x)

        # CA = self._param_to_conc_patlak(x_c)

        # concentration to MR signal - equation 2 in the paper
        E1CA = torch.exp(-self.TR * (self.R1 + self.R1CA * CA))
        CA_trans = self.M0_trans * (1 - E1CA) / (1 - E1CA * torch.cos(self.FA_radian))

        sig = CA_trans + self.sig_baseline - self.M_steady
        ksp = self.ifft2c(sig)

        return ksp


########################################################################################################################
TCM = True

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(device)
DIR = os.path.dirname(os.path.realpath(__file__))
path = '../Breast Sim Pipeline'

# TODO:
breast_img = sio.loadmat('breast_cart_img.mat')['IMG']
# breast_img = sio.loadmat('breast_rad_img_full.mat')['undersamp_grog_img']

breast_img = np.transpose(breast_img, (2, 0, 1))
breast_img = breast_img[:, None, ...]
print('> img shape: ', breast_img.shape)

# only in radial case !
# breast_img = np.flip(breast_img, axis = 2)
# breast_img = np.flip(breast_img, axis = 3)

breast_img_tens = torch.tensor(breast_img.copy(), dtype=torch.float32, device=device)
t1_map = sio.loadmat('Breast_T1.mat')['T10']
r1 = (1 / t1_map)

aif = sio.loadmat('breast_aif.mat')['aif']

breast_ksp_tens = ifft2c(breast_img_tens)
meas_tens = breast_ksp_tens
#meas_tens = breast_img_tens

# reading time array straight from DRO - t is in seconds !!
injection = sio.loadmat('t.mat')['t']
# sample_time = np.concatenate((delay, injection))
sample_time = injection
print('> sample time: ', sample_time)

# Baseline img
x0 = breast_img_tens[0, ...]

oshape = meas_tens.shape

if TCM:
    # TCM
    ishape = [4, 1] + list(oshape[-2:])

else:
    # Patlak
    ishape = [2, 1] + list(oshape[-2:])  # extract the first two elements: [320, 320]

olen = np.prod(oshape)
x = torch.zeros(ishape, dtype=torch.float32,
                requires_grad=True, device=device)

# When using GT parmap from DRO
# parmap = sio.loadmat('parMap.mat')['parMap']
# parmap_tens = torch.tensor(parmap)
# parmap_r = parmap_tens.permute(2, 0, 1)
# parmap_u = parmap_r.unsqueeze(1)
# x = parmap_u

# with torch.no_grad():
#     x[0, ...] = 0.5
#     x[1, ...] = 0.5
#     x[2, ...] = 0.1
#     x[3, ...] = 0.1

model = DCE(ishape, sample_time, x0, r1, aif, device=device)
lossf = nn.MSELoss(reduction='sum')
# optimizer = optim.SGD([x], lr=0.001, momentum=0.9)
optimizer = optim.Adam([x], lr=0.01, amsgrad=True)

# put epoch stopper here when loss < 10e-4
epsilon = 1e-4  # Set the desired difference threshold

torch.autograd.set_detect_anomaly(True)
for epoch in range(50):
    # performs forward pass of the model with the current parameters 'x'
    fwd = model(x)

    # computes the MSE loss between the estimated kspace and the measured kspace data
    res = lossf(torch.view_as_real(fwd), torch.view_as_real(meas_tens))
    #res = lossf(fwd, meas_tens)  # in image space

    # Stop the loop if the condition is met
    if res.item() < epsilon:
        print(f'Stopping at epoch {epoch} because the loss is less than {epsilon}')
        break

    # clears the gradients accumulated in the previous iteration
    optimizer.zero_grad()

    # computes the gradients of the loss wrt the model parameters
    res.backward()
    optimizer.step()  # parameter update

    print('> epoch %3d loss %.9f' % (epoch, res.item() / olen))
    #print('vp' )

x_np = x.cpu().detach().numpy()
print(x_np.dtype)

meas_np = meas_tens.detach().numpy()

outfilestr = 'TCM_test.h5'

f = h5py.File(outfilestr, 'w')
f.create_dataset('param', data=x_np)
f.create_dataset('meas', data=meas_np)

f.close()

