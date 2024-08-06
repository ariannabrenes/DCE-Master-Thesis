import h5py
import os
import torch
import torch.fft as fft
from scipy.interpolate import pchip_interpolate
import numpy as np
import scipy.io as sio
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchaudio.transforms as T
from scipy import signal
# from dce import DCE

def check_for_nan(x, name, index):
    if torch.isnan(x).any():
        print(f"Nan values found in {name} at index {index}")


def check_for_inf(x, name, index):
    if torch.isinf(x).any():
        print(f"Inf values found in {name} at index {index}")


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
                 FA=10.,
                 TR=4.87E-3,
                 device=torch.device('cpu')):
        super(DCE, self).__init__()

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
        t_end = self.sample_time[-1]  # 150 s --> 2.5 min
        step_size = 0.1  # s

        # multiply the aif by all the tissues- assume same aif
        t_samp = torch.arange(0, t_end + step_size, step_size)

        t1_idx = torch.nonzero(t_samp)
        t1 = t_samp[t1_idx]
        dt = torch.diff(t1, dim=0)

        aifci = pchip_interpolate(self.sample_time, self.Cp, t_samp)
        aifci_tens = torch.tensor(aifci, dtype=torch.float32)

        K_time = torch.cumsum(aifci_tens, dim=0) * dt[-1]
        mult = torch.stack((K_time, aifci_tens), 1)  # [1502, 2]

        xr = torch.reshape(x, (self.ishape[0], np.prod(self.ishape[1:])))  # [2, 102400 (320*320)]
        yr = torch.matmul(mult, xr)  # [1502, 102400]

        oshape = [len(t_samp)] + self.ishape[1:]  # [1502, 1, 320, 320]
        yr = torch.reshape(yr, tuple(oshape))  # reshape yr to [1502, 1, 320, 320 ] --> concentration curve

        # Create a boolean mask for original sampling points
        logIdx = torch.zeros(len(t_samp), dtype=torch.bool)
        for i, t in enumerate(self.sample_time):
            closest_idx = (torch.abs(t_samp - t)).argmin().item()
            logIdx[closest_idx] = True

        conc = yr[logIdx, ...]  # back to shape [number of original sampling points, 320, 320]
        return conc

    def _param_to_conc_tcm(self, x):
        # sample time to every 1 second with step_size
        t_end = self.sample_time[-1]  # 150 s --> 2.5 min
        step_size = 0.5  # s

        # multiply the aif by all the tissues- assume same aif
        t_samp = np.arange(0, t_end + step_size, step_size)

        # t_samp = self.sample_time
        aifci = pchip_interpolate(self.sample_time, self.Cp, t_samp)
        aifci_tens = torch.tensor(aifci, dtype=torch.float32)

        # # Add delay to AIF
        # delay_samples = int(4 / step_size)  # -- seconds delay
        # aifci_delayed = np.roll(aifci, delay_samples)
        # aifci_delayed[:delay_samples] = 0  # zero out the initial values
        #
        # aifci_tens = torch.tensor(aifci_delayed, dtype=torch.float32)

        aif_map = aifci_tens.unsqueeze(-1).unsqueeze(-1).repeat(1, 320, 320)
        aif_map = torch.unsqueeze(aif_map, dim=1)

        # x should have size [4, 1, x, y]
        ve = abs(x[0, ...])
        vp = abs(x[1, ...])
        fp = abs(x[2, ...])
        PS = abs(x[3, ...])

        Ce = torch.zeros_like(aif_map)  # [t_sampling_points, 1, x, y]
        cp = torch.zeros_like(aif_map)

        epsilon = 1E-12
        for i in range(1, len(t_samp)):
            dt = t_samp[i] - t_samp[i - 1]

            Ce_prev = Ce[i - 1].clone()
            cp_prev = cp[i - 1].clone()

            d_cp = (fp * aifci_tens[i - 1] + (PS * Ce_prev) - (fp + PS) * cp_prev)
            d_ce = (PS * cp_prev - PS * Ce_prev)

            dcp_dt = d_cp * dt
            dce_dt = d_ce * dt

            # der_vp = torch.where(vp != 0, dcp_dt/vp, torch.ones_like(dcp_dt)*epsilon)
            der_vp = torch.where(vp == 0, torch.zeros_like(dcp_dt), dcp_dt / vp)
            cp[i] = cp_prev + der_vp

            # der_ve = torch.where(ve != 0, dce_dt/ve, torch.ones_like(dce_dt)*epsilon)
            der_ve = torch.where(ve == 0, torch.zeros_like(dce_dt), dce_dt / ve)
            Ce[i] = Ce_prev + der_ve

        vp_clone = vp.clone().repeat(len(t_samp), 1, 1, 1)
        ve_clone = ve.clone().repeat(len(t_samp), 1, 1, 1)

        conc = (vp_clone * cp + ve_clone * Ce)

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
        return conc

    def fft_convolve_time_only_torch(self, aif_map, kernel):
        # Initialize the result tensor with the appropriate shape
        result_shape = (aif_map.shape[0] + kernel.shape[0] - 1, aif_map.shape[1], aif_map.shape[2])
        conv_result = torch.zeros(result_shape, dtype=torch.float32, device=aif_map.device)

        # Perform 1D convolution along the time dimension for each spatial position
        for i in range(aif_map.shape[1]):
            for j in range(aif_map.shape[2]):
                # Extract the 1D signals
                aif_1d = aif_map[:, i, j].unsqueeze(0).unsqueeze(0)  # shape [1, 1, T]
                kernel_1d = kernel[:, i, j].flip(0).unsqueeze(0).unsqueeze(0)  # shape [1, 1, K]

                # Perform the 1D convolution
                conv_1d = F.conv1d(aif_1d, kernel_1d, padding=kernel_1d.shape[-1] - 1)

                # Assign the result to the appropriate slice of the result tensor
                conv_result[:, i, j] = conv_1d.squeeze(0).squeeze(0)[:result_shape[0]]

        # The resulting tensor shape is [t_samp*2 - 1, 320, 320]
        return conv_result

    def fft_convolve_time_only_batched(self, aif_map, kernel):
        # Get the shapes
        T, H, W = aif_map.shape
        K = kernel.shape[0]

        # Reshape aif_map to (H*W, 1, T)
        aif_map_batched = aif_map.permute(1, 2, 0).reshape(H * W, 1, T)

        # Reshape kernel to (H*W, 1, K)
        kernel_batched = kernel.permute(1, 2, 0).reshape(H * W, 1, K).flip(-1)

        # Perform the 1D convolution for each batch separately
        conv_result_batched = torch.cat(
            [F.conv1d(aif_map_batched[i:i + 1], kernel_batched[i:i + 1], padding=K - 1) for i in range(H * W)], dim=0)

        # Reshape the result back to (T+K-1, H, W)
        conv_result = conv_result_batched.reshape(H, W, -1).permute(2, 0, 1)

        # Return the result
        return conv_result[:T + K - 1, :, :]

    def fft_convolve(self, A, kernel):
        # Get the shapes
        T_sampling, H, W = kernel.shape

        # Zero-pad the AIF and kernel to the length of T_sampling * 2 - 1
        padded_length = 2 * T_sampling - 1
        aif_padded = torch.zeros(padded_length, device=A.device)
        kernel_padded = torch.zeros((padded_length, H, W), device=A.device)

        aif_padded[:T_sampling] = A
        kernel_padded[:T_sampling, :, :] = kernel

        # Perform FFT on the zero-padded AIF and Kernel
        aif_fft = torch.fft.fft(aif_padded)
        kernel_fft = torch.fft.fft(kernel_padded, dim=0)

        # Element-wise multiplication in the frequency domain
        result_fft = aif_fft[:, None, None] * kernel_fft

        # Perform inverse FFT to get the convolution result
        conv_result = torch.fft.ifft(result_fft, dim=0).real
        return conv_result

    def _param_to_conc_tcm_SnB(self, x):
        t_end = self.sample_time[-1]  # 150 s --> 2.5 min
        step_size = 0.1  # s

        # multiply the aif by all the tissues- assume same aif
        t_samp = torch.arange(0, t_end + step_size, step_size)
        aifci = pchip_interpolate(self.sample_time, self.Cp, t_samp)

        # Add delay to AIF
        delay_samples = int(3 / step_size)  # 3 seconds delay
        aifci_delayed = np.roll(aifci, delay_samples)
        aifci_delayed[:delay_samples] = 0  # zero out the initial values

        aifci_tens = torch.tensor(aifci_delayed, dtype=torch.float32)
        t_map = t_samp.unsqueeze(-1).unsqueeze(-1).repeat(1, 320, 320)
        aifMap = aifci_tens.unsqueeze(-1).unsqueeze(-1).repeat(1, 320, 320)

        # x should have size [4, 1, x, y]
        ve = x[0, ...]
        vp = x[1, ...]
        fp = x[2, ...]
        ps = x[3, ...]

        Te = ve / ps
        T = (vp + ve) / fp
        Tc = vp / fp

        Te[torch.isnan(Te)] = 0
        T[torch.isnan(T)] = 0
        Tc[torch.isnan(Tc)] = 0

        theta_plus = ((T + Te) + torch.sqrt((T + Te) ** 2 - 4 * Tc * Te)) / (2 * Tc * Te)
        theta_minus = ((T + Te) - torch.sqrt((T + Te) ** 2 - 4 * Tc * Te)) / (2 * Tc * Te)

        theta_plus[torch.isnan(theta_plus)] = 0
        theta_minus[torch.isnan(theta_minus)] = 0

        he = theta_plus * theta_minus * (
                (torch.exp(-t_map * theta_minus) - torch.exp(
                    -t_map * theta_plus)) /
                (theta_plus - theta_minus))

        hp = theta_plus * theta_minus * (
                ((1 - Te * theta_minus) * torch.exp(
                    -t_map * theta_minus) +
                 (Te * theta_plus - 1) * torch.exp(
                            -t_map * theta_plus)) /
                (theta_plus - theta_minus))

        he[torch.isnan(he)] = 0
        hp[torch.isnan(hp)] = 0

        ce = self.fft_convolve_time_only_batched(aifMap, he) * step_size
        cp = self.fft_convolve_time_only_batched(aifMap, hp) * step_size

        # check shapes here
        conc = vp * cp[:len(t_samp), ...] + ve * ce[:len(t_samp), ...]
        # conc = vp * cp + ve * ce

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

        conc = conc[logIdx, ...]
        conc = conc.unsqueeze(1)
        return conc

    def forward(self, x):
        self._check_ishape(x)

        # CA_dro = self._param_to_conc_tcm(x)
        # CA = self._param_to_conc_tcm_SnB(x)
        CA = self._param_to_conc_patlak(x)

        # concentration to MR signal - equation 2 in the paper
        E1CA = torch.exp(-self.TR * (self.R1 + self.R1CA * CA))
        CA_trans = self.M0_trans * (1 - E1CA) / (1 - E1CA * torch.cos(self.FA_radian))

        sig = CA_trans + (self.sig_baseline - self.M_steady)
        ksp = self.ifft2c(sig)

        return ksp


########################################################################################################################
TCM = False
Undersampled = False

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(device)
DIR = os.path.dirname(os.path.realpath(__file__))
data = sio.loadmat('DRO_data/Malignant/Run3_BC.mat')

# TODO:
breast_img = data['IMG']
breast_img = np.transpose(breast_img, (2, 0, 1))
breast_img = abs(breast_img[:, None, ...])  # TOOK ABS() HERE
print('> img shape: ', breast_img.shape)

breast_img_tens = torch.tensor(breast_img.copy(), dtype=torch.float32, device=device)
t1_map = data['T10']
r1 = (1 / t1_map)
aif = data['aif']
concentration = data['cts']

breast_ksp_tens = ifft2c(breast_img_tens)

if Undersampled:
    breast_ksp_tens[:, :, :, 1::2] = 0
    breast_img_tens = abs(fft2c(breast_ksp_tens))  # magnitude

# meas_tens = breast_ksp_tens
meas_tens = breast_ksp_tens

# reading time array straight from DRO - t is in seconds !!
sample_time = sio.loadmat('Data/t.mat')['t']
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
parmap = data['parMap']
parmap_r = parmap.transpose(2, 0, 1)
parmap_u = parmap_r[:, np.newaxis, ...]

parmap_patlak = parmap[:, :, [3, 1]]
parmap_tens = torch.tensor(parmap_patlak, dtype=torch.float32, requires_grad=True, device=device)

# with torch.no_grad():
# # #     # change so all the background values are 0
#     parmap_tens[parmap_tens == 1E-8] = 0
# #
# #     # # create nonzero masks for ve and vp
# #     # ve_nonzero = parmap_tens[0, 0, :, :] != 0
# #     # vp_nonzero = parmap_tens[1, 0, :, :] != 0
# #     #
# #     #  # do inverse of ve and vp only where they don't have a 0 value
# #     # parmap_tens[0, 0, :, :][ve_nonzero] = 1.0 / parmap_tens[0, 0, :, :][ve_nonzero]
# #     # parmap_tens[1, 0, :, :][vp_nonzero] = 1.0 / parmap_tens[1, 0, :, :][vp_nonzero]
# #
#     x = parmap_tens
# #     #
# #     # x[0, ...] = 1E-4
# #     # x[1, ...] = 1E-4
# #     # x[2, ...] = 1E-4
# #     # x[3, ...] = 1E-4

model = DCE(ishape, sample_time, x0, r1, aif, device=device)
lossf = nn.MSELoss(reduction='sum')
optimizer = optim.Adam([x], lr=0.005, amsgrad=True)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.001)
torch.autograd.set_detect_anomaly(True)

for epoch in range(500):
    # performs forward pass of the model with the current parameters 'x'
    fwd = model(x)
    check_for_nan(fwd, "Forward pass", epoch)

    # # print parameters at a specific pixel region - where it would be brightest for that parameter based on GT maps
    # ve_max = (x[0, 0, 270, 175])
    # vp_max = (x[1, 0, 210, 82])
    # fp_max = (x[2, 0, 270, 175])
    # PS_max = (x[3, 0, 280, 90])
    #
    # print(f've: {ve_max}, vp: {vp_max}, fp: {fp_max}, PS: {PS_max} ')

    # computes the MSE loss between the estimated kspace and the measured kspace data
    res = lossf(torch.view_as_real(fwd), torch.view_as_real(meas_tens))
    #res = lossf(fwd, meas_tens)
    check_for_nan(res, "loss", epoch)

    # clears the gradients accumulated in the previous iteration
    optimizer.zero_grad()
    torch.autograd.set_detect_anomaly(True)

    # computes the gradients of the loss wrt the model parameters
    res.backward()
    optimizer.step()  # parameter update
    # scheduler.step()

    print(f'Epoch {epoch}, Loss: {res.item()}, Learning Rate: {optimizer.param_groups[0]["lr"]}')

x_np = x.cpu().detach().numpy()

# fwd_img = fft2c(fwd)
fwd_np = fwd.cpu().detach().numpy()
print(x_np.dtype)
meas_np = meas_tens.detach().numpy()

outfilestr = 'Patlak_BC3_fullsamp_1.h5'
f = h5py.File(outfilestr, 'w')
f.create_dataset('param', data=x_np)
f.create_dataset('output', data=fwd_np)
f.close()
