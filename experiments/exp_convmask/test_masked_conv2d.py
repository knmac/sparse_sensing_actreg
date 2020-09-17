import numpy as np
import mmcv.ops
import torch
from time import time

conv_param = [3, 6, 3, 1, 0]
conv_param = {
    'in_channels': 3,
    'out_channels': 10,
    'kernel_size': 3,
    'stride': 1,
    'padding': 1,
}

conv_mask = mmcv.ops.MaskedConv2d(**conv_param).cuda()
conv_torch = torch.nn.Conv2d(**conv_param).cuda()
conv_torch.weight = conv_mask.weight
conv_torch.bias = conv_mask.bias

x = torch.rand(1, 3, 224, 224).cuda()
# mask = x[:, 0, :, :] > 0.5
mask = torch.zeros([1, 224, 224]).cuda()
mask[:, 224//4: 3*224//4, 224//4: 3*224//4] = 1.0
print(mask.shape, mask.sum())


for _ in range(10):
    conv_mask(x, mask)
    conv_torch(x)*mask.float().unsqueeze(1)


n_runs = 10
t0_lst = []
t1_lst = []
for i in range(n_runs):
    st = time()
    mask_result = conv_mask(x, mask)
    t0_lst.append(time() - st)

    st = time()
    # torch_result = conv_torch(x)*mask.float().unsqueeze(1)[:, :, :-2, :-2]
    torch_result = conv_torch(x)*mask.float().unsqueeze(1)
    t1_lst.append(time() - st)

t0_lst = np.array(t0_lst)
t1_lst = np.array(t1_lst)

print('MaskedConv2d runtime:  {:.06f} +- {:.06f}s'.format(t0_lst.mean(), t0_lst.std()))
print('Standard conv runtime: {:.06f} +- {:.06f}s'.format(t1_lst.mean(), t1_lst.std()))

diff = (mask_result-torch_result).abs().max()
print('Max difference:', diff.item())
