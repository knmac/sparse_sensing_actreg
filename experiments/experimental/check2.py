import torch
from src.utils.misc import MiscUtils

N = 31
ROOT = 'debug/backup5_cuda9.2_cuda10.2_30iter'

for i in range(N):
    print('Iteration', i),
    run1_b = torch.load(ROOT+'/run1_b_{}'.format(i))
    run2_b = torch.load(ROOT+'/run2_b_{}'.format(i))
    MiscUtils.compare_dicts(run1_b, run2_b, verbose=True)
    print('')
