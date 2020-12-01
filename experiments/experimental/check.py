import torch

f102 = torch.load('debug/forward.10.2')
b102 = torch.load('debug/backward.10.2')
f92 = torch.load('debug/forward.9.2')
b92 = torch.load('debug/backward.9.2')

from src.utils.misc import MiscUtils

print('f102 - b102', MiscUtils.compare_dicts(f102, b102))
print('f102 - f92', MiscUtils.compare_dicts(f102, f92))
print('f102 - b92', MiscUtils.compare_dicts(f102, b92))
print('b102 - f92', MiscUtils.compare_dicts(b102, f92))
print('b102 - b92', MiscUtils.compare_dicts(b102, b92))
print('f92 - b92', MiscUtils.compare_dicts(f92, b92))
