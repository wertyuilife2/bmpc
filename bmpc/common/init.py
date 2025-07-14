import torch.nn as nn
from tensordict.nn import TensorDictParams
from common import layers

def weight_init(m):
	"""Custom weight initialization for TD-MPC2."""
	if isinstance(m, nn.Linear):
		nn.init.trunc_normal_(m.weight, std=0.02)
		if m.bias is not None:
			nn.init.constant_(m.bias, 0)
	elif isinstance(m, nn.Embedding):
		nn.init.uniform_(m.weight, -0.02, 0.02)
	elif isinstance(m, nn.ParameterList):
		for i,p in enumerate(m):
			if p.dim() == 3: # Linear
				nn.init.trunc_normal_(p, std=0.02) # Weight
				nn.init.constant_(m[i+1], 0) # Bias
	elif isinstance(m, layers.Ensemble): # fix the issue in https://github.com/nicklashansen/tdmpc2/issues/72
		# print("Q detected.", flush=True)
		nn.init.trunc_normal_(m.params["0","weight"], std=0.02)
		nn.init.trunc_normal_(m.params["1","weight"], std=0.02)
		nn.init.trunc_normal_(m.params["2","weight"], std=0.02)
		nn.init.constant_(m.params["0","bias"], 0)
		nn.init.constant_(m.params["1","bias"], 0)
		nn.init.constant_(m.params["2","bias"], 0)


def zero_(params):
	"""Initialize parameters to zero."""
	for p in params:
		p.data.fill_(0)
