import torch

def has_nan_gradients(model):
	for param in model.parameters():
		if torch.isnan(param.grad).any():
			print('found nan gradients in parameters:')
			for name, param in model.named_parameters():
				if torch.isnan(param.grad).any():
					print(name)
			return True
	return False

def has_nan_parameters(model):
	for param in model.parameters():
		if torch.isnan(param.data).any():
			return True
	return False

def get_gradient_norms(model):
	grad_norms = {}
	for name, param in model.named_parameters():
		if param.grad is None:
			print(f'None gradient: {name}')
			grad_norm = 0
		else:
			grad_norm = param.grad.detach().norm().item()
		grad_norms[f'grad_norm/{name}'] = grad_norm
	return grad_norms

def map_params(model, f, key):
	mapped_params = {}
	for name, param in model.named_parameters():
		mapped_params[f'{key}/{name}'] = f(param.detach()).item()
	return mapped_params