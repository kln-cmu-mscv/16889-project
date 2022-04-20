import torch
import torch.nn as nn
import torchvision


# Directly regress the angles Theta and Phi.This does not incorporate any 
# bounds on the angles that are being predicted. Adding an activation
# function will not give us a linear range that we need. It will saturate 
# for the angles which are at the boundaries.
class PoseRegressor_baseline1(nn.Module):
	def __init__(self, pretrained=True, pool_first=True, **kwargs):
		super(PoseRegressor_baseline1, self).__init__()

		self.resnet = torchvision.models.resnet18(pretrained=pretrained)
		self.resnet.fc = nn.Linear(512,256)
		self.theta_head = nn.Sequential(
						nn.Linear(256, 128),
						nn.ReLU(),
						nn.Linear(128, 1)
		)
		self.phi_head = nn.Sequential(
						nn.Linear(256, 128),
						nn.ReLU(),
						nn.Linear(128, 1)
		)

	def forward(self, x):
		
		y = self.resnet(x)
		theta = self.theta_head(y)
		phi = self.phi_head(y)
		
		return [theta, phi]


# Predict the direction vectors and then calculate the angles theta and phi.
# This did not work quite well in comparison to the baseline.
class PoseRegressor_xyz(nn.Module):
	def __init__(self, pretrained=True, pool_first=True, **kwargs):
		super(PoseRegressor_xyz, self).__init__()

		self.resnet = torchvision.models.resnet18(pretrained=pretrained)
		self.resnet.fc = nn.Linear(512,256)
		self.x_head = nn.Sequential(
						nn.Linear(256, 128),
						nn.ReLU(),
						nn.Linear(128, 1)
		)
		self.y_head = nn.Sequential(
						nn.Linear(256, 128),
						nn.ReLU(),
						nn.Linear(128, 1)
		)
		self.x_head = nn.Sequential(
						nn.Linear(256, 128),
						nn.ReLU(),
						nn.Linear(128, 1)
		)
		self.z_head = nn.Sequential(
						nn.Linear(256, 128),
						nn.ReLU(),
						nn.Linear(128, 1)
		)
		
	def forward(self, x):
		
		y = self.resnet(x)
		x_coord = self.x_head(y)
		y_coord = self.x_head(y)
		z_coord = self.x_head(y)
		
		return torch.stack([x_coord, y_coord, z_coord], dim = 1)


# This is netwrok that is working. The network outputs sine and cosines of the
# angles.As compare to the baseline this incorporates appropriate bounds on the 
# outputs predicted by the network through arctan2 function.
class PoseRegressor_sincos(nn.Module):
	def __init__(self, pretrained=True, pool_first=True, **kwargs):
		super(PoseRegressor_sincos, self).__init__()

		self.resnet = torchvision.models.resnet18(pretrained=pretrained)
		self.resnet.fc = nn.Linear(512,256)
		self.theta_sin = nn.Sequential(
						nn.Linear(256, 128),
						nn.ReLU(),
						nn.Linear(128, 1)
		)
		self.theta_cos = nn.Sequential(
						nn.Linear(256, 128),
						nn.ReLU(),
						nn.Linear(128, 1)
		)
		self.phi_sin = nn.Sequential(
						nn.Linear(256, 128),
						nn.ReLU(),
						nn.Linear(128, 1)
		)
		self.phi_cos = nn.Sequential(
						nn.Linear(256, 128),
						nn.ReLU(),
						nn.Linear(128, 1)
		)

	def forward(self, x):
		y = self.resnet(x)
		theta_sin = self.theta_sin(y)
		theta_cos = self.theta_cos(y)
		phi_sin = self.phi_sin(y)
		phi_cos = self.phi_cos(y)
		
		return torch.stack([phi_sin, phi_cos, theta_sin, theta_cos ], dim = 0)


if __name__ == "__main__":
	
	net = PoseRegressor_sincos(pretrained=True)

	data = torch.autograd.Variable(torch.randn(16,3,224,224))
	output = net(data)
	print(output.shape)
	