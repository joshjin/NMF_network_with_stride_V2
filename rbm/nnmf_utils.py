import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torch.nn as nn

class NNMF_Block(nn.Module):
	def __init__(self,weight,stride=1, padding=0):
		super(NNMF_Block,self).__init__()
		self.register_buffer('weight',weight)
		self.stride = stride
		self.padding = padding
	def forward(self,x):
		x = F.conv2d(x, Variable(self.weight,requires_grad=False), stride = self.stride, padding = (self.padding,self.padding))
		return x

class NNMF_CNN(nn.Module):
	def __init__(self,weights):
		super(NNMF_CNN,self).__init__()
		self.nnmf = self._gen_nnmf_filters(weights)
	def _gen_nnmf_filters(self,weights):
		return nn.Sequential(*[NNMF_Block(torch.FloatTensor(w)) for w in weights])

	def forward(self,x):
		return self.nnmf(x)

def get_initial_filters(input_channel,patch_size):
	filters = np.zeros((input_channel * patch_size * patch_size, input_channel, patch_size, patch_size))
	for i in range(patch_size):
		for j in range(patch_size):
			for k in range(input_channel):
				filters[i * patch_size * input_channel + j * input_channel + k, k, i, j] = 1.
	return filters

def load_nmf_W(fname):
	mat = sio.loadmat(fname)
	w = mat['weights']
	filters = []
	for i in range(len(w[0])):
		curr_w = w[0][i]
		print(curr_w.shape)
		curr_w = np.transpose(curr_w,(0,3,1,2))
		filters.append(curr_w)
	return filters

if __name__ == '__main__':
	from torchvision import datasets, transforms
	w=load_nmf_W('weights_v1.mat')
	print(w[0].shape)
	f=get_initial_filters(1,3)
	print(f.shape)
	kwargs = {'num_workers': 1, 'pin_memory': True} if False else {}
	train_loader = torch.utils.data.DataLoader(
		datasets.MNIST('./data', train = True, download = True, transform = transforms.Compose([
			transforms.ToTensor()])),
		batch_size = 1, shuffle = False, **kwargs)
	for batch_idx, (data, target) in enumerate(train_loader):
		print(np.max(data.numpy()),np.min(data.numpy()))
		#plt.imshow(np.reshape(data.numpy(),(28,28)))
		#plt.show()
		break
	s = sio.loadmat('sample.mat')
	s = s['sample']
	s = s.astype(float)
	print(s.dtype,np.max(s))
	s = torch.FloatTensor(np.reshape(s,(1,1,28,28)))
	print(s.shape)
	#plt.imshow(np.reshape(s,(28,28)))
	#plt.show()
	w0 = get_initial_filters(1,3)
	h0 = F.conv2d(Variable(s),Variable(torch.FloatTensor(w0)))
	ws = load_nmf_W('weights_v1.mat')
	print('max',[np.max(ww) for ww in ws])
	w1 = ws[0]
	#w1 = load_nmf_W('inv_w1_v1.mat','inv_W1')
	h1 = F.conv2d(h0,Variable(torch.FloatTensor(w1)))
	print(h0[0,7,8,12])
	print(h0.size())
	print(h1.size())
	print(h1[0,0,19,12])
	#plt.imshow(np.reshape(h1.data.numpy()[0,1,:,:],(24,24)))
	#plt.show()
	nc = NNMF_CNN([w0,w1])
	rr = nc(Variable(s))
	print(rr.size(),np.max(rr.data.numpy()))
	#plt.imshow(np.reshape(rr.data.numpy()[0,1,:,:],(24,24)))
	#plt.show()

	nn_test = NNMF_CNN([w0]+ws)
	rt = nn_test(Variable(s))
	print(rt.size())