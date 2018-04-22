from time import gmtime,strftime
import os
import math
import numbers
import random
from PIL import Image, ImageOps
import numpy as np
import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import logging
import pickle
from nnmf_utils import NNMF_CNN,get_initial_filters,load_nmf_W
import scipy.io as sio
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N', 
					help='input batch size for training (default: 64)')
parser.add_argument('--no-cuda', action='store_true', default=False,
					help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
					help='random seed (default: 1)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
					help='learning rate (default: 0.01)')
parser.add_argument('--weight-decay', type=float, default=0., metavar='WD',
					help='weigth decay')
parser.add_argument('--rot-test', type=int, default=1, metavar='LDE',
					help='rotate images in test')
parser.add_argument('--lr-decay-epoch', type=int, default=10, metavar='LDE',
					help='number of epochs to decay lr')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
					help='number of epochs to train (default: 10)')
parser.add_argument('--use-method', type=float, default=1, metavar='UC',
					help='0:use complex convolution\n1:correlation')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
					help='how many batches to wait before logging training status')
parser.add_argument('--patch-size', default=5, type=int)
parser.add_argument('--rho-size', default=5, type=int)
parser.add_argument('--theta-size', default=20, type=int)
parser.add_argument('--debug', default=False, type=bool)
parser.add_argument('--test', default='', type=str)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
	torch.cuda.manual_seed(args.seed)

class RandomRotate(object):
    """Rotate the given PIL.Image counter clockwise around its centre by a random degree 
    (drawn uniformly) within angle_range. angle_range is a tuple (angle_min, angle_max). 
    Empty region will be padded with color specified in fill."""
    def __init__(self, angle_range=(-180,180), fill='black'):
        assert isinstance(angle_range, tuple) and len(angle_range) == 2 and angle_range[0] <= angle_range[1]
        assert isinstance(fill, numbers.Number) or isinstance(fill, str) or isinstance(fill, tuple)
        self.angle_range = angle_range
        self.fill = fill

    def __call__(self, img):
        angle_min, angle_max = self.angle_range
        angle = angle_min + random.random() * (angle_max - angle_min)
        theta = math.radians(angle)
        w, h = img.size
        diameter = math.sqrt(w * w + h * h)
        theta_0 = math.atan(float(h) / w)
        w_new = diameter * max(abs(math.cos(theta-theta_0)), abs(math.cos(theta+theta_0)))
        h_new = diameter * max(abs(math.sin(theta-theta_0)), abs(math.sin(theta+theta_0)))
        pad = math.ceil(max(w_new - w, h_new - h) / 2)
        img = ImageOps.expand(img, border=int(pad), fill=self.fill)
        img = img.rotate(angle, resample=Image.BICUBIC)
        return img.crop((pad, pad, w + pad, h + pad))

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
	datasets.MNIST('./data', train = True, download = True, transform = transforms.Compose([
		transforms.ToTensor(),transforms.Normalize((.1307,),(.3081,))])),
	batch_size = args.batch_size, shuffle = True, **kwargs)
test_loader = torch.utils.data.DataLoader(
		datasets.MNIST('./data', train = False, transform = transforms.Compose([
			transforms.ToTensor(),transforms.Normalize((.1307,),(.3081,))])),
		batch_size = args.batch_size, shuffle = True, **kwargs)
if args.rot_test:
	rot_loader = torch.utils.data.DataLoader(
		datasets.MNIST('./data', train = False, transform = transforms.Compose([
			RandomRotate((-180,180)),transforms.ToTensor(),transforms.Normalize((.1307,),(.3081,))])),
		batch_size = args.batch_size, shuffle = True, **kwargs)
class_num = 10
best_acc = 0.
def exp_lr_scheduler(optimizer, epoch, lr_decay=0.2, lr_decay_epoch=50):
	"""Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
	if epoch % lr_decay_epoch:
		return optimizer

	for param_group in optimizer.param_groups:
		param_group['lr'] *= lr_decay
	return optimizer


class MixedCNN(nn.Module):
	def __init__(self,weights=None):
		super(MixedCNN,self).__init__()
		if weights is None:
			w0 = get_initial_filters(1,3)
			ws = load_nmf_W('weights_v1.mat')
			weights = [w0]+ws
		self.nnmf_base = NNMF_CNN(weights)
		self.conv1 = nn.Conv2d(80,80,3)
		self.conv2 = nn.Conv2d(80,160,3)
		self.conv3 = nn.Conv2d(160,256,4)
		self.fco = nn.Linear(256,10)
	def forward(self,x):
		x = x.view(-1,1,28,28)
		x = x * 255
		x = self.nnmf_base(x)
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = self.conv3(x)
		x = x.view(-1,256)
		x = self.fco(x)
		return x

model = MixedCNN()
if args.cuda:
	model = nn.DataParallel(model,device_ids=[0,1])
	model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay = args.weight_decay)
def train(epoch,optimizer,max_iter = 20):
	model.train()
	counter = 0
	optimizer = exp_lr_scheduler(optimizer,epoch,lr_decay_epoch = args.lr_decay_epoch)
	for batch_idx, (data, target) in enumerate(train_loader):
		if args.cuda:
			data, target = data.cuda(), target.cuda()
		data, target = Variable(data), Variable(target)
		optimizer.zero_grad()
		output = model(data)
		pred = output.data.max(1)[1]
		correct = pred.eq(target.data).cpu().sum()
		loss = F.cross_entropy(output, target)
		loss.backward()
		optimizer.step()
		if batch_idx % args.log_interval == 0:
			print('Train Epoch: {} [{}/{} ({}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.data[0]))
			logging.info(strftime("%Y-%m-%d %H:%M:%S ",gmtime()))
			logging.info('Train Epoch: {} [{}/{} ({}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.data[0]))
		counter += 1
		if max_iter > 0 and counter >= max_iter:
			break
def test(epoch,test_loader,max_iter = 5,log_name = ''):
	global best_acc
	model.eval()
	test_loss = 0
	correct = 0
	rot_correct = 0
	counter = 0
	data_counter = 0
	res_mat = np.zeros((10,10))
	for data, target in test_loader:
		#target[target == 9] = 6
		if args.cuda:
			data, target = data.cuda(), target.cuda()
		data, target = Variable(data, volatile=True), Variable(target)
		output = model(data)
		test_loss += F.nll_loss(output,target).data[0]
		pred = output.data.max(1)[1]
		correct += pred.eq(target.data).cpu().sum()
		counter+=1
		data_counter += data.size()[0]
		for target_idx in range(int(target.size()[0])):
			#print(pred[target_idx].cpu().numpy()[])
			try:
				res_mat[(target[target_idx].cpu().data).numpy()[0],
						pred[target_idx].cpu().numpy()[0]] += 1
			except:
				res_mat[(target[target_idx].cpu().data).numpy()[0],
						pred[target_idx]] += 1
		if max_iter > 0 and counter >= max_iter: break
	if args.debug:
		with open('stats.fp','wb') as statout:
			pickle.dump(model.stats,statout)
	test_loss = test_loss
	test_loss /= len(test_loader)
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, (data_counter),
		100. * correct / (data_counter)))
	if log_name is not None:
		logging.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
			test_loss, correct, (data_counter),
			100. * correct / (data_counter)))
	print('Rot Acc:{:.4f}%'.format(100.*correct/data_counter))
	if log_name is not None:
		logging.info('Rot Acc:{:.4f}%'.format(100.*correct/data_counter))
	print('Predict Map:')
	if log_name is not None:
		logging.info('Predict Map:\n')
	for i in range(class_num):
		print(str(i)+'|'+','.join('{:4d}'.format(int(res_mat[i,x])) for x in range(class_num)))
		if log_name is not None:
			logging.info(str(i)+'|'+','.join('{:4d}'.format(int(res_mat[i,x])) for x in range(class_num)))
	acc = 100.*correct/data_counter
	if  acc > best_acc and log_name is not None:
		print(log_name)
		best_acc = acc
		torch.save(model.state_dict(), log_name + '.p7')
if __name__ == '__main__':
	log_name = 'log/' + 'NNMF' + '_' \
				+ strftime("%Y-%m-%d_%H-%M-%S",gmtime()) \
				+ '_batch_size_'+str(args.batch_size) \
				+ '_epochs_' + str(args.epochs) \
				+ '_lr_'+str(args.lr) \
				+ '_wd_'+str(args.weight_decay) \
			 	+ '_uc_'+str(args.use_method) \
				+ '_lde_'+str(args.lr_decay_epoch) \
				+ '_patch_'+str(args.patch_size) \
				+ '_rho_'+str(args.rho_size) \
				+ '_theta_'+str(args.theta_size)
	logging.basicConfig(filename=log_name,level=logging.DEBUG)
	for epoch in range(args.epochs):
		if len(args.test) > 0:
			model.load_state_dict(torch.load(args.test))
			para_dict=dict()
			para_dict[('conv1','real')] = model.cp_conv1.real_polar_weight.cpu().data.numpy()
			para_dict[('conv1','imag')] = model.cp_conv1.imag_polar_weight.cpu().data.numpy()
			para_dict[('conv2','real')] = model.cp_conv2.real_polar_weight.cpu().data.numpy()
			para_dict[('conv2','imag')] = model.cp_conv2.imag_polar_weight.cpu().data.numpy()
			para_dict[('conv3','real')] = model.cp_conv3.real_polar_weight.cpu().data.numpy()
			para_dict[('conv3','imag')] = model.cp_conv3.imag_polar_weight.cpu().data.numpy()
			with open('para.fp','wb') as fout:
				pickle.dump(para_dict,fout)
			if args.rot_test:
				test(0,rot_loader,-1,None)
			else:
				test(0,test_loader,-1,None)	
			break
		train(epoch,optimizer,-1)
		if args.rot_test:
			test(0,rot_loader,-1,log_name)
			test(epoch,test_loader,-1,None)
		else:
			test(0,rot_loader,-1,None)
			test(epoch,test_loader,-1,log_name)