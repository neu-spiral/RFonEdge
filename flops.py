from __future__ import print_function
import os
import sys
import torch
import argparse
from torchsummary import summary
from torch.autograd import Variable
from collections import OrderedDict

from models.resnet_1d import ResNet18_1d, ResNet34_1d, ResNet50_1d
from models.vgg_1d import VGG
from models.baseline_1d import Baseline
from testers import *
from calculate_flops.calflops import calflops

def remove_module(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            k = k.replace('module.', '')
        new_state_dict[k] = v
    return new_state_dict

# Training settings
parser = argparse.ArgumentParser(description='Load Models')

parser.add_argument('--load_model', type=str, default="", help='For loading the exist Model')
parser.add_argument('--slice_size', type=int, default=198, help='input size')
parser.add_argument('--devices', type=int, default=50, help='number of classes')
args = parser.parse_args()

args.config = 'profile/config_res50_full_v2.yaml'

'''
Use cuda
'''
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
writer = None

'''
set up model archetecture
'''
model = ResNet50_1d(args.slice_size,args.devices)
print(model)
input_np = np.random.uniform(0, 1, (1, 2, args.slice_size))
input_var = Variable(torch.FloatTensor(input_np), requires_grad=False)
output = model(input_var)
    
model.to(device)
summary(model, input_size=(2, 198))    

print("\n>_ Loading... {}\n".format(args.load_model))
model.load_state_dict(remove_module(torch.load(args.load_model)))
model.train(False)
model.eval()

'''
calculate flops
'''
rate = test_column_sparsity(model)
calflops(model, input_var, args.config)