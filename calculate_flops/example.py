from __future__ import print_function
import torchvision.models as models
import torch
from ptflops import get_model_complexity_info
import argparse
import os
import sys
from resnet_1d import ResNet50_1d
# import wdsr_b
# from option2 import parser
# from wdsr_b import *
# from args import *

parser = argparse.ArgumentParser(description='Load Models')
parser.add_argument('--slice_size', type=int, default=512, help='input size')
parser.add_argument('--devices', type=int, default=50, help='number of classes')


with torch.cuda.device(0):
	args = parser.parse_args()
	net = ResNet50_1d(args.slice_size,args.devices)
	# net = models.densenet161()
	flops, params = get_model_complexity_info(net, (2, 512), as_strings=True, print_per_layer_stat=True)
	# flops, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
	print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
	print('{:<30}  {:<8}'.format('Number of parameters: ', params))

# 144P(256×144) 240p(426×240) 360P(640×360) 480P(854×480)