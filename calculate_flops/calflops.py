from __future__ import print_function
import torchvision.models as models
from collections import OrderedDict
import torch
import argparse
import os
import sys
import yaml

sys.path.append('/home/tong/RTML/cifar-admm-prune-pytorch/calculate_flops/')
from ptflops.flops_counter import get_model_complexity_info
from thop import profile


def calflops(model, inputs, config=''): 
    
    if not isinstance(config, str):
        raise Exception("filename must be a str")
        
    if config:
        config = os.path.join(os.path.dirname(__file__), config)
        print('config file: ', config)
        
        with open(config, "r") as stream:
            try:
                raw_dict = yaml.load(stream)
                prune_ratios = raw_dict['prune_ratios']

            except yaml.YAMLError as exc:
                print(exc)
    else:
        prune_ratios = OrderedDict()
        with torch.no_grad():
            for name, W in (model.named_parameters()):
                prune_ratios[name] = 0
    
            
    model.train(False)
    model.eval()
    macs, params = profile(model, inputs=(inputs, ), rate = prune_ratios)
    # flops, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs * 2/1000000000)) # GMACs
    print('{:<30}  {:<8}'.format('Number of parameters: ', params/1000000)) # M