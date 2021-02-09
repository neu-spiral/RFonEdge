import random
import pickle
import math
import os
import numpy as np
import torch

from evaluate_model import compute_accuracy
from torch.utils.data import DataLoader

def get_model(model_flag, params={}):
    if model_flag.lower() == 'baseline':
        from Models.BaselineModel import getBaselineModel
        return getBaselineModel(
            slice_size = params['slice_size'],
            classes = params['classes'],
            cnn_stacks = params['cnn_stacks'],
            fc_stacks = params['fc_stacks'],
            channels = params['channels'],
            dropout_flag = params['dropout_flag'],
            fc1 = params['fc1'],
            fc2 = params['fc2'],
            batchnorm=params['batchnorm'],
            weights = params['pre_weight']
            )
    elif model_flag.lower() == 'baseline_2d':
        from Models.BaselineModel2D import getBaselineModel2D
        return getBaselineModel2D(
            slice_size = params['slice_size'],
            classes = params['classes'],
            cnn_stacks = params['cnn_stacks'],
            fc_stacks = params['fc_stacks'],
            channels = params['channels'],
            dropout_flag = params['dropout_flag'],
            fc1 = params['fc1'],
            fc2 = params['fc2'],
            batchnorm=params['batchnorm']
            )
    elif model_flag.lower() == 'vgg16':
        from Models.VGG16 import VGG16
        return VGG16(
            input_shape = (params['slice_size'], params['slice_size'], 3),
            output_shape = params['classes'],
            weights = params['pre_weight']
            )
    elif model_flag.lower() == 'resnet50':
        from Models.ResNetTF import ResNetTF
        return ResNetTF(
            input_shape = (params['slice_size'], params['slice_size'], 3),
            output_shape = params['classes'],
            weights = params['pre_weight']
            )
    elif model_flag.lower() == 'resnet1d':
        from Models.ResNet1D import ResNet1D
        return ResNet1D(
            input_shape = (params['slice_size'], 2),
            output_shape = params['classes']
            )
    elif model_flag.lower() == 'firlayer':
        from Models.BaselineModelFIR import getBaselineModel
        return getBaselineModel(
            slice_size = params['slice_size'],
            classes = params['classes'],
            cnn_stacks = params['cnn_stacks'],
            fc_stacks = params['fc_stacks'],
            channels = params['channels'],
            dropout_flag = params['dropout_flag'],
            fc1 = params['fc1'],
            fc2 = params['fc2'],
            batchnorm=params['batchnorm'],
            fir_size = params['fir_size']
            )
    elif model_flag.lower() == 'firlayer_resnet1d':
        from Models.ResNet1DFIR import ResNet1DFIR
        return ResNet1DFIR(
            input_shape = (params['slice_size'], 2),
            output_shape = params['classes'],
            fir_size = params['fir_size']
            )
    elif model_flag.lower() == 'baseline_remove_channel':
        from Models.BaselineModelRemoveChannel import get_model
        return get_model(
            slice_size = params['slice_size'],
            classes = params['classes'],
            cnn_stacks = params['cnn_stacks'],
            fc_stacks = params['fc_stacks'],
            channels = params['channels'],
            dropout_flag = params['dropout_flag'],
            fc1 = params['fc1'],
            fc2 = params['fc2'],
            batchnorm=params['batchnorm'],
            weights = params['pre_weight'],
            fir_size = params['fir_size'],
            use_preamble = params['use_preamble'],
            merge_preamble = params['merge_preamble']
            )
    elif model_flag.lower() == 'deepsigcnn':
        from Models.DeepSigCNN import getDeepSigCNNModel
        return getDeepSigCNNModel(
            slice_size = params['slice_size'],
            classes = params['classes'])

class TrainValTest():
    def __init__(self, base_path,save_path,val_from_train = False):

        self.base_path = base_path
        self.save_path = save_path
        self.val_from_train = val_from_train
        print(base_path)
        print(save_path)


    def load_data(self, sampling):
        with open(os.path.join(self.base_path, "label.pkl"),'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            self.labels = u.load()

        with open(os.path.join(self.base_path, "device_ids.pkl"), 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            self.device_ids = u.load()
        
        with open(os.path.join(self.base_path, "stats.pkl"), 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            self.stats = u.load()
        
        with open(os.path.join(self.base_path, "partition.pkl"), 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            self.partition = u.load()
        
        self.ex_list  = self.partition['train']
        
        if 'val' in self.partition:
            self.val_list = self.partition['val']
        else:
            if self.val_from_train:
                random.shuffle(self.ex_list)
                self.val_list = self.ex_list[int(0.9*len(self.ex_list)):]
                self.ex_list = self.ex_list[0:int(0.9*len(self.ex_list))]
            else:
                self.val_list = self.partition['test']
        self.test_list = self.partition['test']

        print("# of training exp:%d, validation exp:%d, testing exp:%d" % (len(self.ex_list), len(self.val_list), len(self.test_list)))
        
        # add for calculating balanced sampling
        if sampling.lower() == 'balanced':
            file = open(os.path.join(self.stats_path, "ex_per_device.pkl"), 'r')
            ex_per_device = pickle.load(file)
            self.ex_per_device = ex_per_device
            file.close()

        # we get the rep_time_per_device and pass it to new generator

            max_num_ex_per_dev = max(ex_per_device.values())
            self.rep_time_per_device = {dev: math.floor(max_num_ex_per_dev / num) if math.floor(max_num_ex_per_dev / num) <= 2000 else 2000 for dev,num in ex_per_device.items()}
        else:
            self.rep_time_per_device = {dev:1 for dev in self.device_ids.keys()}

    def GenerateData(self, batch_size, slice_size, K, files_per_IO, 
        generator_type='new', processor_type='no', shrink = 1.0, training_strategy='big', 
        file_type='mat', normalize=False, decimated=False, 
        add_padding=False, padding_type='zero', try_concat=False, crop=0, use_preamble=False, 
        aug_var=0.0434, aug_mean=0.045, aug_taps=11, conv2d=False, aug_granularity=None):

        ex_list = self.ex_list[0:int(len(self.ex_list)*shrink)]
        val_list = self.val_list[0:int(len(self.val_list)*shrink)]
        labels = self.labels
        device_ids = self.device_ids
        stats = self.stats

        corr_fact = 1
        if decimated:
            corr_fact = 10

        # TODO: Get different kinds of generator, including preprocessing ones
        generator_type = generator_type.lower()
        if generator_type == 'new':
            #from DataGenerators.DataGenerator import IQPreprocessDataGenerator
            import DataGenerator as DG

        if training_strategy == 'small':
            rep_time_per_device = self.rep_time_per_device
        else:
            rep_time_per_device = None

        processor_type = processor_type.lower()
        if processor_type == 'no':
            processor = None
        elif processor_type == 'tensor':
            processor = DG.IQTensorPreprocessor()
        elif processor_type =='fft':
            processor = DG.IQFFTPreprocessor()
        elif processor_type =='add_axis':
            processor = DG.AddAxisPreprocessor()
        elif processor_type == 'fir':
            processor = DG.IQFIRPreprocessor(fir_type='gaussian', test_mode=False, gaussian_filter=None, aug_var=aug_var, aug_mean=aug_mean, aug_taps=aug_taps)

        # len(device_ids)
        
        data_mean = None
        data_std = None
        if 'mean' in stats and 'std' in stats:
            data_mean = stats['mean']
            data_std = stats['std']

        save_path = self.save_path
        #if per example augmenation:
        print("aug_granularity is: " +str(aug_granularity))

        DataParams = {'batch_size': batch_size, 'shuffle': True, 'num_workers':0}
        training_set = DG.IQPreprocessDataGenerator(ex_list, False, labels,
            device_ids, stats['avg_samples'] * len(ex_list) / corr_fact, processor,
            len(device_ids), files_per_IO=files_per_IO, batch_size=batch_size,slice_size=slice_size, K=K,
            normalize=normalize, mean_val=data_mean,
            std_val=data_std,  rep_time_per_device = rep_time_per_device,
            file_type=file_type, add_padding=add_padding, padding_type=padding_type, 
            try_concat=try_concat, crop=crop, use_preamble=use_preamble, conv2d=conv2d,
            aug_granularity=aug_granularity, aug_taps=aug_taps, aug_mean=aug_mean, 
            aug_var=aug_var, save_path=save_path)
        
        train_generator = DataLoader(training_set, **DataParams)
        
        return train_generator


    def train_model(self, args, model, train_loader, criterion, optimizer, epoch):
        atch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        idx_loss_dict = {}
        
        model.train()
        for i, (input, target) in enumerate(train_loader):
            input=input.float()
            input = input.cuda()
            target = target.cuda()
            
            # compute output
            output = model(input)
            ce_loss = criterion(output, target)

            # measure accuracy and record loss
            acc1,_ = accuracy(output, target, topk=(1,5))
            losses.update(ce_loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            ce_loss.backward()
            optimizer.step()

            # print(i)
            if i % 1000 == 0:
                for param_group in optimizer.param_groups:
                    current_lr = param_group['lr']
                print('({0}) lr:[{1:.5f}]  '
                      'Epoch: [{2}][{3}/{4}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f}% ({top1.avg:.3f}%)\t'
                      .format('adam', current_lr,
                       epoch, i, len(train_loader), loss=losses, top1=top1))
            if i % 100 == 0:
                idx_loss_dict[i] = losses.avg
        return model
    
    def test_model(self, args, model):
        
        test_list = self.test_list[0:int(len(self.test_list)*args.shrink)]
        labels = self.labels
        device_ids = self.device_ids
        
        # here i just want to test if multigpu model works, so hardcode here
        data_mean = None
        data_std = None
        if 'mean' in self.stats and 'std' in self.stats:
            data_mean = self.stats['mean']
            data_std = self.stats['std']
        
        model.eval()
        acc_slice, acc_ex, preds = compute_accuracy(ex_list=test_list, labels=labels, device_ids=device_ids, slice_size=args.slice_size, model=model, batch_size = args.batch_size, vote_type='prob_sum', processor=args.preprocessor, test_stride=args.test_stride, file_type=args.file_type, normalize=args.normalize, mean_val=data_mean, std_val=data_std, add_padding=args.add_padding, padding_type=args.padding_type, crop=args.crop, use_preamble=args.use_preamble)
        
        # save predictions to the pickle file
        preds_pickle_file = args.save_path_exp + "/" + "preds.pkl"
        with open(preds_pickle_file,'wb') as fp:
            pickle.dump(preds,fp)
        
        return acc_slice, acc_ex, preds
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
