from __future__ import print_function
import os
import sys
import logging
import argparse
import time
from time import strftime
import torch
import torch.optim as optim
from torchvision import datasets, transforms

from models.resnet_1d import ResNet18_1d, ResNet34_1d, ResNet50_1d

import admm
from admm import GradualWarmupScheduler
from admm import CrossEntropyLossMaybeSmooth
from admm import mixup_data, mixup_criterion
from testers import *

from TrainValTest import TrainValTest

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def check_and_create(dir_path):
    if os.path.exists(dir_path):
        return True
    else:
        os.makedirs(dir_path)
        return False
    
# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 admm training')
parser.add_argument('--logger', action='store_true', default=True,
                    help='whether to use logger')
parser.add_argument('--arch', type=str, default=None,
                    help='[vgg, resnet, convnet, alexnet]')
parser.add_argument('--depth', default=None, type=int,
                    help='depth of the neural network, 16,19 for vgg; 18, 50 for resnet')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--multi-gpu', action='store_true', default=False,
                    help='for multi-gpu training')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--admm-epochs', type=int, default=1, metavar='N',
                    help='number of interval epochs to update admm (default: 1)')
parser.add_argument('--optmzr', type=str, default='sgd', metavar='OPTMZR',
                    help='optimizer used (default: adam)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--lr-decay', type=int, default=30, metavar='LR_decay',
                    help='how many every epoch before lr drop (default: 30)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--load-model', type=str, default="",
                    help='For loading the exist Model')
parser.add_argument('--load-mask', type=str, default="",
                    help='For loading the exist Model')
parser.add_argument('--save-model', type=str, default="",
                    help='For Saving the current Model')
parser.add_argument('--masked-retrain', action='store_true', default=False,
                    help='for masked retrain')
parser.add_argument('--verbose', action='store_true', default=True,
                    help='whether to report admm convergence condition')
parser.add_argument('--admm', action='store_true', default=False,
                    help="for admm training")
parser.add_argument('--rho', type=float, default = 0.0001,
                    help ="define rho for ADMM")
parser.add_argument('--rho-num', type=int, default = 5,
                    help ="define how many rohs for ADMM training")
parser.add_argument('--sparsity-type', type=str, default='random-pattern',
                    help ="define sparsity_type: [irregular,column,filter,pattern,random-pattern]")
parser.add_argument('--config-file', type=str, default='config_vgg16',
                    help ="config file name")
parser.add_argument('--combine-progressive', default=False, type=str2bool,
                    help="for filter pruning after column pruning")
parser.add_argument('--purification', default=False, type=str2bool,
                    help="purification after pruning")

parser.add_argument('--lr-scheduler', type=str, default='default',
                    help='define lr scheduler')
parser.add_argument('--warmup', action='store_true', default=False,
                    help='warm-up scheduler')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='M',
                    help='warmup-lr, smaller than original lr')
parser.add_argument('--warmup-epochs', type=int, default=0, metavar='M',
                    help='number of epochs for lr warmup')
parser.add_argument('--mixup', action='store_true', default=False,
                    help='ce mixup')
parser.add_argument('--alpha', type=float, default=0.0, metavar='M',
                    help='for mixup training, lambda = Beta(alpha, alpha) distribution. Set to 0.0 to disable')
parser.add_argument('--smooth', action='store_true', default=False,
                    help='lable smooth')
parser.add_argument('--smooth-eps', type=float, default=0.0, metavar='M',
                    help='smoothing rate [0.0, 1.0], set to 0.0 to disable')
parser.add_argument('--no-tricks', action='store_true', default=False,
                    help='disable all training tricks and restore original classic training process')

########### From RFMLS: multi-gpu; batch-size
parser.add_argument('--exp_name', default='exp1', type=str, help='Specify the experiment name')
parser.add_argument('--base_path', default='/scratch/RFMLS/dataset100/dataset_with_val_9000train/', type=str, help='Specify the base path')
parser.add_argument('--save_path', default='/scratch/zhou.fan1/filtered/', type=str, help='Specify the save path')
parser.add_argument('--file_type', default='mat', type=str, help='Specify type of file you want to read')
parser.add_argument('--decimated', default=False, type=str2bool, help='Specify if the data in the files is decimated, if so and you are using the same stats file as the undecimated then the generator will take this into account')
parser.add_argument('--val_from_train', default=False, type=str2bool, help='If validation not present in partition file, generate one from the training set. (If false, use test set as validation)')
parser.add_argument('--test_on_val_set',default=False, type=str2bool, help='If true it will test the trained model on validation data, for tuning hyperparameters')
parser.add_argument('--train', default=False, type=str2bool, help='Specify doing training or not')
    
parser.add_argument('-ss', '--slice_size', default=1024, type=int, help='Specify the slice size')
parser.add_argument('-d', '--devices', default=100, type=int, help='Specify the number of total devices')
parser.add_argument('--cnn_stack', default=3, type=int, help='[Baseline Model] Specify the number of cnn layers')
parser.add_argument('--fc_stack', default=2, type=int, help='[Baseline Model] Specify the number of fc layers')
parser.add_argument('--channels', default=128, type=int, help='[Baseline Model] Specify the number of channels of cnn')
parser.add_argument('--fc1', default=256, type=int, help='[Baseline Model] Specify the number of neurons in the first fc layer')
parser.add_argument('--fc2', default=128, type=int, help='[Baseline Model] Specify the number of neurons in the penultimate fc layer')

# Data Generator    
parser.add_argument('--generator', default='new', type=str, help='Specify which generator to use')
parser.add_argument('--add_padding', default=False, type=str2bool, help='If examples are smaller than slice size addpadding')
parser.add_argument('--padding_type', default='zero', type=str, help='"zero"-padding and "stride"-padding')
parser.add_argument('--try_concat', default=False, type=str2bool, help='If examples are smaller than slice size and using demodulated data, try and concat them')
parser.add_argument('--preprocessor', default='no', type=str, help='Specify which preprocessor to use')
parser.add_argument('--K', default=1, type=int, help='Specify the batch down sampling factor K')
parser.add_argument('-fpio', '--files_per_IO', default=500000, type = int, help='Specify the files loaded to memory per IO')
parser.add_argument('--shrink', default=1, type=float, help='Dataset down sampling factor')
parser.add_argument('--normalize', default='True', type=str2bool, help='Specify if you want to normalize the data using mean and std in stats files (if stats does not have this info, it is ignored)')
parser.add_argument('--crop', default=0, type=int, help='if crop > 0 the generator crops the examples to a maximum length of crop')
parser.add_argument('--training_strategy', default='big', type=str, help='Specify which sampling strategy to use')
parser.add_argument('--sampling', default='model', type=str, help='Specify which sampling strategy to use')
parser.add_argument('--fir_size', default=11, type=int, help='FIR filter size.')
parser.add_argument('--use_preamble', default=False, type=str2bool, help='Using preamble to train channel-removing filter')
parser.add_argument('--merge_preamble', default=False, type=str2bool, help='Merge preamble with slice to train channel-removing filter')

parser.add_argument('--id_gpu', default=0, type=int, help='If --multigpu=False, this arguments specify which gpu to use.')
parser.add_argument('--test_stride', default=16, type=int, help='Specify the stride to use for testing')
parser.add_argument('--per_example_strategy', default='prob_sum', type=str, help='Specify the strategy used to compute the per wxample accuracy: (majority, prob_sum, log_prob_sum, all)')

################## augmentation parameters #####################
parser.add_argument('--aug_var', default='0.0434', type=float, help='variance of noise for data augmentation')
parser.add_argument('--aug_mean', default='0.045', type=float, help='mean of noise for data augmentation')
parser.add_argument('--aug_taps', default=11, type=int, help='Number of complex taps for data augmentation')
parser.add_argument('--aug_granularity', default='per_ex', type=str, help='granularity of fir selection for training pipelinecan be per_ex, per_batch, per_slice')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.manual_seed(args.seed)
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
kwargs = {'num_workers': args.workers, 'pin_memory': True} if args.cuda else {}
writer = None
print('Use Cuda:',use_cuda)
# ------------------ save path ----------------------------------------------
args.save_path_exp = args.save_path
check_and_create(args.save_path_exp)
setting_file = os.path.join(args.save_path_exp, args.exp_name+'.config')

print("*************** Configuration ***************")
with open(setting_file, 'w') as f:
    args_dic = vars(args)
    for arg, value in args_dic.items():
        line = arg + ' : ' + str(value)
        print(line)
        f.write(line+'\n')

### Data Loader ###
pipeline = TrainValTest(base_path=args.base_path, save_path=args.save_path_exp,
                        val_from_train=args.val_from_train)
pipeline.load_data(sampling=args.sampling)
train_loader = pipeline.GenerateData(args.batch_size, args.slice_size, args.K, args.files_per_IO, 
                                   generator_type=args.generator, processor_type=args.preprocessor, 
                                   training_strategy = args.training_strategy,
                                   file_type=args.file_type, normalize=args.normalize, 
                                   decimated=args.decimated, add_padding=args.add_padding,
                                   padding_type=args.padding_type, try_concat=args.try_concat,
                                   crop=args.crop,
                                   use_preamble=args.use_preamble, aug_var=args.aug_var,
                                   aug_mean=args.aug_mean, aug_taps=args.aug_taps)
# set up model archetecture
if args.arch == "resnet":
    if args.depth == 50:
        model = ResNet50_1d(args.slice_size,args.devices)
    if args.depth == 34:
        model = ResNet34_1d(args.slice_size,args.devices)
    if args.depth == 18:
        model = ResNet18_1d(args.slice_size,args.devices)
    print(model)
    
if args.multi_gpu:
    model = torch.nn.DataParallel(model)
model.cuda()
if args.load_model:
    original_model_name = args.load_model
    print("\n>_ Loading baseline/progressive model..... {}\n".format(original_model_name))
    model.load_state_dict(torch.load(original_model_name))
    
if args.train:
    print('*************** Training Model ***************')
    test_column_sparsity(model)
    optimizer_init_lr = 0.0001
    best_acc = 0
    optimizer = torch.optim.Adam(model.parameters(), optimizer_init_lr)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(1, 20):
        start = time.time()
        
        #adjust learning rate
        lr = optimizer_init_lr * (0.5 ** (epoch // 3))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        model = pipeline.train_model(args, model, train_loader, criterion, optimizer, epoch)
        end_train = time.time()
        acc_slice, acc_ex, preds = pipeline.test_model(args, model)
        end_test = time.time()
        print("Training time: {:.3f}; Testing time: {:.3f}; Testing slice-acc: {:.3f}; Testing exp-acc: {:.3f}".format(end_train-start, end_test-end_train, acc_slice, acc_ex))

        if acc_ex > best_acc:
            best_acc = acc_ex
            print("Saving model...\n")
            torch.save(model.state_dict(), args.save_path_exp+"/{}{}.pt".format(
                args.arch, args.depth))

else:
    print('*************** Not Training Model ***************')
    acc_slice, acc_ex, preds = pipeline.test_model(args, model)
    print("Testing slice-acc: {:.4f}; Testing exp-acc: {:.4f}".format(acc_slice, acc_ex))
    test_column_sparsity(model)
    test_filter_sparsity(model)
    
""" disable all bag of tricks"""
if args.no_tricks:
    # disable all trick even if they are set to some value
    args.lr_scheduler = "default"
    args.warmup = False
    args.mixup = False
    args.smooth = False
    args.alpha = 0.0
    args.smooth_eps = 0.0


def main():
    if (args.admm and args.masked_retrain):
        raise ValueError("can't do both masked retrain and admm")
        
    elif (not args.admm) and (not args.masked_retrain) and args.purification:
        print("Model Purification")
        post_column_prune(model,0.04)
        post_filter_prune(model,0.23)
        #acc_slice, acc_ex, preds = pipeline.test_model(args,model)
        rate = test_filter_sparsity(model)
        
        print("Testing slice-acc: {:.3f}; Testing exp-acc: {:.3f}".format(acc_slice, acc_ex))
        torch.save(model.state_dict(), args.save_path_exp+"/prunned_{}_{}.pt".format(acc_ex, rate))
        sys.exit(1)
        
    
    print("The config arguments showed as below:")
    print(args)

    """ bag of tricks set-ups"""
    criterion = CrossEntropyLossMaybeSmooth(smooth_eps=args.smooth_eps).cuda()
    args.smooth = args.smooth_eps > 0.0
    args.mixup = args.alpha > 0.0

    optimizer_init_lr = args.warmup_lr if args.warmup else args.lr

    optimizer = None
    if args.optmzr == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), optimizer_init_lr, momentum=0.9, weight_decay=1e-4)
    elif args.optmzr == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), optimizer_init_lr)

    scheduler = None
    if args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader), eta_min=4e-08)
    elif args.lr_scheduler == 'default':
        # my learning rate scheduler for cifar, following https://github.com/kuangliu/pytorch-cifar
        epoch_milestones = [65, 100, 130, 190, 220, 250, 280]

        """Set the learning rate of each parameter group to the initial lr decayed
            by gamma once the number of epoch reaches one of the milestones
        """
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[i * len(train_loader) for i in epoch_milestones], gamma=0.5)
    else:
        raise Exception("unknown lr scheduler")

    if args.warmup:
        scheduler = GradualWarmupScheduler(optimizer, multiplier=args.lr / args.warmup_lr, total_iter=args.warmup_epochs * len(train_loader), after_scheduler=scheduler)


    """====================="""
    """ multi-rho admm train"""
    """====================="""
    initial_rho = args.rho
    if args.admm:
        admm_prune(initial_rho, criterion, optimizer, scheduler)


    """=============="""
    """masked retrain"""
    """=============="""
    if args.masked_retrain:
        masked_retrain(initial_rho, criterion, optimizer, scheduler)
    
    


def admm_prune(initial_rho, criterion, optimizer, scheduler):
    for i in range(args.rho_num):
        current_rho = initial_rho * 10 ** i
        if i == 0:
            original_model_name = args.load_model
            print("\n>_ Loading baseline/progressive model..... {}\n".format(original_model_name))
            model.load_state_dict(torch.load(original_model_name))  # admm train need basline model
        else:
            model.load_state_dict(torch.load(args.save_path_exp+"/prunned_{}{}_{}_{}_{}_{}.pt".format(
                args.arch, args.depth, current_rho / 10, args.config_file, args.optmzr, args.sparsity_type)))
        model.cuda()

        ADMM = admm.ADMM(model, file_name="./profile/" + args.config_file + ".yaml", rho=current_rho)
        admm.admm_initialization(args, ADMM=ADMM, model=model)  # intialize Z variable

        # admm train
        best_prec1 = 0.

        for epoch in range(1, args.epochs + 1):
            print("current rho: {}".format(current_rho))
            train(ADMM, train_loader, criterion, optimizer, scheduler, epoch, args)
            #t_loss, prec1 = test(model, criterion, test_loader)
            acc_slice, acc_ex, preds = pipeline.test_model(args,model)
            print("Testing slice-acc: {:.3f}; Testing exp-acc: {:.3f}".format(acc_slice, acc_ex))
            
            best_prec1 = max(acc_ex, best_prec1)

        print("Best Acc: {:.4f}%".format(best_prec1))
        print("Saving model...\n")
        torch.save(model.state_dict(), args.save_path_exp+"/prunned_{}{}_{}_{}_{}_{}.pt".format(
            args.arch, args.depth, current_rho, args.config_file, args.optmzr, args.sparsity_type))



def masked_retrain(initial_rho, criterion, optimizer, scheduler):
    
    if args.load_mask:
        '''
        Load pre-mask and added to the full model
        '''
        print("\n>_ Loading Mask: "+ args.load_mask)
        mask = torch.load(args.load_mask)
        for name, W in (model.named_parameters()):
            if name in mask and W.shape==mask[name].shape:
                weight = mask[name].cpu().detach().numpy()
                non_zeros = weight != 0
                non_zeros = non_zeros.astype(np.float32)
                zero_mask = torch.from_numpy(non_zeros).cuda()
                W.data *= zero_mask
        test_column_sparsity(model)
     
    else:
        print("\n>_ Loading file: "+args.save_path_exp+"/prunned_{}{}_{}_{}_{}_{}.pt".format(
            args.arch, args.depth, initial_rho * 10 ** (args.rho_num - 1), args.config_file, args.optmzr,
            args.sparsity_type))
        model.load_state_dict(torch.load(args.save_path_exp+"/prunned_{}{}_{}_{}_{}_{}.pt".format(
            args.arch, args.depth, initial_rho * 10 ** (args.rho_num - 1), args.config_file, args.optmzr,
            args.sparsity_type)))
    model.cuda()
        
    ADMM = admm.ADMM(model, file_name="./profile/" + args.config_file + ".yaml", rho=initial_rho)
    print(ADMM.prune_ratios)
    best_prec1 = [0]
    admm.hard_prune(args, ADMM, model)
    epoch_loss_dict = {}
    testAcc = []

    for epoch in range(1, args.epochs + 1):
        idx_loss_dict = train(ADMM, train_loader, criterion, optimizer, scheduler, epoch, args)
        acc_slice, prec1, preds = pipeline.test_model(args,model)
        print("Testing slice-acc: {:.4f}; Testing exp-acc: {:.4f}".format(acc_slice, prec1))
        #rate = test_filter_sparsity(model)
        #t_loss, prec1 = test(model, criterion, test_loader)
        
        if prec1 > max(best_prec1):
            print("\n>_ Got better accuracy, saving model with accuracy {:.3f}% now...\n".format(prec1))
            torch.save(model.state_dict(), args.save_path_exp+"/retrained_{}{}_acc_{:.3f}_{}rhos_{}_{}.pt".format(
                args.arch, args.depth, prec1, args.rho_num, args.config_file, args.sparsity_type))
            print("\n>_ Deleting previous model file with accuracy {:.3f}% now...\n".format(max(best_prec1)))
            #if len(best_prec1) > 1:
            #    os.remove(args.save_path_exp+"/retrained_{}{}_acc_{:.3f}_{}rhos_{}_{}.pt".format(
            #        args.arch, args.depth, max(best_prec1), args.rho_num, args.config_file, args.sparsity_type))

        epoch_loss_dict[epoch] = idx_loss_dict
        testAcc.append(prec1)

        best_prec1.append(prec1)
        print("current best acc is: {:.4f}".format(max(best_prec1)))

    rate = test_column_sparsity(model)
    rate = test_filter_sparsity(model)

    print("Best Acc: {:.4f}%".format(max(best_prec1)))
    #np.save(strftime("./plotable/%m-%d-%Y-%H:%M_plotable_{}.npy".format(args.sparsity_type)), epoch_loss_dict)
    #np.save(strftime("./plotable/%m-%d-%Y-%H:%M_testAcc_{}.npy".format(args.sparsity_type)), testAcc)


def train(ADMM, train_loader,criterion, optimizer, scheduler, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    idx_loss_dict = {}

    # switch to train mode
    model.train()

    if args.masked_retrain and not args.combine_progressive:
        print("full acc re-train masking")
        masks = {}
        for name, W in (model.named_parameters()):
            # if name not in ADMM.prune_ratios:
            #     continue
            # above_threshold, W = admm.weight_pruning(args, W, ADMM.prune_ratios[name])
            # W.data = W
            # masks[name] = above_threshold
            weight = W.cpu().detach().numpy()
            non_zeros = weight != 0
            non_zeros = non_zeros.astype(np.float32)
            zero_mask = torch.from_numpy(non_zeros).cuda()
            W = torch.from_numpy(weight).cuda()
            W.data = W
            masks[name] = zero_mask
    elif args.combine_progressive:
        print("progressive admm-train/re-train masking")
        masks = {}
        for name, W in (model.named_parameters()):
            weight = W.cpu().detach().numpy()
            non_zeros = weight != 0
            non_zeros = non_zeros.astype(np.float32)
            zero_mask = torch.from_numpy(non_zeros).cuda()
            W = torch.from_numpy(weight).cuda()
            W.data = W
            masks[name] = zero_mask

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate
        if args.admm:
            admm.admm_adjust_learning_rate(optimizer, epoch, args)
        else:
            scheduler.step()

        input=input.float()
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        if args.mixup:
            input, target_a, target_b, lam = mixup_data(input, target, args.alpha)

        # compute output
        output = model(input)

        if args.mixup:
            ce_loss = mixup_criterion(criterion, output, target_a, target_b, lam, args.smooth)
        else:
            ce_loss = criterion(output, target, smooth=args.smooth)

        if args.admm:
            admm.z_u_update(args, ADMM, model, device, train_loader, optimizer, epoch, input, i, writer)  # update Z and U variables
            ce_loss, admm_loss, mixed_loss = admm.append_admm_loss(args, ADMM, model, ce_loss)  # append admm losss

        # measure accuracy and record loss
        acc1,_ = accuracy(output, target, topk=(1,5))

        losses.update(ce_loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()

        if args.admm:
            mixed_loss.backward()
        else:
            ce_loss.backward()

        if args.combine_progressive:
            with torch.no_grad():
                for name, W in (model.named_parameters()):
                    if name in masks:
                        W.grad *= masks[name]
        if args.masked_retrain:
            with torch.no_grad():
                for name, W in (model.named_parameters()):
                    if name in masks:
                        W.grad *= masks[name]

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print(i)
        if i % args.log_interval == 0:
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']
            print('({0}) lr:[{1:.5f}]  '
                  'Epoch: [{2}][{3}/{4}]\t'
                  'Status: admm-[{5}] retrain-[{6}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f}% ({top1.avg:.3f}%)\t'
                  .format(args.optmzr, current_lr,
                   epoch, i, len(train_loader), args.admm, args.masked_retrain, batch_time=data_time, loss=losses, top1=top1))
        if i % 100 == 0:
            idx_loss_dict[i] = losses.avg
    return idx_loss_dict



def test(model, criterion, test_loader):
    model.eval()
    losses = AverageMeter()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            losses.update(loss.item(), data.size(0))
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    print('\nTest set loss: {:.4f},  * Acc@1: {}/{} ({:.2f}%)\n'.format(
        losses.avg, correct, len(test_loader.dataset),
        100. * float(correct) / float(len(test_loader.dataset))))
    return losses.avg, (100. * float(correct) / float(len(test_loader.dataset)))

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

def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600*need_hour) / 60)
    need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
    return need_hour, need_mins, need_secs

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.3 ** (epoch // args.lr_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    start_time = time.time()
    main()
    duration = time.time() - start_time
    need_hour, need_mins, need_secs = convert_secs2time(duration)
    print('total runtime: {:02d}:{:02d}:{:02d}'.format(need_hour, need_mins, need_secs))
