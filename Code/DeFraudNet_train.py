###Network train code partially taken from https://github.com/andreasveit/densenet-pytorch/blob/master/train.py###
import argparse
import os
import shutil
import time
import numpy as np
from PIL import Image
import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import DeFraudNet_dataloader as data
import DeFraudNet_model as dn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
# used for logging to TensorBoard
from tensorboard_logger import configure, log_value

#Defining the GPU being used
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] ="1"

#Batchsize for dataloader is an argument
parser = argparse.ArgumentParser(description='PyTorch Patch Based DeFraudNet Training')

parser.add_argument('--epochs', default=400, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)') 
parser.add_argument('-b', '--batch-size', default=1, type=int, 
                    help='mini-batch size (default: 1)')
parser.add_argument('--lr', '--learning-rate', default=0.06, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=40, type=int,
                    help='total number of layers (default: 40)')
parser.add_argument('--growth', default=48, type=int,
                    help='number of new channels per layer (default: 12)')
parser.add_argument('--reduction_ratio', default=12, type=int,
                    help='Amount of reduction for Attention Network (default: 12)')
parser.add_argument('--droprate', default=0.2, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--reduce', default=0.5, type=float,
                    help='compression rate in transition stage (default: 0.5)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='To not use bottleneck block')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='Test_2011', type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)


###Initializing precision and ACE###
best_prec1 = 0
best_ace1 = 10

def main():
    global args, best_prec1,best_ace1,n
    args = parser.parse_args()
    if args.tensorboard:
        print("Tensorboard is True")
        configure("runs/%s"%(args.name))
    validation_ratio = 0.15

    # Data loading code
    ##Normalize data according to the requirement
    normalize = transforms.Normalize(mean=[x/255.0 for x in [114.3, 135.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 65.7]])

    if args.augment:
    	transform_train = transforms.Compose([
            	transforms.ToPILImage(),
            	transforms.Resize((200,200)),
            	transforms.RandomCrop(150, padding=4),
            	transforms.RandomAffine(20),
            	transforms.RandomHorizontalFlip(0.5),
            	transforms.ToTensor(),
            	normalize,
 		])
	
    	transform_train_p = transforms.Compose([
		transforms.ToPILImage(), # because the input dtype is numpy.ndarray
	    	transforms.Resize((50,50)),
	    	transforms.RandomCrop(32, padding=4),
            	transforms.RandomAffine(10),
            	transforms.RandomHorizontalFlip(0.5), # because this method is used for PIL Image dtype
            	transforms.ToTensor(),
 		])
    else:
    	transform_train = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((192,192)),
                transforms.RandomAffine(20),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                normalize,
                ])

    	transform_train_p = transforms.Compose([
                transforms.ToPILImage(), # because the input dtype is numpy.ndarray
                transforms.Resize((50,50)),
                transforms.RandomAffine(10),
                transforms.RandomHorizontalFlip(0.5), # because this method is used for PIL Image dtype
                transforms.ToTensor(),
                ])


    kwargs = {'num_workers': 4, 'pin_memory': True}
    train_dataset = data.fingerprint_data('Add path for train data ',transform = transform_train,patch_transform = transform_train_p)    
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(validation_ratio * num_train))
    random_seed = 2
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader= torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,sampler=train_sampler,**kwargs)
    print("Length of the Batchwise Training set",len(train_loader))

    val_loader= torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=valid_sampler,**kwargs)
    print("Length of the Batchwise Validation set",len(val_loader))


    # create model ##Changed the second parameter which is number of classes to 2
    num_classes = 2
    model = dn.Model(growth_rate=args.growth, num_layers=args.layers,theta =args.reduce,drop_rate=args.droprate,num_classes=2)
    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # for training on multiple GPUs. 
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use

    print("The complete Model is:",model)
    model = model.cuda()
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            best_ace1 = checkpoint['best_ace1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,momentum=args.momentum,nesterov=True,weight_decay=args.weight_decay)
    n = args.epochs
    for epoch in range(args.start_epoch, args.epochs):
        print("Started Training!!!!")
        adjust_learning_rate(optimizer, epoch,n)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1,ace1 = validate(val_loader, model, criterion, epoch)
        
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        is_best_ace = ace1 < best_ace1
        best_prec1 = max(prec1, best_prec1)
        best_ace1 = min(ace1, best_ace1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'best_ace1': best_ace1,
        }, is_best,is_best_ace)
        
    print('Best Accuracy on validation for Dataset is {} and best ACE is {}:'.format( best_prec1,best_ace1))

    
def train(train_loader, model, criterion, optimizer, epoch):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    print("Entered training mode!!!")
    # switch to train mode
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        patches_input = input[1]
        target_patches = target[1]
        for j in range(2,len(input)):
                patches_input = torch.cat((patches_input,input[j]),0)
                target_patches = torch.cat((target_patches,target[j]),0)
        target_var_1 = (target[0]).cuda()
        target_var_2 = target_patches.cuda()
        input_var_1 = (input[0]).cuda()
        input_var_2 = patches_input.cuda()
        with torch.no_grad():
                input_var_1 = torch.autograd.Variable(input_var_1)
                target_var_1 = torch.autograd.Variable(target_var_1)
                input_var_2 = torch.autograd.Variable(input_var_2)
                target_var_2 = torch.autograd.Variable(target_var_2)
        # compute output
        output = model(input_var_1,input_var_2)
        loss = criterion(output, target_var_1)
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target_var_1, topk=(1,))[0]
        ace1 = ACE(output.data, target_var_1, topk=(1,))[0]
        losses.update(loss.data, input_var_1.size(0))
        top1.update(prec1, input_var_1.size(0))
        top2.update(ace1,input_var_1.size(0))
	# compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'ACE@1 {top2.val:.3f} ({top2.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      loss=losses, top1=top1,top2=top2))
    # log to TensorBoard
    if args.tensorboard:
        log_value('train_loss', losses.avg, epoch)
        log_value('train_acc', top1.avg, epoch)
        log_value("train_ace",top2.avg,epoch)
    print("For Dataset, train loss = {} and Train accuracy = {} and Train ACE = {} for epoch {}".format(losses.avg,top1.avg,top2.avg,epoch))

print('Finished Training!!!')

def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    print("Entered Validation!!!")
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        patches_input = input[1]
        target_patches = target[1]
        for j in range(2,len(input)):
                patches_input = torch.cat((patches_input,input[j]),0)
                target_patches = torch.cat((target_patches,target[j]),0)
        target = target[0].cuda()
        input = input[0].cuda()
        target_patches = target_patches.cuda()
        patch_input = patches_input.cuda()
        with torch.no_grad():
                input_var_1 = torch.autograd.Variable(input)
                target_var_1 = torch.autograd.Variable(target)
                input_var_2 = torch.autograd.Variable(patch_input)
                target_var_2 = torch.autograd.Variable(target_patches)

        # compute output
        output = model(input_var_1,input_var_2)
        loss = criterion(output, target_var_1)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target_var_1, topk=(1,))[0]
        ace1 = ACE(output.data, target_var_1, topk=(1,))[0]
        losses.update(loss.data, input_var_1.size(0))
        top1.update(prec1, input_var_1.size(0))
        top2.update(ace1,input_var_1.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'ACE@1 {top2.val:.3f} ({top2.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1,top2=top2))
    ###Print the validation loss for the required dataset###
    print("For Dataset, Validation loss = {} and Validation accuracy = {} and Validation ACE = {} for epoch {}".format(losses.avg,top1.avg,top2.avg,epoch))
    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    print(' * ACE@1 {top2.avg:.3f}'.format(top2=top2))
    # log to TensorBoard
    if args.tensorboard:
        log_value('val_loss', losses.avg, epoch)
        log_value('val_acc', top1.avg, epoch)
        log_value('val_ace',top2.avg,epoch)
    return top1.avg,top2.avg



def save_checkpoint(state, is_best,is_best_ace, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/"%(args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(args.name) + 'model_best.pth.tar')
    if is_best_ace:
        shutil.copyfile(filename, 'runs/%s/'%(args.name) + 'model_best.pth.tar')

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


def adjust_learning_rate(optimizer, epoch,n):
    """Sets the learning rate to the initial LR decayed by 10 after 25% ,50% and 75% of  epochs"""
    lr = args.lr * (0.1 ** (epoch // (n*0.25))) * (0.1 ** (epoch // (n*0.5))) *(0.1 ** (epoch // (n*0.75)))
    #lr = args.lr * (0.1 ** (epoch // (n*0.5))) * (0.1 ** (epoch // (n*0.75)))
    print("The Changed Learning rate is = ",lr)
    # log to TensorBoard
    if args.tensorboard:
        log_value('learning_rate', lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

####Testing on the Datasets###

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0) 
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size)) 
    return res


def ACE(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0) 
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    incorrect = pred.ne(target.view(1, -1).expand_as(pred))
    ace = []	
    for k in topk:
        incorrect_k = incorrect[:k].view(-1).float().sum(0)
        ace.append((incorrect_k/2.0).mul_(100.0/batch_size))
    return ace

if __name__ == '__main__':
    main()


    
