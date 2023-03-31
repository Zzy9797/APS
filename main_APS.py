import argparse
from cProfile import label
from hashlib import new
import imp
import os
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from models.ResNet18 import res18feature
from models.PhaseAggregationLoss import PhaseAggregationLoss
from dataset.MyDataset import MyDataset
import datetime
import torch.nn.functional as F
import random
from torchsampler import ImbalancedDatasetSampler

now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H-%M]-")
data_path = './RAF'
checkpoint_path = ''

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default=data_path)
parser.add_argument('--checkpoint_path', type=str, default='./checkpoint/' + time_str )
parser.add_argument('--best_checkpoint_path', type=str, default='./checkpoint/'+time_str)
parser.add_argument('--log_path',type=str, default='./log/')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('-e', '--evaluate', default=False, action='store_true', help='evaluate model on test set')
parser.add_argument('--pretrained_backbone_path', type=str, default='./resnet18_msceleb.pth', help='pretrained_backbone_path')
parser.add_argument('--workers', type=int, default=4, help='number of workers')
parser.add_argument('--out_dimension', type=int, default=64, help='feature dimension')



parser.add_argument('--lr', default=0.0002, type=float)
parser.add_argument('--fc_lr',default=0.0001,type=float)
parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
parser.add_argument('--stepsize',type=int,default=5)
parser.add_argument('--epochs', type=int, default=80,help='number of epochs')
parser.add_argument('--alpha', default=0.5, type=float)
parser.add_argument('--lba',default=0.001,type=float)
parser.add_argument('--boundry', default=1, type=float)
parser.add_argument('--gpu', type=str, default='3')
parser.add_argument('--seed', type=int, default=6)
parser.add_argument('--dataset', type=str, default=data_path.split('/')[-1])
parser.add_argument('--class_number', type=int, default=7)

args = parser.parse_args()

def main():
    seed=args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark=False
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    best_acc = 0
    print('Training time: ' + now.strftime("%m-%d %H:%M"))

    
    # create model
    res18 = res18feature(args)
    fc_6 = nn.Linear(args.out_dimension, args.class_number-1)
    fc_2= nn.Linear(args.out_dimension, 1)
    res18.cuda()
    fc_6.cuda()
    fc_2.cuda()

    params = res18.parameters()
    params_fc_6 = fc_6.parameters()
    params_fc_2 = fc_2.parameters()
    criterion = nn.CrossEntropyLoss().cuda()
    criterion_l1=nn.L1Loss().cuda()
    criterion_PALoss = PhaseAggregationLoss(args).cuda()
    params_center = criterion_PALoss.parameters()
    criterion_l2=nn.MSELoss().cuda()
    optimizer = torch.optim.Adam([
        {'params': params}, 
        {'params': params_fc_6, 'lr': args.fc_lr},
        {'params': params_fc_2, 'lr': args.fc_lr},
        {'params': params_center, 'lr': args.fc_lr},
       ], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    recorder = RecorderMeter(args.epochs)

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'test')

    data_transforms = transforms.Compose([  
        transforms.ToTensor(),
        transforms.RandomResizedCrop((224, 224),scale=(0.8,1)),
        transforms.RandomGrayscale(p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.25))
    ])
    data_transforms_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    if args.evaluate:
        test_dataset = MyDataset(valdir,data_transforms_val,phase='train')
        val_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)
        validate(val_loader, res18,fc_6,fc_2, criterion_l2,center, args)
        return

    train_dataset = MyDataset(traindir,transform=data_transforms)

    test_dataset = MyDataset(valdir,data_transforms_val)


    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    
    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        current_learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
        print('Current learning rate: ', current_learning_rate)
        txt_name = args.log_path + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write('Current learning rate: ' + str(current_learning_rate) + '\n')

  
        train_los,center = train(train_loader, res18,fc_6,fc_2, criterion, criterion_l1,criterion_PALoss,optimizer, epoch, args)

        val_acc  = validate(val_loader, res18,fc_6,fc_2, criterion_l2,center, args)

        scheduler.step()
        wandb.log({
        "Test_Accuracy": val_acc,
         "Train_Loss": train_los,
        })
        recorder.update(epoch, train_los, 0, 0, val_acc)  
        curve_name = time_str + 'cnn.png'
        recorder.plot_curve(os.path.join(args.log_path, curve_name))

        # remember best acc and save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        print('Current best accuracy: ', best_acc)
        txt_name = args.log_path + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write('Current best accuracy: ' + str(best_acc) + '\n')

        save_checkpoint({'epoch': epoch + 1,
                         'model_state_dict': res18.state_dict(),
                         'fc_6_state_dict': fc_6.state_dict(),
                         'fc_2_state_dict':fc_2.state_dict(),
                         'center_state_dict':center,
                         'best_acc': best_acc,
                         'optimizer': optimizer.state_dict(),
                         'recorder': recorder}, is_best, args)
        end_time = time.time()
        epoch_time = end_time - start_time
        print("An Epoch Time: ", epoch_time)
        txt_name =args.log_path + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write(str(epoch_time) + '\n')


def train(train_loader, res18,fc_6,fc_2,criterion, criterion_l1,criterion_centor,optimizer, epoch, args):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(train_loader),
                             [losses, top1],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    res18.train()
    for i, (images, target) in enumerate(train_loader):
        images = images.cuda()
        target = target.cuda()
        target_2 = make_new_target_2(target).unsqueeze(1).cuda()
        output=res18(images)
        feature_6,target_6,double,list_6= make_input_target_6(output,target)
        output_2 = fc_2(output)
        target_center= torch.linspace(0, args.class_number-2,args.class_number-1).unsqueeze(0).transpose(0,1).squeeze().long().cuda()
        if double:
            feature_6 = feature_6.cuda()
            target_6 = target_6.cuda()
            output_6 = fc_6(feature_6)
            output_6 = out_attention(output_2,output_6,list_6)
            soft_output_6 = torch.softmax(output_6,dim=1)
            center_loss,center=criterion_centor(soft_output_6,target_6)
            
            loss=args.alpha*criterion_l1(output_2,target_2)+(1-args.alpha)*(criterion(output_6,target_6))+args.lba*center_loss
        else:
            loss=criterion_l1(output_2,target_2)

        losses.update(loss.item(), images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print loss and accuracy
        if i % args.print_freq == 0:
            progress.display(i)

    return losses.avg,center


def validate(val_loader, res18,fc_6,fc_2, criterion, center,args):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    top1_2 = AverageMeter('Accuracy_2', ':6.3f')
    progress = ProgressMeter(len(val_loader),
                             [losses, top1],
                             prefix='Test: ')

    # switch to evaluate mode
    res18.eval()
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()
            target_2 = make_new_target_2(target).cuda()
            
            output = res18(images)
            output_2 = fc_2(output)
            output_6 = fc_6(output)
            sigmoid=nn.Sigmoid()

            new_output_2=sigmoid(output_2)+args.boundry
            new_output_6=output_6/new_output_2
            new_output_6 = torch.softmax(new_output_6,dim=1)

            output6,target6=get_output6_target6(new_output_6,target)
            if i==0:
                output_save=output6
                target_save=target6
            else:
                output_save=torch.cat([output_save,output6],dim=0)
                target_save=torch.cat([target_save,target6],dim=0)
            

            acc,acc_2,pred = dis_accuracy(output_2, new_output_6,center,target,target_2,criterion)
            top1.update(acc.item(), images.size(0))
            top1_2.update(acc_2.item(), images.size(0))


            if i % args.print_freq == 0:
                progress.display(i)
            
        print(' **** Accuracy {top1.avg:.3f} *** '.format(top1=top1))
        print(' **** 二分类准确率 {top1_2.avg:.3f} *** '.format(top1_2=top1_2))
        with open(args.log_path + time_str + 'log.txt', 'a') as f:
            f.write(' * Accuracy {top1.avg:.3f}'.format(top1=top1) + '\n')
    return top1.avg

def get_output6_target6(output,target):
    list_6=[]
    for i,num in enumerate(target):
        if target[i]==torch.tensor(0).cuda():
            pass
        else:
            list_6.append(i)
    target6=torch.ones(len(list_6)).cuda()
    output6=torch.ones(len(list_6),output.size(1)).cuda()
    for i,num in enumerate(list_6):
        target6[i]=target[num]
        output6[i]=output[num]
    return output6,target6

def save_checkpoint(state, is_best, args):
    checkpoint_path=args.checkpoint_path+'model_for_'+args.dataset+'.pth'
    best_checkpoint_path=args.best_checkpoint_path+'model_for_'+args.dataset+'_best.pth'
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_checkpoint_path)

def make_new_target_2(labels,classes=7):
    new_target_2=torch.ones(labels.size()[0]).cuda()
    for i in range(labels.size()[0]):
        if labels[i]==torch.tensor(0).cuda():
            new_target_2[i]=0
        else:
            new_target_2[i]=1
    return new_target_2

def out_attention(input_2,input_6,target):
    sgd=nn.Sigmoid().cuda()
    new_input_2 = torch.zeros(input_6.size(0),1).cuda()
    for i,num in enumerate(target):
        new_input_2[i]=input_2[num]
    new_input_2=sgd(new_input_2)+args.boundry
    new_output_6 = torch.zeros(input_6.size()).cuda()
    
    for i,num in enumerate(target):
        new_output_6[i] = input_6[i]/new_input_2[i] 
    return new_output_6

def make_input_target_6(inputs,labels,classes=7):
    list_6=[]
    for i in range(inputs.size()[0]):
        if labels[i]==torch.tensor(0).cuda():
            pass
        else:
            list_6.append(i)
    if len(list_6)!=0:
        double=True
        new_input=inputs[list_6[0]].unsqueeze(0)
        new_target=torch.as_tensor([labels[list_6[0]]-1])
        for i,j in enumerate(list_6):
            if i==0:
                pass
            else:
                new_input=torch.cat((new_input,inputs[list_6[i]].unsqueeze(0)),dim=0)
                new_target=torch.cat((new_target,torch.as_tensor([labels[list_6[i]]-1])),dim=0)
    else:
        double=False
        new_input=None
        new_target=None
    return new_input,new_target,double,list_6



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        print(print_txt)
        txt_name = args.log_path + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def dis_accuracy(output_2,output_6, center,target,target_2,MSE):
    pred=torch.Tensor(output_2.size()[0]).cuda()
    pred_2=torch.Tensor(output_2.size()[0]).cuda()
    with torch.no_grad(): 
        batch_size = output_2.size(0)
        for b in range(batch_size):
            if output_2[b]<0.5:
                pred[b]=0   
                pred_2[b]=0
            else:
                pred_list=[]
                for i in range(args.class_number-1):
                    pred_list.append(MSE(output_6[b].unsqueeze(0),center[i].unsqueeze(0)))
                pred[b]=pred_list.index(min(pred_list))+1
                pred_2[b]=1
        correct=pred.eq(target).sum()
        correct_2=pred_2.eq(target_2).sum()
        acc=correct.float().mul_(100.0 / batch_size)
        acc_2=correct_2.float().mul_(100.0 / batch_size)
        return acc,acc_2,pred


class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""

    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)    # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 30
        self.epoch_losses[idx, 1] = val_loss * 30
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):

        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1800, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 5
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x30', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x30', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print('Saved figure')
        plt.close(fig)


if __name__ == '__main__':
    main()

