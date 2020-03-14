from __future__ import print_function
import sys
if len(sys.argv) != 6:
    print('Usage:')
    print('python train.py datacfg cfgfile weightfile backupdir trainlist')
    exit()

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.autograd import Variable

import dataset
import random
import math
import os
from utils import *
from cfg import parse_cfg
from region_loss import RegionLoss
from darknet_defense import Darknet
# from models.tiny_yolo import TinyYoloNet

# Training settings
datacfg = sys.argv[1]
cfgfile = sys.argv[2]
weightfile = sys.argv[3]

data_options = read_data_cfg(datacfg)
net_options = parse_cfg(cfgfile)[0]

backupdir = sys.argv[4]
trainlist = sys.argv[5]


nsamples = file_lines(trainlist)
ngpus = 1
num_workers = 10

batch_size = int(net_options['batch'])
max_batches = int(net_options['max_batches'])
learning_rate = float(net_options['learning_rate'])
momentum = float(net_options['momentum'])
decay = float(net_options['decay'])
steps = [float(step) for step in net_options['steps'].split(',')]
scales = [float(scale) for scale in net_options['scales'].split(',')]

# Train parameters
max_epochs = max_batches * batch_size // nsamples + 1
use_cuda = True
seed = int(time.time())
eps = 1e-5
save_interval = 2000  # iterations
dot_interval = 70  # batches

# Test parameters
conf_thresh = 0.25
nms_thresh = 0.4
iou_thresh = 0.5

if not os.path.exists(backupdir):
    os.makedirs(backupdir)
###############
torch.manual_seed(seed)
if use_cuda:
    torch.cuda.manual_seed(seed)

model = Darknet(cfgfile)
region_loss = model.loss

model.load_weights(weightfile)
model.print_network()

model.seen = 0

region_loss.seen = model.seen
processed_batches = model.seen / batch_size

init_width = model.width
init_height = model.height
init_epoch = model.seen // nsamples

kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
if use_cuda:
    if ngpus > 1 and False:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()

params_dict = dict(model.named_parameters())
params = []
for key, value in params_dict.items():
    if key.find('.bn') >= 0 or key.find('.bias') >= 0:
        params += [{'params': [value], 'weight_decay': 0.0}]
    else:
        params += [{'params': [value], 'weight_decay': decay * batch_size}]
optimizer = optim.SGD(model.parameters(), lr=learning_rate / batch_size,
                      momentum=momentum, dampening=0, weight_decay=decay * batch_size)


def adjust_learning_rate(optimizer, batch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate
    for i in range(len(steps)):
        scale = scales[i] if i < len(scales) else 1
        if batch >= steps[i]:
            lr = lr * scale
            if batch == steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr / batch_size
    return lr


checkpoint_start_time = time.time()


def train(epoch):
    global processed_batches, checkpoint_start_time
    t0 = time.time()
    cur_model = model
    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(trainlist, shape=(init_width, init_height),
                            shuffle=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                            ]),
                            train=True,
                            seen=0,
                            batch_size=batch_size,
                            num_workers=num_workers),
        batch_size=batch_size, shuffle=False, **kwargs)

    lr = adjust_learning_rate(optimizer, processed_batches)
    model.train()
    features = []

    def hook_feature(module, input, output):
        features.append(output)
    h1 = model._modules.get('models')[16].register_forward_hook(
        hook_feature)  # layer chosen for calculating defense loss
    t1 = time.time()

    avg_time = torch.zeros(9)
    count = 0

    logging('epoch %d, processed %d samples, lr %f' %
            (epoch, epoch * len(train_loader.dataset), lr))
    for batch_idx, (data, target, path) in enumerate(train_loader):
        features = []
        t2 = time.time()
        adjust_learning_rate(optimizer, processed_batches)
        processed_batches = processed_batches + 1
        if use_cuda:
            data = data.cuda()
        t3 = time.time()
        data, target = Variable(data, requires_grad=True), Variable(target.float())
        t4 = time.time()
        optimizer.zero_grad()
        t5 = time.time()
        output = model(data)
        t6 = time.time()
        region_loss.seen = region_loss.seen + data.data.size(0)
        loss = region_loss(output, target, features[0])
        t7 = time.time()
        loss.backward()
        t8 = time.time()
        optimizer.step()
        t9 = time.time()
        if False and batch_idx > 1:
            avg_time[0] = avg_time[0] + (t2 - t1)
            avg_time[1] = avg_time[1] + (t3 - t2)
            avg_time[2] = avg_time[2] + (t4 - t3)
            avg_time[3] = avg_time[3] + (t5 - t4)
            avg_time[4] = avg_time[4] + (t6 - t5)
            avg_time[5] = avg_time[5] + (t7 - t6)
            avg_time[6] = avg_time[6] + (t8 - t7)
            avg_time[7] = avg_time[7] + (t9 - t8)
            avg_time[8] = avg_time[8] + (t9 - t1)
            print('-------------------------------')
            print('       load data : %f' % (avg_time[0] / (batch_idx)))
            print('     cpu to cuda : %f' % (avg_time[1] / (batch_idx)))
            print('cuda to variable : %f' % (avg_time[2] / (batch_idx)))
            print('       zero_grad : %f' % (avg_time[3] / (batch_idx)))
            print(' forward feature : %f' % (avg_time[4] / (batch_idx)))
            print('    forward loss : %f' % (avg_time[5] / (batch_idx)))
            print('        backward : %f' % (avg_time[6] / (batch_idx)))
            print('            step : %f' % (avg_time[7] / (batch_idx)))
            print('           total : %f' % (avg_time[8] / (batch_idx)))
        t1 = time.time()
        del loss, output, data
        torch.cuda.empty_cache()

        print('')
        t1 = time.time()
        logging('training with %f samples/s' %
                (len(train_loader.dataset) / (t1 - t0)))
        checkpoint_stop_time = time.time()
        training_time = checkpoint_stop_time - checkpoint_start_time

        if (processed_batches + 1) % save_interval == 0:
            logging('save weights to %s/%06d.weights' %
                    (backupdir, processed_batches + 1))
            cur_model.save_weights(
                '%s/%06d.weights' % (backupdir, processed_batches + 1))
    h1.remove()


for epoch in range(init_epoch, 1000):
    train(epoch)
