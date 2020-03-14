''' 10/21/2018

    Take a PASCAL VOC2007 class and train a universal patch for the left-top location
    so that YOLO becomes blind to that particular class. Follow the gradient
    of the loss with respect to the image to reduce the confidence of the
    class being targeted. Reduce YOLO class scores, not
    objectness.

    Train one patch per class for training images.
    Test AP scores on test images.
'''

from __future__ import print_function
import sys
if len(sys.argv) != 12:
    print('Usage:')
    print('python train_universal_patch.py datacfg cfgfile weightfile trainlist \
                    backupdir noise_result_dir reqd_class_index gpu logfile batch_size max_epochs')
    exit()

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2
''' add import for saving images
'''
from torchvision.utils import save_image

import dataset
import random
import math
import os
from utils import *
from cfg import parse_cfg

from darknet import Darknet_NoiseAdded_BlindnessAttack

# Training settings
datacfg       = sys.argv[1]
cfgfile       = sys.argv[2]
weightfile    = sys.argv[3]

data_options  = read_data_cfg(datacfg)
net_options   = parse_cfg(cfgfile)[0]

trainlist     = sys.argv[4]
testlist      = data_options['valid']
backupdir     = sys.argv[5]
nsamples      = file_lines(trainlist)
gpus          = sys.argv[8]
ngpus         = len(gpus.split(','))
num_workers   = int(data_options['num_workers'])

batch_size    = int(sys.argv[10])

noise_result_dir = sys.argv[6]
reqd_class_index = int(sys.argv[7])

#Train parameters
use_cuda      = True
seed = 100

''' Add parameters for patch
'''
# Patch parameters
max_epochs = int(sys.argv[11])
patchSize = 100
num_iter = 10
start_x = 5
start_y = 5


if not os.path.exists(noise_result_dir):
    os.makedirs(noise_result_dir)

if not os.path.exists(os.path.dirname(sys.argv[9])):
    os.makedirs(os.path.dirname(sys.argv[9]))

###############
torch.manual_seed(seed)
if use_cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    torch.cuda.manual_seed(seed)

model       = Darknet_NoiseAdded_BlindnessAttack(cfgfile)
region_loss = model.loss

model.load_weights(weightfile)
model.print_network()

model.seen = 0

region_loss.seen  = model.seen
processed_batches = 0

init_width        = model.width
init_height       = model.height

kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}

if use_cuda:
    if ngpus > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()

''' Create mask.
'''
mask = torch.zeros(1, 3, init_height, init_width)
mask = mask.cuda()
num_fooled = 0

#FileLogger to give output to console as well as a file simultaneously.
class FileLogger(object):
    def __init__(self):
        self.terminal = sys.stdout
        if not os.path.exists(os.path.dirname(sys.argv[9])):
            os.makedirs(os.path.dirname(sys.argv[9]))
        self.log = open(sys.argv[9], "w")                    #take log file as argument

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

sys.stdout = FileLogger()

def train(epoch):
    global processed_batches
    t0 = time.time()
    if ngpus > 1:
        cur_model = model.module
    else:
        cur_model = model

    ''' Change train argument to False. Why?
    '''

    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(trainlist, shape=(init_width, init_height),
                       shuffle=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ]),
                       train=False,
                       seen=cur_model.seen,
                       batch_size=batch_size,
                       num_workers=num_workers),
        batch_size=batch_size, shuffle=True, **kwargs)

    ''' Use YOLO in evaluation mode
    '''
    model.eval()
    num_fooled = 0
    t1 = time.time()

    lr = 1e-2
    # The optimizer for updating the patch and the learning rate
    optimizer = torch.optim.Adam([model.noise], lr=lr)

    print("[" + time.asctime(time.localtime(time.time())) + "]" + 'Training file: ' + trainlist)
    print("[" + time.asctime(time.localtime(time.time())) + "]" + "Learning Rate: " + str(lr))

    print("[" + time.asctime(time.localtime(time.time())) + "]" + 'Training patch')

    for batch_idx, (data0, target0, imgpaths) in enumerate(train_loader):

        print("Trying to fool:" + imgpaths[0])
        fooled = False

        if use_cuda:
            data0 = data0.cuda()
            # target0 = target0.cuda()
        data0, target0 = Variable(data0, requires_grad=False), Variable(target0, requires_grad=False)

        data_cloned = data0.clone()
        for i in range(num_iter):
            mask.fill_(0)  # mask is a CUDA tensor of size 1x3x416x416. Fill with zeros.
            mask[:, :, start_y:start_y + patchSize, start_x:start_x + patchSize] = 1 # only the area with Patch is changed to all ones.

            # The patch area in the cloned tensor is changed to all zeros.
            # The remaining area contains the clean image data.
            data_cloned.data[:, :, start_y:start_y + patchSize, start_x:start_x + patchSize] = 0

            # Zero noise gradients
            optimizer.zero_grad()

            # multiply with mask to keep the patch and rest zeros
            # clamp to keep in 0-1 image tensor range
            model.noise.data = model.noise.data * mask
            model.noise.data = torch.clamp(model.noise.data, 0, 1)


            output_patch = model(data_cloned) # forward patched image

            # Change - to pass the targeted class as parameter to loss function
            loss, max_class_prob = region_loss(output_patch, target0, reqd_class_index)

            loss.backward()

            # backpropagate into the model noise
            optimizer.step()

            print("[" + time.asctime(time.localtime(time.time())) + "]" + 'Batch num:%d Iteration: %d / %d Loss : %f noise norm: %f Fooled so far: %d Max_prob: %f' \
                                % (batch_idx, i, num_iter, loss.item(), model.noise.norm(), num_fooled, max_class_prob))

            if max_class_prob < 0.35:           # Changed
                num_fooled = num_fooled + 1
                break

    if (epoch+1)%5 == 0:
        model.noise.data = torch.clamp(model.noise.data, 0, 1)
        np.save(noise_result_dir +'/'+ 'epoch_' + str(epoch+1) + '_universal_patch.npy', model.noise.data.cpu().squeeze())
        save_image(model.noise.data.squeeze(), noise_result_dir +'/'+ 'epoch_' + str(epoch+1) + '_universal_patch.png')


for epoch in range(0, max_epochs):
    print('Epoch {}'.format(epoch))
    train(epoch)

