from __future__ import print_function

import numpy as np
import os
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as functional

from torch.autograd import Variable
from advertorch.attacks import LinfPGDAttack
from GCE import *

np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser(
    description='PGD Adversarial Attacks using GuidedComplementEntropy')
parser.add_argument('--GCE', action='store_true',
                    help='Using GuidedComplementEntropy as a loss function for crafting adversarial examples')
parser.add_argument('--alpha', '-a', default=0.333, type=float,
                    help='alpha for guiding factor') 
parser.add_argument('--model', default='default', type=str, help='load a training model from your (physical) path')
parser.add_argument('--batch-size', '-b', default=64,
                    type=int, help='mini-batch size (default: 64)')
parser.add_argument('--eps', '-e', default=0.3, type=float,
                    help='Set an eplison value for PGD adversarial attacks')

args = parser.parse_args()

# load training model
ckpt_name = args.model
checkpoint = torch.load(ckpt_name)
net = checkpoint['net']
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch'] + 1
rng_state = checkpoint['rng_state']
torch.set_rng_state(rng_state)


# scale to [0, 1] without standard normalize
transform_train = transforms.Compose([
    transforms.ToTensor()
    # transforms.Normalize((0.1307,), (0.3081,))
])

transform_test = transforms.Compose([
    transforms.ToTensor()
])


# trainset = torchvision.datasets.MNIST(root='./data', train=True,
#                                         download=False, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
#                                           shuffle=True, num_workers=2)
          
        
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000,
                                         shuffle=False, num_workers=2)


if args.GCE:
    adversary = LinfPGDAttack(
        net, loss_fn=GuidedComplementEntropy(args.alpha), eps=args.eps,
        nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0,
        targeted=False)
else:
    adversary = LinfPGDAttack(
        net, loss_fn=nn.CrossEntropyLoss(), eps=args.eps,
        nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0,
        targeted=False)

net.eval()
correct = 0
total = 0
for step, data in enumerate(testloader, 0):

    inputs, labels = data

    inputs = inputs.cuda()
    labels = labels.cuda()

    adv_inputs = adversary.perturb(inputs, labels)
    adv_inputs = Variable(adv_inputs)
    labels = Variable(labels)
    outputs_adv = net(adv_inputs)

    _, predicted = torch.max(outputs_adv.data, 1)
    total += labels.size(0)
    correct += predicted.eq(labels.data).cpu().sum()
    correct = correct.item()

print("Classification accuracy : {}%".format(100.*correct/total))