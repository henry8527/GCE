# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
from cleverhans.model import Model as ClModel

from advertorch.attacks import LocalSearchAttack
from advertorch.attacks import SinglePixelAttack
from advertorch.attacks import SpatialTransformAttack
from advertorch.attacks import JacobianSaliencyMapAttack
from advertorch.attacks import LBFGSAttack
from advertorch.attacks import CarliniWagnerL2Attack
from advertorch.attacks import FastFeatureAttack
from advertorch.attacks import MomentumIterativeAttack
from advertorch.attacks import LinfPGDAttack
from advertorch.attacks import L2BasicIterativeAttack
from advertorch.attacks import GradientAttack
from advertorch.attacks import LinfBasicIterativeAttack
from advertorch.attacks import GradientSignAttack
from advertorch.defenses import JPEGFilter
from advertorch.defenses import BitSqueezing
from advertorch.defenses import MedianSmoothing2D
from advertorch.defenses import AverageSmoothing2D
from advertorch.defenses import GaussianSmoothing2D
from advertorch.defenses import BinaryFilter


DIM_INPUT = 15
NUM_CLASS = 5
BATCH_SIZE = 16

IMAGE_SIZE = 16
COLOR_CHANNEL = 3


# ###########################################################
# model definitions for testing


class SimpleModel(nn.Module):
    def __init__(self, dim_input=DIM_INPUT, num_classes=NUM_CLASS):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(dim_input, 10)
        self.fc2 = nn.Linear(10, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class SimpleImageModel(nn.Module):

    def __init__(self, num_classes=NUM_CLASS):
        super(SimpleImageModel, self).__init__()
        self.num_classes = NUM_CLASS
        self.conv1 = nn.Conv2d(
            COLOR_CHANNEL, 8, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(4)
        self.linear1 = nn.Linear(4 * 4 * 8, self.num_classes)

    def forward(self, x):
        out = self.maxpool1(self.relu1(self.conv1(x)))
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        return out


class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2)
        self.linear1 = nn.Linear(7 * 7 * 64, 200)
        self.relu3 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(200, 10)

    def forward(self, x):
        out = self.maxpool1(self.relu1(self.conv1(x)))
        out = self.maxpool2(self.relu2(self.conv2(out)))
        out = out.view(out.size(0), -1)
        out = self.relu3(self.linear1(out))
        out = self.linear2(out)
        return out


class MLP(nn.Module):
    # MLP-300-100

    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(28 * 28, 300)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(300, 100)
        self.relu2 = nn.ReLU(inplace=True)
        self.linear3 = nn.Linear(100, 10)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.linear1(out)
        out = self.relu1(out)
        out = self.linear2(out)
        out = self.relu2(out)
        out = self.linear3(out)
        return out


class SimpleModelTf(ClModel):

    def __init__(self, dim_input, num_classes, session=None):
        import keras
        self.sess = session
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(10, input_shape=(dim_input, )))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Dense(num_classes))
        self.model = model
        self.flag_weight_set = False

    def set_weights(self, weights):
        self.model.set_weights(weights)
        self.flag_weight_set = True

    def load_state_dict(self, w):
        self.set_weights([
            w['fc1.weight'].cpu().numpy().transpose(),
            w['fc1.bias'].cpu().numpy(),
            w['fc2.weight'].cpu().numpy().transpose(),
            w['fc2.bias'].cpu().numpy(),
        ])

    def get_logits(self, data):
        assert self.flag_weight_set, "Weight Not Set!!!"
        return self.model(data)

    def get_probs(self, data):
        assert self.flag_weight_set, "Weight Not Set!!!"
        return tf.nn.softmax(logits=self.model(data))


def load_weights_pt(model_pt, layers):
    import tensorflow as tf
    w = model_pt.state_dict()
    layers[0].W = tf.Variable(tf.convert_to_tensor(
        w['fc1.weight'].cpu().numpy().transpose(), tf.float32))
    layers[0].b = tf.Variable(tf.convert_to_tensor(
        w['fc1.bias'].cpu().numpy(), tf.float32))
    layers[2].W = tf.Variable(tf.convert_to_tensor(
        w['fc2.weight'].cpu().numpy().transpose(), tf.float32))
    layers[2].b = tf.Variable(tf.convert_to_tensor(
        w['fc2.bias'].cpu().numpy(), tf.float32))


def setup_simple_model(model_pt, input_shape):
    from cleverhans_tutorials.tutorial_models import MLP, Linear, ReLU
    layers = [Linear(10),
              ReLU(),
              Linear(10)]
    layers[0].name = 'fc1'
    layers[1].name = 'relu'
    layers[2].name = 'fc2'
    model = MLP(layers, input_shape)
    load_weights_pt(model_pt, layers)
    return model



# ###########################################################
# model and data generation functions for testing



def generate_random_toy_data(clip_min=0., clip_max=1.):
    data = torch.Tensor(BATCH_SIZE, DIM_INPUT).uniform_(clip_min, clip_max)
    label = torch.LongTensor(BATCH_SIZE).random_(NUM_CLASS)
    return data, label


def generate_random_image_toy_data(clip_min=0., clip_max=1.):
    data = torch.Tensor(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE).uniform_(
        clip_min, clip_max)
    label = torch.LongTensor(BATCH_SIZE).random_(NUM_CLASS)
    return data, label


def generate_data_model_on_vec():
    data, label = generate_random_toy_data()
    model = SimpleModel()
    model.eval()
    return data, label, model


def generate_data_model_on_img():
    data, label = generate_random_image_toy_data()
    model = SimpleImageModel()
    model.eval()
    return data, label, model


# ###########################################################
# construct data needed for testing
vecdata, veclabel, vecmodel = generate_data_model_on_vec()
imgdata, imglabel, imgmodel = generate_data_model_on_img()


# ###########################################################
# construct groups and configs needed for testing defenses


defense_kwargs = {
    BinaryFilter: {},
    BitSqueezing: {"bit_depth": 4},
    MedianSmoothing2D: {},
    GaussianSmoothing2D: {"sigma": 3, "channels": COLOR_CHANNEL},
    AverageSmoothing2D: {"kernel_size": 5, "channels": COLOR_CHANNEL},
    JPEGFilter: {},
}

defenses = defense_kwargs.keys()

# store one suitable data for test
defense_data = {
    BinaryFilter: vecdata,
    BitSqueezing: vecdata,
    MedianSmoothing2D: imgdata,
    GaussianSmoothing2D: imgdata,
    AverageSmoothing2D: imgdata,
    JPEGFilter: imgdata,
}

nograd_defenses = [
    BinaryFilter,
    BitSqueezing,
    JPEGFilter,
]

withgrad_defenses = [
    MedianSmoothing2D,
    GaussianSmoothing2D,
    AverageSmoothing2D,
]

image_only_defenses = [
    MedianSmoothing2D,
    GaussianSmoothing2D,
    AverageSmoothing2D,
    JPEGFilter,
]

# as opposed to image-only
general_input_defenses = [
    BitSqueezing,
    BinaryFilter,
]



# ###########################################################
# construct groups and configs needed for testing attacks



# as opposed to image-only
general_input_attacks = [
    GradientSignAttack,
    LinfBasicIterativeAttack,
    GradientAttack,
    L2BasicIterativeAttack,
    LinfPGDAttack,
    MomentumIterativeAttack,
    FastFeatureAttack,
    CarliniWagnerL2Attack,
    LBFGSAttack,
    JacobianSaliencyMapAttack,
    SinglePixelAttack,
]

image_only_attacks = [
    SpatialTransformAttack,
    LocalSearchAttack,
]

label_attacks = [
    GradientSignAttack,
    LinfBasicIterativeAttack,
    GradientAttack,
    L2BasicIterativeAttack,
    LinfPGDAttack,
    MomentumIterativeAttack,
    CarliniWagnerL2Attack,
    LBFGSAttack,
    JacobianSaliencyMapAttack,
    SpatialTransformAttack,
]

feature_attacks = [
    FastFeatureAttack,
]

batch_consistent_attacks = [
    GradientSignAttack,
    LinfBasicIterativeAttack,
    GradientAttack,
    L2BasicIterativeAttack,
    LinfPGDAttack,
    MomentumIterativeAttack,
    FastFeatureAttack,
    JacobianSaliencyMapAttack,
    # CarliniWagnerL2Attack,  # XXX: not exactly sure: test says no
    # LBFGSAttack,  # XXX: not exactly sure: test says no
    # SpatialTransformAttack,  # XXX: not exactly sure: test says no
]


targeted_only_attacks = [
    JacobianSaliencyMapAttack,
]


# ###########################################################
# helper functions


def merge2dicts(x, y):
    z = x.copy()
    z.update(y)
    return z
