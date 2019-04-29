# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
import torch.nn as nn

from advertorch.utils import clamp
from advertorch.utils import normalize_by_pnorm
from advertorch.utils import clamp_by_pnorm
from advertorch.utils import is_float_or_torch_tensor
from advertorch.utils import batch_multiply
from advertorch.utils import batch_clamp
from advertorch.utils import replicate_input

from .base import Attack
from .base import LabelMixin
from .utils import rand_init_delta


def perturb_iterative(xvar, yvar, predict, nb_iter, eps, eps_iter, loss_fn,
                      delta_init=None, minimize=False, ord=np.inf,
                      clip_min=0.0, clip_max=1.0):
    """
    Iteratively maximize the loss over the input. It is a shared method for
    iterative attacks including IterativeGradientSign, LinfPGD, etc.

    :param xvar: input data.
    :param yvar: input labels.
    :param predict: forward pass function.
    :param nb_iter: number of iterations.
    :param eps: maximum distortion.
    :param eps_iter: attack step size per iteration.
    :param loss_fn: loss function.
    :param delta_init: (optional) tensor contains the random initialization.
    :param minimize: (optional bool) whether to minimize or maximize the loss.
    :param ord: (optional) the order of maximum distortion (inf or 2).
    :param clip_min: (optional float) mininum value per input dimension.
    :param clip_max: (optional float) maximum value per input dimension.
    :return: tensor containing the perturbed input.
    """
    if delta_init is not None:
        delta = delta_init
    else:
        delta = torch.zeros_like(xvar)

    delta.requires_grad_()
    for ii in range(nb_iter):
        outputs = predict(xvar + delta)
        loss = loss_fn(outputs, yvar)
        if minimize:
            loss = -loss

        loss.backward()
        if ord == np.inf:
            grad_sign = delta.grad.data.sign()
            delta.data = delta.data + batch_multiply(eps_iter, grad_sign)
            delta.data = batch_clamp(eps, delta.data)
            delta.data = clamp(xvar.data + delta.data, clip_min, clip_max
                               ) - xvar.data

        elif ord == 2:
            grad = delta.grad.data
            grad = normalize_by_pnorm(grad)
            delta.data = delta.data + batch_multiply(eps_iter, grad)
            delta.data = clamp(xvar.data + delta.data, clip_min, clip_max
                               ) - xvar.data
            if eps is not None:
                delta.data = clamp_by_pnorm(delta.data, ord, eps)
        else:
            error = "Only ord = inf and ord = 2 have been implemented"
            raise NotImplementedError(error)

        delta.grad.data.zero_()

    x_adv = clamp(xvar + delta, clip_min, clip_max)
    return x_adv



class PGDAttack(Attack, LabelMixin):
    """
    The projected gradient descent attack (Madry et al. 2017).
    The attack performs nb_iter steps of size eps_iter, while always staying
    within eps from the initial point.
    Paper: https://arxiv.org/pdf/1706.06083.pdf
    """

    def __init__(
            self, predict, loss_fn=None, eps=0.3, nb_iter=40,
            eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            ord=np.inf, targeted=False):
        """
        Create an instance of the PGDAttack.

        :param predict: forward pass function.
        :param loss_fn: loss function.
        :param eps: maximum distortion.
        :param nb_iter: number of iterations
        :param eps_iter: attack step size.
        :param rand_init: (optional bool) random initialization.
        :param clip_min: mininum value per input dimension.
        :param clip_max: maximum value per input dimension.
        :param ord: norm type of the norm constraints
        :param targeted: if the attack is targeted
        """
        super(PGDAttack, self).__init__(
            predict, loss_fn, clip_min, clip_max)
        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.rand_init = rand_init
        self.ord = ord
        self.targeted = targeted
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")

        assert is_float_or_torch_tensor(self.eps_iter)
        assert is_float_or_torch_tensor(self.eps)

    def perturb(self, x, y=None):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.

        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted
                    labels.
                  - if self.targeted=True, then y must be the targeted labels.
        :return: tensor containing perturbed inputs.
        """
        x, y = self._verify_and_process_inputs(x, y)

        delta = torch.zeros_like(x)
        delta = nn.Parameter(delta)
        if self.rand_init:
            rand_init_delta(
                delta, x, self.ord, self.eps, self.clip_min, self.clip_max)
            delta.data = clamp(
                x + delta.data, min=self.clip_min, max=self.clip_max) - x

        rval = perturb_iterative(
            x, y, self.predict, nb_iter=self.nb_iter,
            eps=self.eps, eps_iter=self.eps_iter,
            loss_fn=self.loss_fn, minimize=self.targeted,
            ord=self.ord, clip_min=self.clip_min,
            clip_max=self.clip_max, delta_init=delta)

        return rval.data


class LinfPGDAttack(PGDAttack):
    def __init__(
            self, predict, loss_fn=None, eps=0.3, nb_iter=40,
            eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            targeted=False):
        ord = np.inf
        super(LinfPGDAttack, self).__init__(
            predict, loss_fn, eps, nb_iter, eps_iter, rand_init,
            clip_min, clip_max, ord, targeted)


class L2PGDAttack(PGDAttack):
    def __init__(
            self, predict, loss_fn=None, eps=0.3, nb_iter=40,
            eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            targeted=False):
        ord = 2
        super(L2PGDAttack, self).__init__(
            predict, loss_fn, eps, nb_iter, eps_iter, rand_init,
            clip_min, clip_max, ord, targeted)


class L2BasicIterativeAttack(PGDAttack):
    """Like GradientAttack but with several steps for each epsilon."""

    def __init__(self, predict, loss_fn=None, eps=0.1, nb_iter=10,
                 eps_iter=0.05, clip_min=0., clip_max=1., targeted=False):
        ord = 2
        rand_init = False
        super(L2BasicIterativeAttack, self).__init__(
            predict, loss_fn, eps, nb_iter, eps_iter, rand_init,
            clip_min, clip_max, ord, targeted)


class LinfBasicIterativeAttack(PGDAttack):
    """
    Like GradientSignAttack but with several steps for each epsilon.
    Aka Basic Iterative Attack.
    Paper: https://arxiv.org/pdf/1611.01236.pdf
    """

    def __init__(self, predict, loss_fn=None, eps=0.1, nb_iter=10,
                 eps_iter=0.05, clip_min=0., clip_max=1., targeted=False):
        ord = np.inf
        rand_init = False
        super(LinfBasicIterativeAttack, self).__init__(
            predict, loss_fn, eps, nb_iter, eps_iter, rand_init,
            clip_min, clip_max, ord, targeted)


class MomentumIterativeAttack(Attack, LabelMixin):
    """
    The L-inf projected gradient descent attack (Dong et al. 2017).
    The attack performs nb_iter steps of size eps_iter, while always staying
    within eps from the initial point. The optimization is performed with
    momentum.
    Paper: https://arxiv.org/pdf/1710.06081.pdf
    """

    def __init__(
            self, predict, loss_fn=None, eps=0.3, nb_iter=40, decay_factor=1.,
            eps_iter=0.01, clip_min=0., clip_max=1., targeted=False):
        """
        Create an instance of the MomentumIterativeAttack.

        :param predict: forward pass function.
        :param loss_fn: loss function.
        :param eps: maximum distortion.
        :param nb_iter: number of iterations
        :param decay_factor: momentum decay factor.
        :param eps_iter: attack step size.
        :param clip_min: mininum value per input dimension.
        :param clip_max: maximum value per input dimension.
        :param targeted: if the attack is targeted.
        """
        super(MomentumIterativeAttack, self).__init__(
            predict, loss_fn, clip_min, clip_max)
        self.eps = eps
        self.nb_iter = nb_iter
        self.decay_factor = decay_factor
        self.eps_iter = eps_iter
        self.targeted = targeted
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")

    def perturb(self, x, y=None):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.

        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted
                    labels.
                  - if self.targeted=True, then y must be the targeted labels.
        :return: tensor containing perturbed inputs.
        """
        x, y = self._verify_and_process_inputs(x, y)

        delta = torch.zeros_like(x)
        g = torch.zeros_like(x)

        delta = nn.Parameter(delta)

        for i in range(self.nb_iter):

            if delta.grad is not None:
                delta.grad.detach_()
                delta.grad.zero_()

            imgadv = x + delta
            outputs = self.predict(imgadv)
            loss = self.loss_fn(outputs, y)
            if self.targeted:
                loss = -loss
            loss.backward()

            g = self.decay_factor * g + normalize_by_pnorm(
                delta.grad.data, p=1)
            # according to the paper it should be .sum(), but in their
            #   implementations (both cleverhans and the link from the paper)
            #   it is .mean(), but actually it shouldn't matter

            delta.data += self.eps_iter * torch.sign(g)
            # delta.data += self.eps / self.nb_iter * torch.sign(g)

            delta.data = clamp(
                delta.data, min=-self.eps, max=self.eps)
            delta.data = clamp(
                x + delta.data, min=self.clip_min, max=self.clip_max) - x

        rval = x + delta.data
        return rval


class FastFeatureAttack(Attack):
    """
    Fast attack against a target internal representation of a model using
    gradient descent (Sabour et al. 2016).
    Paper: https://arxiv.org/abs/1511.05122
    """

    def __init__(self, predict, loss_fn=None, eps=0.3, eps_iter=0.05,
                 nb_iter=10, rand_init=True, clip_min=0., clip_max=1.):
        """
        Create an instance of the FastFeatureAttack.

        :param predict: forward pass function.
        :param loss_fn: loss function.
        :param eps: maximum distortion.
        :param eps_iter: attack step size.
        :param nb_iter: number of iterations
        :param clip_min: mininum value per input dimension.
        :param clip_max: maximum value per input dimension.
        """
        super(FastFeatureAttack, self).__init__(
            predict, loss_fn, clip_min, clip_max)
        self.eps = eps
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.rand_init = rand_init
        self.clip_min = clip_min
        self.clip_max = clip_max
        if self.loss_fn is None:
            self.loss_fn = nn.MSELoss(reduction="sum")

    def perturb(self, source, guide, delta=None):
        """
        Given source, returns their adversarial counterparts
        with representations close to that of the guide.

        :param source: input tensor which we want to perturb.
        :param guide: targeted input.
        :param delta: tensor contains the random initialization.
        :return: tensor containing perturbed inputs.
        """
        # Initialization
        if delta is None:
            delta = torch.zeros_like(source)
            if self.rand_init:
                delta = delta.uniform_(-self.eps, self.eps)
        else:
            delta = delta.detach()

        delta.requires_grad_()

        source = replicate_input(source)
        guide = replicate_input(guide)
        guide_ftr = self.predict(guide).detach()

        xadv = perturb_iterative(source, guide_ftr, self.predict,
                                 self.nb_iter, eps_iter=self.eps_iter,
                                 loss_fn=self.loss_fn, minimize=True,
                                 ord=np.inf, eps=self.eps,
                                 clip_min=self.clip_min,
                                 clip_max=self.clip_max,
                                 delta_init=delta)

        xadv = clamp(xadv, self.clip_min, self.clip_max)

        return xadv.data
