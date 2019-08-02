## Overview


This repository contains the PyTorch implementation with adversarial-robust training objective introduced in the following paper:

> _Improving Adversarial Robustness via Guided Complement Entropy_. <br>
**Hao-Yun Chen**\*, Jhao-Hong Liang\*, Shih-Chieh Chang, Jia-Yu Pan, Yu-Ting Chen, Wei Wei, Da-Cheng Juan. <br> <https://arxiv.org/abs/1903.09799>

## Introduction
In this paper, we propose a new training paradigm called Guided Complement Entropy (GCE) that is capable of achieving **"adversarial defense for free,"** which involves no additional procedures in the process of improving adversarial robustness. In addition to maximizing model probabilities on the ground-truth class like cross-entropy, we neutralize its probabilities on the incorrect classes along with a "guided" term to balance between these two terms. We show in the experiments that our method achieves better model robustness with even better performance compared to the commonly used cross-entropy training objective.

## Robustness under White-Box attacks

We demonstrate model accuracies trained with different training objective: Baseline (cross-entropy) and **GCE (Guided Complement Entropy)** under various of SOTA white-box attacks.

 MNIST is used to be the benchmark dataset and the perturbations are all set to be 0.2.

| Attack              | Baseline  | GCE |
|:-------------------|:---------------------|:---------------------|
| [FGSM (Goodfellow et al. 2015)][1]               |               38.88%  |               62.74%  |
| [MIM (Dong et al. 2018)][2]                |               2.29%  |               39.81%  |
| [PGD (Madry et al. 2018)][3]                |               1.58%  |               9.55%  |

[1]: https://arxiv.org/abs/1412.6572
[2]: https://arxiv.org/abs/1710.06081
[3]: https://arxiv.org/abs/1706.06083

## Usage

For training a robust model with Guided Complement Entropy. (alpha as the strength in our proposed guided factor)

	python main.py --GCE --alpha 0.333 

For testing the robustness to PGD adversarial attacks on the previous training model.

	python LinfPGDAttack.py --GCE --model <model path>
	

## Dependencies

* Python 3.6 
* Pytorch 1.0 +



## Acknowledgement

The implementations for PGD adversarial attacks are adapted from [Advertorch](https://github.com/BorealisAI/advertorch).