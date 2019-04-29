## Overview


This repository contains the PyTorch implementation with adversarial-robust models
introduced in the following paper:

> _Improving Adversarial Robustness via Guided Complement Entropy_. <br>
**Hao-Yun Chen**\*, Jhao-Hong Liang\*, Shih-Chieh Chang, Jia-Yu Pan, Yu-Ting Chen, Wei Wei, Da-Cheng Juan. <br> <https://arxiv.org/abs/1903.09799>


## Usage

For training a robust model with Guided Complement Entropy. (alpha as the strength in our proposed guided factor)

	python main.py --GCE --alpha 0.333 

For testing the robustness to PGD adversarial attacks on the previous training model.

	python LinfPGDAttack.py --GCE --model <model path>
	

## Acknowledgement

The implementations for PGD adversarial attacks are adapted from [Advertorch](https://github.com/BorealisAI/advertorch).