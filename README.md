# TriggerDefense

This is the code for "I did not know your model's training data last summer: "Backdoor" Defense against Membership Inference Attacks". We are trying to implement a defense for Merbership Inference Attacks.


## Requirements
+ Python3.7
+ Tensorflow 2.1.0
+ Tensorflow Datasets
+ Scikit-learn
+ tqdm
+ Numpy
+ Pillow
+ OpenCV

## Code Usage
dataLoader.py is to provide the data for other modules.

### Train Base Model:
Run the target Model: python Target.py. For the hyperparmeter, you can refer it in the paper.
The model weights will be saved in the following folder: weights/Baseline. And you could change the dataset's name and model's name, which is included in dataLoader.py and ModelUtil.py seperately.

### Try Single Directive Defense:
Run the single directive model: python SingleDirective.py

### Try Single Generative Defense:
Run the Single Generative model: python SingleGenerative.py

### Try Dual Directive Defense:
Run the Dual Directive model: python DualDirective.py

### Try Dual Generative Defense:
Run the Dual Generative model: python DualGenerative.py

### Defenses:
We also compare our defense with the following two defenses:

+ Diiferential Privacy: [code](https://github.com/tensorflow/privacy)
+ Adverserial Regularization: [paper](https://arxiv.org/pdf/1807.05852.pdf) - [code](https://github.com/NNToan-apcs/python-DP-DL)