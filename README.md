# Improving Pedestrian Prediction Models with Self-Supervised Continual Learning

<!--
ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->
![CI testing](https://github.com/tud-arm/scl/workflows/CI%20testing/badge.svg?branch=main&event=push)


<img src="figs/initial_image.png" alt="">

This repository contains the code for the paper:

**<a href="https://arxiv.org">Improving Pedestrian Prediction Models with Self-Supervised Continual Learning</a>**
<br>
<a href="https://c-salmi.github.io/">Chadi Salmi</a>,
<a href="">Luzia Knoedler</a>,
<a href="http://www.tudelft.nl/staff/h.zhu/">Hai Zhu</a>,
<a href="http://www.tudelft.nl/staff/bruno.debrito/">Bruno Brito</a>,
<a href="http://www.tudelft.nl/staff/j.alonsomora/">Javier Alonso-Mora</a>
<br>
Accepted for publication in [RA-L + ICRA 2021].

Autonomous mobile robots require accurate human motion predictions to safely and efficiently navigate among pedestrians, whose behavior may adapt to environmental changes. 
This paper introduces a self-supervised continual learning framework to improve data-driven pedestrian prediction models online across various scenarios continuously. 
In particular, we exploit online streams of pedestrian data, commonly available from the robot's detection and tracking pipelines, to refine the prediction model and its performance in unseen scenarios.
To avoid the forgetting of previously learned concepts, a problem known as catastrophic forgetting,
our framework includes a regularization loss to penalize changes of model parameters that are important for previous scenarios and retrains on a set of previous examples to retain past knowledge.
Experimental results on real and simulation data show that our approach can improve prediction performance in unseen scenarios while retaining knowledge from seen scenarios when compared to naively training the prediction model online.

## Description   
The focus of this repo is to test online learning methods that train trajectory prediction model architectures in simulation as well as in the real world. Rosinterfaces are provided to easily couple the predictors with existing robots running ROS.

## Installation Instructions   
First, install the package (it is advised to create a python virtual environment):
```bash
# clone project   
git clone https://github.com/tud-amr/scl
cd scl

# (advised) Create virtual env
mkdir venv && cd venv
python3 -m venv scl
source scl/bin/activate
cd ..
pip install --upgrade pip

# install project   
pip install -r requirements.txt
pip install -e .   
 ```   

## Run the code
First, start the [simulation environment](https://github.com/srl-freiburg/pedsim_ros):
 ```
 roslaunch pedsim_simulator pedsim.launch scenario:=corridor
```

Models and other scripts in this repository should run from the base directory. This makes resolving paths easier and more consistent.

Example of a command to run inference on a saved prediction model using the `rosinterface.predictor_node`:
 ```bash
python -m project.rosinterfaces.predictor_node \
  --model EthPredictor \
  --save_name eth-ucy_pos \
  --frequency 20
```

Example of a command to train an EthPredictor model on eth and ucy datasets for 50 epochs.
 ```bash
# run module (example: train the ewc_eth_predictor)   
python -m project.training_routines.ewc \
  --model EthPredictor \
  --datasets eth ucy \
  --tbptt 15 \
  --max_epochs 50 \
  --save_name my_model
```
