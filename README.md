# Improving Pedestrian Prediction Models with Self-Supervised Continual Learning

<img src="figs/initial_image.png" alt="">

This repository will contain the code for the paper:

**<a href="https://arxiv.org">Improving Pedestrian Prediction Models with Self-Supervised Continual Learning</a>**
<br>
<a href="">Chadi Salmi</a>,
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
Ths is a repository to train, validate and test recent trajectory prediction models. The focus of this repo is to test online learning
methods to train the trajectory prediction models. A number of rosinterfaces are provided to couple the predictors with
existing robots running ROS.

## Instalation Instructions   
First, install the package (it is advised to create a python virtual environment):
```bash
# clone project   
git clone https://github.com/c-salmi/online-learning-trajectory-prediction
cd online-learning-trajectory-prediction

# (advised) Create virtual env
mkdir venv && cd venv
python3 -m venv online-learning
source online-learning/bin/activate
cd ..
pip install --upgrade pip

# install project   
pip install -r requirements.txt
pip install -e .   
 ```   

## Run the code
First, start the simulation enviornment:
 ```
 roslaunch pedsim_simulator pedsim.launch scenario:=corridor
```

To run a specific model (or other script) in this repository, it is advised to run it from the base working directory.
This makes resolving paths easier and more consistent.

Example of a command to run a saved model through ROS
 ```bash
python -m project.rosinterfaces.predictor_node \
  --model EthPredictor \
  --save_name eth-ucy_pos \
  --frequency 20
```

Example of a command to train an EthPredictor model on eth and ucy dataset for 50 epochs.
 ```bash
# run module (example: train the ewc_eth_predictor)   
python -m project.training_routines.ewc \
  --model EthPredictor \
  --datasets eth ucy \
  --tbptt 15 \
  --max_epochs 50 \
  --save_name my_model
```

## Make your own script (Work in progress)
This project is setup as a package which means you can now easily import any file into any other file like so:

```python
from project.datatools.trajpred_datamodule import TrajpredDatamodule
from project.training_routines.standard import StandardTrainingRoutine
from project.models.eth_predictor import Model, Processor
from pytorch_lightning import Trainer

# model / processor
model = Model()
processor = Processor()

# data
data = TrajpredDatamodule(processor)

# training routine
tr = StandardTrainingRoutine(model)

# train
trainer = Trainer()
trainer.fit(tr, data)

# test using the best model!
trainer.test(test_dataloaders=data.test_dataloader())
```
 