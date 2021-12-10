# Improving Pedestrian Prediction Models with Self-Supervised Continual Learning

<img src="figs/initial_image.png" alt="">

This repository will contain the code for the paper:

**<a href="https://arxiv.org">Improving Pedestrian Prediction Models with Self-Supervised Continual Learning</a>**
<br>
<a href="">Chadi Salmi</a>,
<a href="">Luzia Knoedler</a>,
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