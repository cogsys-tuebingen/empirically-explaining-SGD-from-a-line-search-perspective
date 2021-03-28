# Empirically explaining SGD from a line search perspective:
This repository provides the code to rerun the experiments from the paper  _Empirically explaining SGD from a line search perspective_ TODO archive link.
Further on, this code can be used to get further results on other datasets and models.

In directory TODO the code to measure the full batch loss during SGD training is found. 
Edit the configuration_sgd_data_sampling.txt to run the code on different datasets or models.

In directory TODO the code to analyze the data is found. TODO. 

If you have any questions or suggestions, please do not hesitate to contact me: maximus.mutschler(at)uni-tuebingen.de

## About the paper:
_Empirically explaining SGD from a line search perspective_ analyzes the training trjectory of SGD, used to train a ResNet-20 on 8% of Cifar-10, on a significantly deeper level.
For each update step the full-batch loss as well as all sample losses are measured on a line in update step direction.
Form these measurements the following core results are obtained:
1. the full batch loss along line in update step direction behaves parabolic.
2. with the correct learning rate SGD always performs an exact line search on the full-batch loss.
3. Increasing the batch size by a factor has the same effect as decreasing the learning rate by the same factor.
4. The update step size to the minimum of the full-batch loss behaves almost proportional to the norm of the direction defining batch.

<p float="left"> 
<img src="/images/line1.png" title="full-batch loss along update step direction" alt="full-batch loss along update step direction" width="380" />
<img src="images/line2.png" title="full-batch loss along update step direction" alt="full-batch loss along update step direction" width="380" />
<img src="images/line3.png" title="full-batch loss along update step direction" alt="full-batch loss along update step direction" width="380" />
</p>


However, to show the generality of those observations this code has to be run on more datasets and models.
Thus, feel free to do so.

Maximus Mutschler







