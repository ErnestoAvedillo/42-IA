<!--HEADER-->
<h1 align="center"> 42 Outer | 
 <picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://cdn.simpleicons.org/42/white">
  <img alt="42" width=40 align="top" src="https://cdn.simpleicons.org/42/Black">
 </picture>
 Cursus 
<img alt="Warning" src="https://raw.githubusercontent.com/Mqxx/GitHub-Markdown/main/blockquotes/badge/dark-theme/warning.svg"> On going
</h1>
<!--FINISH HEADER-->

## MULTILAYER PERCEPTION

### Descripion
- With this exercise we lean to programm a neural network.
##### File configuration
- **_histogram.py_**: Plots an istogram for all 31 fields of the data.
- **_clean_data_.py_**: cleans all data not necesary and eliinates all outliers (5 sigma). It's execution willl create 2 files: data_train.csv and data_test.csv
- **_layer.py_**: Class to calculate the acivation formula for the forward and backward propagation
- **_layer.py_**: class for each layer
- **_network.py_**: computes the complete neural network.
- **_plotting.py_**: Plots the given points and the linear result.



##### Description
- **Execution**: 
    the process to execute this study is:

    python histogram.py data.csv
    
    python clean_data.py data.csv 

    python train_all.py data_train.csv

### Pictures
 python histogram.py data.csv
<p>
  <img src="./pictures/Screenshot from 2025-03-02 11-06-49.png">
  <img src="./pictures/Screenshot from 2025-03-02 11-08-05.png">
</p>

### Resources

* **[Understanding Backpropagation](https://towardsdatascience.com/understanding-backpropagation-abcc509ca9d0/)**
* **[Neural Networks and deep kearning](http://neuralnetworksanddeeplearning.com/)**
* **[The Softmax function and its derivative](https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/)**
* **[What is Adam Optimizer?](https://www.geeksforgeeks.org/adam-optimizer/)**
* **[ML | Momentum-based Gradient Optimizer](https://www.geeksforgeeks.org/ml-momentum-based-gradient-optimizer-introduction/)**
* **[Explicación del Optimizador Adagrad: Cómo funciona, aplicación y comparaciones](https://www.datacamp.com/es/tutorial/adagrad-optimizer-explained)**
* **[Nesterov Momentum Explained with examples in TensorFlow and PyTorch](https://medium.com/@giorgio.martinez1926/nesterov-momentum-explained-with-examples-in-tensorflow-and-pytorch-4673dbf21998)**
* **[RMSProp Optimizer in Deep Learning](https://www.geeksforgeeks.org/rmsprop-optimizer-in-deep-learning/)**