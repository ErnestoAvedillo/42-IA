from matplotlib import pyplot as plt
import numpy as np

def print_graph(x:np.array, y:np.array, theta:np.array):
    if y.size != x.shape[0]:
        print(f"Both x and y must have same dimension but x has {x.hape[0]}and y has {y.size}")
        exit(1)
    if x.ndim == 1:
        x = x.reshape(x.size, 1)
    if y.ndim == 1:
        y = y.reshape(y.size, 1)
    Table = np.concatenate((x, y), axis=1)
    Sorted_table = Table[Table[:,0].argsort()]
    plt.scatter(Sorted_table[:,0], Sorted_table[:,1], color='red', label='Real data')
    x_list = np.arange(Sorted_table[0,0], Sorted_table[-1,0],abs(Sorted_table[0,0]- Sorted_table[-1,0]) / 1000)
    x_predicted = np.ones((x_list.size,1))
    for i in range(1, theta.size):
        x_predicted = np.concat((x_predicted,np.pow(x_list,i).reshape(x_list.size, 1)),axis = 1)
    y_predicted = np.dot(x_predicted, theta)
    #print (x_predicted)
    #print (theta)
    #print (y_predicted)
    plt.plot(x_list, y_predicted, color='blue', label='Predicted data')
    plt.show()