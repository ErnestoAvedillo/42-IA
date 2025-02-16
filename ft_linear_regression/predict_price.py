import numpy as np
import pandas as pd
from plotting import print_graph
from regression import linear_regression
from plot_loss import print_loss
import json
import sys

if len(sys.argv) < 2:
    print("You must provide a csv file as an argument")
    sys.exit()
else:
    try:
        data = pd.read_csv(sys.argv[1])
    except:
        print("The file provided does not exist or is not a csv file")
        sys.exit()
    if len(sys.argv) == 3:
        try:
            grado_regression = int (sys.argv[2])
        except:
            print("second argument must me an integer.")
            exit(-1)
        x = np.array([data['km']])
        for i in range(2,grado_regression + 1):
            x = np.concat((x, np.power(np.array([data['km']]), i)), axis = 0 )
        x = x.T
    else:    
        x = np.array([data['km']]).T
    y = np.array(data['price'])
    theta, loss = linear_regression(x, y)
    #save data in json file
    arguments_file = "arguments.json"
    with open(arguments_file, "w", encoding="utf-8") as myfile:
        argument_dicc = {"arguments":theta.tolist()}
        json.dump(argument_dicc, myfile, indent = 4, ensure_ascii = False)
    #graph data
    #m = y.size
    #if x.ndim == 1:
    #    x = x.reshape(m, 1)
    #x = np.concat(( np.ones(m).reshape(m,1) ,x), axis = 1)
    #y_predicted = np.dot(x, theta).reshape(m, 1)
    #y = y.reshape(m, 1)
    print (f"The loss is {loss[-1]}")
    print_graph(x[:,0], y, theta)
    print_loss(loss)