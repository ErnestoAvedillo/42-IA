from matplotlib import pyplot as plt
import numpy as np

def print_loss(loss:np.array):
    if loss is None:
        exit(1)
    m = loss.size
    loss = loss[25:]
    epochs = range(0,loss.size)
    plt.plot(epochs, loss, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()