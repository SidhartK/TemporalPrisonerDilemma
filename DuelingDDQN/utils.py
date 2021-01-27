import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, figure_file):
    plt.plot(x, scores)
    plt.savefig(figure_file)
    plt.cla()

def plot_learning_curve_ro(x, scores, figure_file):
    plt.plot(x, scores, 'ro')
    plt.savefig(figure_file)
    plt.cla()

def plot_learning_curve_last100(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)
    plt.cla()
