import numpy as np
import pandas as pd
from typing import Tuple
from matplotlib import pyplot as plt


def get_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # load the data
    train_df = pd.read_csv('data/mnist_train.csv')
    test_df = pd.read_csv('data/mnist_test.csv')

    X_train = train_df.drop('label', axis=1).values
    y_train = train_df['label'].values

    X_test = test_df.drop('label', axis=1).values
    y_test = test_df['label'].values

    return X_train, X_test, y_train, y_test


def normalize(X_train, X_test) -> Tuple[np.ndarray, np.ndarray]:
    # normalize the data
    X_train_normalized = 2 * X_train.astype('float32') / 255.0 - 1
    X_test_normalized = 2 * X_test.astype('float32') / 255.0 - 1
    
    return X_train_normalized, X_test_normalized


def plot_metrics(metrics,parameters) -> None:
    
    learning_rate,num_iters,C = parameters
    
    # plot the results
    directory = "./plots/"
    # plot accuracy
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    fig.tight_layout(pad=5.0)

    axs.plot([m[0] for m in metrics], [m[1] for m in metrics])
    axs.set_xlabel('k')
    axs.set_ylabel('Accuracy')
    axs.set_title('Accuracy vs k')
    filename = "plot_acc_"+str(learning_rate)+"_"+str(num_iters)+"_"+str(C)+".png"
    plt.savefig(directory+filename)

    # plot precision
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    fig.tight_layout(pad=5.0)
    
    axs.plot([m[0] for m in metrics], [m[2] for m in metrics])
    axs.set_xlabel('k')
    axs.set_ylabel('Precision')
    axs.set_title('Precision vs k')
    filename = "plot_pre_"+str(learning_rate)+"_"+str(num_iters)+"_"+str(C)+".png"
    plt.savefig(directory+filename)

    # plot recall
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    fig.tight_layout(pad=5.0)

    axs.plot([m[0] for m in metrics], [m[3] for m in metrics])
    axs.set_xlabel('k')
    axs.set_ylabel('Recall')
    axs.set_title('Recall vs k')
    filename = "plot_rec_"+str(learning_rate)+"_"+str(num_iters)+"_"+str(C)+".png"
    plt.savefig(directory+filename)

    # plot f1-score
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    fig.tight_layout(pad=5.0)

    axs.plot([m[0] for m in metrics], [m[4] for m in metrics])
    axs.set_xlabel('k')
    axs.set_ylabel('F1-score')
    axs.set_title('F1-score vs k')
    filename = "plot_F1_"+str(learning_rate)+"_"+str(num_iters)+"_"+str(C)+".png"
    plt.savefig(directory+filename)
