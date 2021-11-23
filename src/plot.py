from typing import List
import numpy as np
import random
import json
import matplotlib.pyplot as plt
from numpy.random.mtrand import f
from kmeans import K_means
from dataloader import Dataloader


def show_cluster_img(data: np.array, labels: np.array, cluster: List[List]) -> None:
    """Show several samples from each cluster

    Args:
        data (np.array): image data
        labels (np.array): the true labels
        cluster (List[List]): the cluster results
    """
    fig = plt.figure(figsize=(12,10))                                                # Initialize a figure                               
    plt.ion()                                                                        # Turn on interactive mode                                 
    for i, idxs in enumerate(cluster):                                               # Traverse each cluster
        chosen_idx = [idxs[random.randint(0, len(idxs))] for _ in range(5)]          # Choose 5 sample randomly per cluster
        for j, idx in enumerate(chosen_idx):                                         # Traverse each sample
            image = data[idx]                                                        # Get cressponding image
            image = image.reshape(-1, 32, 32)                                        # Reshape the image to 3 dimensions
            label = labels[idx]                                                      # Get the true label
            ax = fig.add_subplot(len(cluster), 5, i * 5 + j + 1)                     # Set a subplot and show an image
            ax.imshow(image.T)                                                       # Show the image
            ax.set_title(f'label: {label} cluster: {i}')                             # Set title with cluster and label
            ax.set_xticks([])                                                        # Clear the ticks in X axis
            ax.set_yticks([])                                                        # Clear the ticks in Y axis
    plt.tight_layout()                                                               # Tight the images distribution
    plt.ioff()                                                                       # Turn off interactive mode
    plt.show()                                                                       # Show the plot


def train_and_demonstrate() -> None:
    """
    Train single model and show the result
    """
    model = K_means(k=4, norm_ord=2, thereshold=0.00001)                                    # Initialize a K-means model
    dataloader = Dataloader(path= "/home/xukang/GitRepo/CoursePJ-KmeansForCifar10/")        # Initialize a dataloader
    _ = model.train(data=dataloader.data, DBI_calculate=False)                              # Train the model
    show_cluster_img(data= dataloader.data, labels=dataloader.label, cluster=model.cluster) # Show the results


def plot_DBI_curve(results_path: str, result_name: str) -> None:
    """Plot a DBI curve

    Args:
        results_path (str): the result file path
        result_name (str): name of the experiment
    """
    path = results_path + result_name + "/config_result.json"               # Get the result file path
    
    with open(path, 'r') as f:                                              # Open the file
        config_result = json.load(f)                                        # Load the result
    results = config_result['results']                                      # Get the DBI data

    x_data, y_data = [], []                                                 # For save iteration and DBI data
    for i, item in enumerate(results):                                      # Traverse each item
        x_data.append(item[0])                                              # Append the iteration
        y_data.append(item[1])                                              # Append the DBi data
    
    plt.figure()                                                            # Initialize a plot
    plt.plot(x_data, y_data, label=result_name, color='c', marker='x')      # Plot the curve of the DBI

    plt.legend()                                                            # Add the legend
    plt.grid()                                                              # Add the grid
    plt.title('DBI curves')                                                 # Set title 
    plt.xlabel('Iteration')                                                 # Set label of x axis
    plt.ylabel('DBI')                                                       # Set label of y axis
    plt.show()                                                              # Show the plot


def plot_all_DBI_curves(result_path: str) -> None:
    """Plot all DBi curves of experiments

    Args:
        result_path (str): the results file path
    """
    fig = plt.figure()                                                          # Initialize a plot
    plt.ion()                                                                   # Turn on the interactive mode
    for k in range(2, 13):                                                      # Traverse each K value
        x_data, y_data = [[], []], [[], []]                                     # Save for two sets of data
        for i in [1, 2]:                                                        # Traverse l1-norm and l2-norm
            path = result_path + f'K_{k}_Ord_{i}' + '/config_result.json'       # Get the result file path
            with open(path, 'r') as f:                                          # Open the file
                results = json.load(f)['results']                               # Load the result data
            for j, item in enumerate(results):                                  # Traverse each iteration and DBI value
                x_data[i-1].append(item[0])                                     # Add the interation
                y_data[i-1].append(item[1])                                     # Add the DBI value
        
        ax = fig.add_subplot(4, 3, k-1)                                         # Locate the subplot
        ax.plot(x_data[0], y_data[0], marker='x')                               # Plot the curve of l1-norm
        ax.plot(x_data[1], y_data[1], marker='x')                               # Plot the curve of l2-norm
        ax.grid()                                                               # Add grid to the plot
        ax.set_title(f'DBI curve for K = {k}')                                  # Set the plot title

    plt.legend(('l1_nrom', 'l2_norm'))                                          # Set the legend of curves
    plt.subplots_adjust(top=0.92, bottom=0.08, 
                        left=0.10, right=0.95, 
                        hspace=0.35, wspace=0.35)                               # Addjust the distribution of subplots
    plt.ioff()                                                                  # Turn off interactive mode
    plt.show()                                                                  # Show the plot