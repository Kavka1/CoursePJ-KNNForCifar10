from typing import List
import numpy as np
import random
import json
import matplotlib.pyplot as plt
from numpy.random.mtrand import f
from kmeans import K_means
from dataloader import Dataloader


def show_cluster_img(data: np.array, labels: np.array, cluster: List[List]) -> None:
    fig = plt.figure(figsize=(12,10))                                                                                   
    plt.ion()                                                                                                           
    for i, idxs in enumerate(cluster):            
        chosen_idx = [idxs[random.randint(0, len(idxs))] for _ in range(5)]
        for j, idx in enumerate(chosen_idx):
            image = data[idx]
            image = image.reshape(-1, 32, 32)
            label = labels[idx]
            ax = fig.add_subplot(len(cluster), 5, i * 5 + j + 1)
            ax.imshow(image.T)
            ax.set_title(f'label: {label} cluster: {i}')
            ax.set_xticks([])
            ax.set_yticks([])   
    plt.tight_layout()                                                                             
    plt.ioff()                                                                                     
    plt.show()


def train_and_demonstrate() -> None:
    model = K_means(k=4, norm_ord=2, thereshold=0.00001)
    dataloader = Dataloader(path= "/home/xukang/GitRepo/CoursePJ-KmeansForCifar10/")
    _ = model.train(data=dataloader.data, DBI_calculate=False)
    show_cluster_img(data= dataloader.data, labels=dataloader.label, cluster=model.cluster)


def plot_DBI_curve(results_path: str, result_name: str) -> None:
    path = results_path + result_name + "/config_result.json"
    
    with open(path, 'r') as f:
        config_result = json.load(f)
    results = config_result['results']

    x_data, y_data = [], []
    for i, item in enumerate(results):
        x_data.append(item[0])
        y_data.append(item[1])
    
    plt.figure()
    plt.plot(x_data, y_data, label=result_name, color='c', marker='x')

    plt.legend()
    plt.grid()
    plt.title('DBI curves')
    plt.xlabel('Iteration')
    plt.ylabel('DBI')
    plt.show()


def plot_all_DBI_curves(result_path: str) -> None:
    fig = plt.figure()
    plt.ion()
    for k in range(2, 13):
        x_data, y_data = [[], []], [[], []]
        for i in [1, 2]:
            path = result_path + f'K_{k}_Ord_{i}' + '/config_result.json'
            with open(path, 'r') as f:
                results = json.load(f)['results']
            for j, item in enumerate(results):
                x_data[i-1].append(item[0])
                y_data[i-1].append(item[1])
        
        ax = fig.add_subplot(4, 3, k-1)
        ax.plot(x_data[0], y_data[0], marker='x')
        ax.plot(x_data[1], y_data[1], marker='x')
        ax.grid()
        ax.set_title(f'DBI curve for K = {k}')

    plt.legend(('l1_nrom', 'l2_norm'))
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.35, wspace=0.35)
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    '''
    plot_DBI_curve(
        results_path= '/home/xukang/GitRepo/CoursePJ-KmeansForCifar10/results/',
        result_name= 'K_5_Ord_2'
    )
    '''
    train_and_demonstrate()
    '''
    plot_all_DBI_curves(
        result_path= '/home/xukang/GitRepo/CoursePJ-KmeansForCifar10/results/'
    )
    '''