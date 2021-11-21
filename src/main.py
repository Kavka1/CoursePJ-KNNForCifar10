from typing import List, Dict, Tuple
import numpy as np
import pickle
import matplotlib.pyplot as plt
from numpy import random

from numpy.core.fromnumeric import argmin, mean, size
from numpy.random.mtrand import rand, randint


def load_file(file_path: str) -> Tuple[np.array, np.array]:
    total_data = []
    total_label = []
    for i in range(1, 6):
        path = file_path + f'data_batch_{i}'
        with open(path, mode='rb') as f:
            source = pickle.load(f, encoding='latin1')
            data = source['data']
            label = source['labels']
        total_data.append(data)
        total_label.append(label)

    total_data = np.concatenate(total_data, axis=0)
    total_label = np.concatenate(total_label, axis=0)

    random_idx = np.random.randint(0, len(total_data)-1, size=int(len(total_data)/5))
    data = total_data[random_idx]
    label = total_label[random_idx]

    return data, label


def KNN(data: np.array, label:np.array, k: int = 10, norm_ord: int = 2, threshold: float = 0.0001) -> List:
    idx = np.random.randint(0, len(data), size=k)
    mean_vector = data[idx]

    end_cluster = []

    count = 0
    while True:
        cluster = [[] for _ in range(k)]
        for i, item in enumerate(data):
            dist = [np.linalg.norm(item - mean_vector[j], ord = norm_ord) for j in range(k)]
            category = argmin(dist)
            cluster[category].append(i)

        mean_vector_new = [np.mean(data[idxs], axis=0) for idxs in cluster]
        change = [np.mean(mean_vector_new[i] - mean_vector[i]) for i in range(k)]
        print(f"Iteration {count}: mean vector change: {change} change mean: {mean(change)}")
        mean_vector = mean_vector_new

        count += 1

        if abs(mean(change)) < threshold:
            end_cluster = cluster
            break

    return end_cluster


def show_cluster_img(data: np.array, labels: np.array, cluster: List[List]) -> None:
    fig = plt.figure(figsize=(10,10))                                                                                   
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


def calculate_DBI(data: np.array, labels: np.array, cluster: List[List]) -> None:
    DBI = 0.0

    mean_vec_cluster = [np.mean(data[idxs], axis=0) for idxs in cluster]
    avg_cluster = np.zeros(shape=len(cluster))
    for c, idxs in enumerate(cluster):
        size = len(cluster)
        dist_sum = 0.0
        for i in idxs:
            for j in idxs:
                dist_sum += np.linalg.norm(data[i] - data[j], ord=1)
        avg_cluster[c] = 2/(size * (size - 1)) * dist_sum

    matrix = np.zeros(shape=(len(cluster), len(cluster)))
    for i in range(len(cluster)):
        for j in range(len(cluster)):
            if i == j:
                continue
            d_cen_ij = np.linalg.norm(mean_vec_cluster[i] - mean_vec_cluster[j], ord=1)
            matrix[i][j] = np.array((avg_cluster[i] + avg_cluster[j]) / (d_cen_ij + 1e-8))
        DBI += max(matrix[i])

    DBI = DBI / len(cluster)
    print(f"-------------DBI : {DBI}------------")


if __name__ == '__main__':
    data, label = load_file(file_path='/Users/xukang/Project/Repo/CoursePJ-KNNCifar10/cifar-10-batches-py/')
    
    cluster = KNN(data, label, k=10, norm_ord=2, threshold= 0.001)

    
    calculate_DBI(data, label, cluster)

    show_cluster_img(data, label, cluster)


    print(f"over")