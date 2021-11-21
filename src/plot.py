from typing import List
import numpy as np
import random
import matplotlib.pyplot as plt


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