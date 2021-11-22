from typing import List, Tuple, Dict
import numpy as np


class K_means(object):
    def __init__(self, k: int, norm_ord: int, thereshold: float) -> None:
        """
        K means method wrapper

        Args:
            k (int): the number of k
            norm_ord (int): the order of the norm
            thereshold (float): termination thereshold
        """
        super(K_means, self).__init__()             # Super class init
        self.k = k                                  # K num
        self.norm_ord = norm_ord                    # Order for the norm
        self.thereshold = thereshold                # Terminate the k means running when the mean change of the mean vector bellow this thereshold

        self.cluster = [[] for _ in range(k)]       # Save the index of images for each cluster
        self.mean_vector = [0. for _ in range(k)]   # Save the mean vector for each cluster

    def train(self, data: np.array) -> List[Tuple[int, float]]:
        """
        Running the clusterring for the dataset.

        Args:
            data (np.array): dataset of images
        
        Returns:
            List[Tuple[int, float]]: [(iteration, DBI), ...]
        """
        mean_vector = data[np.random.randint(0, len(data) - 1, size = self.k)]      # Randomly choose the initialized mean vector
        count = 0                                                                   # Iteration count
        Count_DBI = []                                                              # Save the DBI during training
        while True:                                                                 # Roll util termination condition meets
            cluster = [[] for _ in range(self.k)]                                   # Indexes of images from each cluster
            for i, item in enumerate(data):                                         # Roll each image
                dist = [np.linalg.norm(item - mean_vector[j], ord=self.norm_ord) for j in range(self.k)]    # Calculate the distance between the image and each mean vector
                category = np.argmin(np.array(dist))                                            # Get the nearest mean vector
                cluster[category].append(i)                                                     # Append this image index to the coressponding cluster
            mean_vector_new = np.array([np.mean(data[idxs], axis=0) for idxs in cluster])       # Calculate new mean vectors

            change = np.mean(mean_vector_new - mean_vector)                         # Calculate the change between updated mean vector and the old mean vector
            print(f"Iteration {count}: mean vector change: {change}")               # Log

            if abs(change) < self.thereshold:                                       # If the change of mean vectors smaller than the thereshold
                self.cluster = cluster                                              # Save the cluster
                self.mean_vector = mean_vector_new                                  # Save the mean vector
                print(f"Termination:\n---DBIs: {Count_DBI} \n---Num of data in each cluster: {[len(idxs) for idxs in self.cluster]}")    # Log
                Count_DBI.append((count, self.DBI_calculate(cluster, mean_vector_new, data)))   # Calculate the last DBI
                break                                                               # Terminate the while cycle
            else:        
                mean_vector = mean_vector_new                                       # Update the mean vector
                if count % 5 == 0:                                                 # Calculate DBI per 10 iteration
                    DBI = self.DBI_calculate(cluster, mean_vector, data)            # Calculate DBI number
                    Count_DBI.append((count, DBI))                                  # Save DBI data
                    print(f"--------Iteration: {count}-----DBI: {DBI}--------")     # Log
                count += 1                                                          # Update the iteration number
        
        return Count_DBI                                                            # Return the process log of train

    def DBI_calculate(self, cluster: List[List], mean_vector: np.array, data: np.array) -> float:
        """
        Calculate DBI for the clusters

        Args:
            cluster (List[List]): k clusters
            mean_vector (np.array): mean vector for each cluster
            data (np.array): dataset

        Returns:
            float: DBI number
        """
        cluster_avg = [0. for _ in range(len(cluster))]                             # Save Avg(C)
        for c in range(len(cluster)):                                               # Calculate all Avg(C_i)
            dist_sum = 0.                                                           # Sum of distances between each data 
            indexes = cluster[c]                                                    # Tne data indexes in each cluster
            for i in range(len(indexes)):                                           
                for j in range(i+1, len(indexes)):
                    dist_sum += np.linalg.norm(data[indexes[i]] - data[indexes[j]], ord=1)  # Sum of each distance between samples
            cluster_avg[c] = dist_sum * 2 / (len(indexes) * (len(indexes) - 1) + 1e-6)     # Calculate Avg(C_i)

        DBI = 0.                                                                    
        for c in range(len(cluster)):                                               # Rollout each cluster
            term = []                                                               # Save the inner term in the DBI
            for k in range(len(cluster)):                                           # Calculate (Avg(C_i) + Avg(C_j)) / d_cen(C_i, C_j)
                if k == c:                                                          # Except i == j
                    continue
                dist_ck = np.linalg.norm(mean_vector[c] - mean_vector[k], ord=1)    # Calculate d_cen(C_i, C_j)
                term.append((cluster_avg[c] + cluster_avg[k]) / dist_ck)            # Append each term
            DBI += max(term)                                                        # Max{(Avg(C_i) + Avg(C_j)) / d_cen(C_i, C_j)}
        
        return DBI / len(cluster)                                                   # Average by k
