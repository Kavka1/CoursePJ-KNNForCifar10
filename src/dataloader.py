from typing import Tuple
import numpy as np
import pickle


class Dataloader(object):
    def __init__(self, path: str) -> None:
        """
        Dataset loader

        Args:
            path (str): the file path of CIFAR-10 dataset
        """
        super(Dataloader, self).__init__()      # Super class init
        
        self.data = None                        # For image
        self.label = None                       # For label
        self.path = path + 'dataset/'           # Dataset path

        self._load_dataset()                    # Load the dataset from the file path
        self._preprocess()                      # Preprocess the data

    def _load_dataset(self) -> None:
        """
        Load the CIFAR-10 dataset
        """
        total_data = []                                     # Save the image data
        total_label = []                                    # Save the label data
        for i in range(1, 6):                               # Scroll the six batch
            path = self.path + f'data_batch_{i}'            # Get the specific batch data file
            with open(path, mode='rb') as f:                # Read the dataset
                source = pickle.load(f, encoding='latin1')  # Load the data from the batch
                data = source['data']                       # Load the image data
                label = source['labels']                    # Load the label data
            total_data.append(data)                         # Append this batch of images to List
            total_label.append(label)                       # Append this batch of label to List

        total_data = np.concatenate(total_data, axis=0)     # Concat the six batch of image data to one numpy array with 2 dimension
        total_label = np.concatenate(total_label, axis=0)   # Concat the six batch of image data to one numpy array with 2 dimension

        random_idx = np.random.randint(0, len(total_data)-1, size=int(len(total_data)/5))   # Randomly choose 1/5 data
        self.data = total_data[random_idx]                  # Get the image data
        self.label = total_label[random_idx]                # Get the label data

    def _preprocess(self) -> None:
        """
        Preprocess the image data
        """
        self.data = self.data / 255.                        # [0, 255] -> [0, 1]

    def get_data(self) -> Tuple[np.array, np.array]:
        """
        Get the image and label

        Returns:
            Tuple[np.array, np.array]: data, label
        """
        return self.data, self.label                        # Return the image and label