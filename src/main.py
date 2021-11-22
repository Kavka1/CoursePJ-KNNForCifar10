import os
from typing import List, Dict, Tuple
import numpy as np
import copy
from multiprocessing import Process
from dataloader import Dataloader
from kmeans import K_means
from utils import load_config, save_experiment_result


def experiment(config: Dict) -> None:
    """Run a single experiment

    Args:
        config (Dict): The experiment config
    """
    loader = Dataloader(config["pj_path"])          # Load data
    model = K_means(
        k = config['k'],
        norm_ord = config['norm_ord'],
        thereshold = 0.00001
    )                                               # Initialize a k-means model
    count_dbi = model.train(loader.get_data()[0])   # Train the model 
    save_experiment_result(config, count_dbi)       # Save the results


if __name__ == '__main__':
    origin_config = load_config(
        path= os.path.abspath('.') + '/config.json'
    )                                                   # Load config file
    for single_k in origin_config['k']:                 # Traverse each k num
        for single_ord in origin_config['norm_ord']:    # Traverse each ord num
            config = copy.copy(origin_config)           # Copy a new config
            config.update({
                'k': single_k,
                'norm_ord': single_ord
            })                                          # Set the specific setting for a experiment
            Process(
                target= experiment,
                args= (config, )
            ).start()                                   # 1 experiment to 1 precess