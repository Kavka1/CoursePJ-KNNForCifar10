from os import path
from typing import Dict, List, Tuple
import json
import sys
import os


def load_config(path: str) -> Dict:
    """Load the config file

    Args:
        path (str): The path of config file path

    Returns:
        Dict: Dict of config
    """
    with open(path, 'r') as f:          # Open the config file
        config = json.load(f)           # Load via json
    return config                       # Return the Dict of config


def check_path(path: str) -> None:
    """Check whether the path exists, if not then create the direction

    Args:
        path (str): [the path which needs check]
    """
    if not os.path.exists(path):                                        # Check whether this path exist
        os.mkdir(path)                                                  # if don't exists, then create this direction


def save_experiment_result(config: Dict, count_dbi: List[Tuple[int, float]]) -> None:
    """Save the config and experiment results

    Args:
        config (Dict): experiment config
        count_dbi (List[Tuple[int, float]]): the experiment result: [(iteration, DBI)]
    """
    save_path = config['pj_path'] + 'results/' + f"K_{config['k']}_Ord_{config['norm_ord']}/"   # Get the save path
    check_path(save_path)                                                                       # Check the existence of the path

    config.update({'results': count_dbi})                                                       # Merge result data to config

    with open(save_path + 'config_result.json', 'w') as f:                                      # Open file
        json.dump(config, f)                                                                    # Write data to the file