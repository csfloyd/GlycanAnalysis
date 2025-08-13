"""
Utility module for reaction networks.

This module contains utility functions and data classes for handling data
and performing common operations.
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import random

import signal

# Timeout handler function
def timeout_handler(signum, frame):
    raise TimeoutError("ODE integration timed out")

def compute_probs(C_full, L_vec, l0):
    """
    Compute probabilities from concentrations and conservation laws.
    
    Args:
        C_full: Full concentration array
        L_vec: Conservation law vector
        l0: Conservation law constant
        
    Returns:
        Probability array
    """
    return (L_vec[:, np.newaxis] * C_full / l0) if C_full.ndim == 2 else (L_vec * C_full / l0)


@dataclass
class InputData:
    """
    Manages labeled input data for training and testing.
    """

    def __init__(self, n_classes, data_list, split_fac=0.75):
        """
        Initializes training and testing datasets.

        Parameters:
        - n_classes: Number of output classes
        - data_list: List of data samples per class
        - split_fac: Fraction of data used for training
        """
        self.n_classes = n_classes
        self.labels = self._create_labels()
        self.data_list = data_list 
        self.split_fac = split_fac
        self.training_data, self.testing_data = self._split_shuffle_data(data_list, split_fac)

    def _create_labels(self):
        """Creates one-hot encoded labels for classification."""
        return [np.eye(self.n_classes)[n] for n in range(self.n_classes)]

    def _split_shuffle_data(self, data_list, split_fac):
        """Splits data into training and testing sets."""
        tr_data, te_data = [], []
        for nc in range(self.n_classes):
            sub_data = data_list[nc]
            random.shuffle(sub_data)
            n_train = round(split_fac * len(sub_data))
            tr_data.append(iter(sub_data[:n_train]))
            te_data.append(iter(sub_data[n_train:]))
        return tr_data, te_data

    def refill_iterators(self):
        """Refills the training and testing iterators from data_list."""
        self.training_data, self.testing_data = self._split_shuffle_data(self.data_list, self.split_fac)

    def get_next_training_sample(self, class_number):
        """Returns the next training sample for the given class, refilling iterators as needed."""
        try:
            return next(self.training_data[class_number])
        except StopIteration:
            self.refill_iterators()
            return next(self.training_data[class_number])