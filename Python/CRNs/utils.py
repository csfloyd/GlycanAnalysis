"""
Utility module for reaction networks.

This module contains utility functions and data classes for handling data
and performing common operations.
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import random
import time
import signal
import pickle

# Timeout handler function
def timeout_handler(signum, frame):
    raise TimeoutError("ODE integration timed out")

class TimeProfiler:
    """
    A class for profiling execution time of different code sections.
    
    This class provides a clean interface for timing various operations
    and generating summary reports.
    """
    
    def __init__(self):
        """Initialize the time profiler with empty timing data."""
        self.timers = {}
        self.total_start_time = None
        self.total_end_time = None
        
    def start_total_timer(self):
        """Start the overall timer for the entire process."""
        self.total_start_time = time.time()
        
    def end_total_timer(self):
        """End the overall timer for the entire process."""
        self.total_end_time = time.time()
        
    def start_timer(self, timer_name: str):
        """Start timing a specific operation.
        
        Args:
            timer_name: Name of the timer/category to start
        """
        if timer_name not in self.timers:
            self.timers[timer_name] = {'total_time': 0.0, 'start_time': None}
        self.timers[timer_name]['start_time'] = time.time()
        
    def end_timer(self, timer_name: str):
        """End timing a specific operation and accumulate the time.
        
        Args:
            timer_name: Name of the timer/category to end
        """
        if timer_name in self.timers and self.timers[timer_name]['start_time'] is not None:
            elapsed_time = time.time() - self.timers[timer_name]['start_time']
            self.timers[timer_name]['total_time'] += elapsed_time
            self.timers[timer_name]['start_time'] = None
            
    def get_timer(self, timer_name: str) -> float:
        """Get the total accumulated time for a specific timer.
        
        Args:
            timer_name: Name of the timer/category
            
        Returns:
            Total accumulated time in seconds
        """
        return self.timers.get(timer_name, {}).get('total_time', 0.0)
        
    def get_total_time(self) -> float:
        """Get the total elapsed time for the entire process.
        
        Returns:
            Total elapsed time in seconds
        """
        if self.total_start_time is None or self.total_end_time is None:
            return 0.0
        return self.total_end_time - self.total_start_time
        
    def print_summary(self, detailed: bool = True):
        """Print a detailed timing summary.
        
        Args:
            detailed: Whether to print detailed breakdown or just summary
        """
        total_time = self.get_total_time()
        if total_time == 0.0:
            print("No timing data available. Make sure to call start_total_timer() and end_total_timer().")
            return
            
        print(f"\n=== TIMING SUMMARY ===")
        print(f"Total time: {total_time:.2f} seconds")
        
        # Sort timers by total time (descending)
        sorted_timers = sorted(self.timers.items(), 
                              key=lambda x: x[1]['total_time'], 
                              reverse=True)
        
        for timer_name, timer_data in sorted_timers:
            timer_time = timer_data['total_time']
            percentage = (timer_time / total_time) * 100
            print(f"{timer_name}: {timer_time:.2f} seconds ({percentage:.1f}%)")
            
        # Calculate "other" time
        tracked_time = sum(timer_data['total_time'] for timer_data in self.timers.values())
        other_time = total_time - tracked_time
        if other_time > 0:
            other_percentage = (other_time / total_time) * 100
            print(f"Other operations: {other_time:.2f} seconds ({other_percentage:.1f}%)")
            
        print(f"=====================\n")
        
    def reset(self):
        """Reset all timing data."""
        self.timers = {}
        self.total_start_time = None
        self.total_end_time = None
        
    def context_manager(self, timer_name: str):
        """Context manager for automatic timing.
        
        Usage:
            with profiler.context_manager("operation_name"):
                # code to time
                pass
        """
        return TimerContext(self, timer_name)


class TimerContext:
    """Context manager for automatic timing within a TimeProfiler."""
    
    def __init__(self, profiler: TimeProfiler, timer_name: str):
        self.profiler = profiler
        self.timer_name = timer_name
        
    def __enter__(self):
        self.profiler.start_timer(self.timer_name)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.end_timer(self.timer_name)


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

@dataclass
class NetworkDataLogger:
    """
    A class for logging and managing reaction network data during sampling.
    
    This class provides a clean interface for storing reaction network parameters,
    conservation group changes, cycles, and sign conditions for each sampled network.
    """
    
    def __init__(self, filepath: str = None):
        """
        Initialize the data logger.
        
        Args:
            filepath: Optional path to a saved data file to load
        """
        self.network_data = []
        
        if filepath is not None:
            self.load_data(filepath)
        
    def log_network(self, 
                   r_n, 
                   M_0t1, 
                   M_1t0, 
                   M_0b1, 
                   cycles, 
                   sign_conditions,
                   C_full_list,
                   iteration: int = None,
                   seed: int = None,
                   **kwargs):
        # Extract essential network parameters for reconstruction
        network_params = {
            'all_complexes': r_n.all_complexes,
            'reactions': r_n.reactions,
            'n_species': r_n.n_species,
            'n_complexes': r_n.n_complexes,
            'n_reactions': r_n.n_reactions,
            'n_lcs': r_n.n_lcs,
            'L': r_n.L,
            'complexes_per_class': r_n.complexes_per_class,
            'reactions_per_class': r_n.reactions_per_class,
            'force_reverse': r_n.force_reverse,
            'subset_group_ind': r_n.subset_group_ind
        }
        
        # Create data entry
        data_entry = {
            'network_params': network_params,
            'conservation_changes': {
                'M_0t1': M_0t1,
                'M_1t0': M_1t0,
                'M_0b1': M_0b1
            },
            'cycles': cycles,
            'n_cycles': len(cycles[1]),
            'sign_conditions': sign_conditions,
            'n_sign_conditions': len(set(str(s) for s in sign_conditions)),
            'C_full_list': C_full_list,
            'iteration': iteration,
            'seed': seed,
            'additional_data': kwargs
        }
        
        self.network_data.append(data_entry)
        
        
    def save_data(self, filepath: str):


        save_data = {
            'network_data': self.network_data
        }
            
        with open(filepath, "wb") as file:
            pickle.dump(save_data, file)
            
        print(f"Data saved to {filepath}")
        
    def load_data(self, filepath: str):

        
        with open(filepath, "rb") as file:
            loaded_data = pickle.load(file)
            
        self.network_data = loaded_data['network_data']
            
        print(f"Data loaded from {filepath}")
        
    def get_network_by_index(self, index: int):

        if 0 <= index < len(self.network_data):
            return self.network_data[index]
        else:
            raise IndexError(f"Network index {index} out of range (0-{len(self.network_data)-1})")
            
    def filter_networks(self, **criteria):

        filtered_data = []
        
        for entry in self.network_data:
            match = True
            for key, value in criteria.items():
                if key in entry and entry[key] != value:
                    match = False
                    break
            if match:
                filtered_data.append(entry)
                
        return filtered_data
        
    def clear_data(self):
        """Clear all logged data."""
        self.network_data = []