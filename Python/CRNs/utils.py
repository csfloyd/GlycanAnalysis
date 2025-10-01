"""
Utility module for reaction networks.

This module contains utility functions and data classes for handling data
and performing common operations.
"""

import numpy as np
from typing import List, Dict, Tuple
from CRNs.generation import generate_positive_initial_concentrations_nnls
from dataclasses import dataclass
import random
import time
import signal
import pickle
import itertools
from scipy.integrate import solve_ivp

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

def get_sign_condition_ids(sign_conditions):
    # Create a dictionary to store unique sign conditions and their IDs
    unique_sign_conditions = {}
    sign_condition_ids = []

    # Assign IDs to each sign condition
    for sign_condition in sign_conditions:
        # Convert sign condition to tuple of tuples so it can be used as dict key
        # Each inner list needs to be converted to tuple to be hashable
        sign_condition_tuple = tuple(tuple(x) if isinstance(x, list) else x for x in sign_condition)
        
        # If we haven't seen this sign condition before, assign it a new ID
        if sign_condition_tuple not in unique_sign_conditions:
            unique_sign_conditions[sign_condition_tuple] = len(unique_sign_conditions)
        
        # Add the ID to our list
        sign_condition_ids.append(unique_sign_conditions[sign_condition_tuple])

    return sign_condition_ids

def count_unique_tensors_with_ids(tensor_list, conservative_comparison = False):
    def are_equal(tensor1, tensor2):
        """Check if two tensors are equal according to the given criteria."""
        import numpy as np
        
        # Convert to numpy arrays if they aren't already
        t1 = np.array(tensor1, dtype=float)
        t2 = np.array(tensor2, dtype=float)
        
        # Element-wise multiplication
        product = t1 * t2
        
        # Check if any product is -1 (they disagree at a position where at least one is zero)
        # Use np.any() explicitly to avoid boolean context issues
        if not conservative_comparison:
            # Check if any elements are different (not equal)
            has_negative_one = np.any(t1 != t2)
        else:
            product = t1 * t2
            has_negative_one = np.any(product == -1)
        
        return not has_negative_one
    
    def find_representative(tensor, representatives):
        """Find if tensor is equivalent to any existing representative."""
        for rep in representatives:
            if are_equal(tensor, rep):
                return rep
        return None
    
    # Store unique tensors as lists and their IDs
    unique_tensors = []
    tensor_ids = []
    
    for tensor in tensor_list:
        # Check if this tensor is equivalent to any existing unique tensor
        representative = find_representative(tensor, unique_tensors)
        
        if representative is None:
            # This is a new unique tensor
            unique_tensors.append(tensor)
            tensor_ids.append(len(unique_tensors) - 1)
        else:
            # This tensor is equivalent to an existing one, use that ID
            for i, rep in enumerate(unique_tensors):
                if are_equal(tensor, rep):
                    tensor_ids.append(i)
                    break
    
    return len(unique_tensors), tensor_ids

def clamp_sign(sign_array, off):
    result = np.zeros_like(sign_array)
    result[sign_array > off] = 1
    result[sign_array < -off] = -1
    return result

def convert_to_log(data_logger, network_index):
    dC_dl_list = data_logger.network_data[network_index]['dC_dl_list']
    C_full_list = data_logger.network_data[network_index]['C_full_list']
    l0_list = data_logger.network_data[network_index]['l0_list']
    d_log_C_d_log_l0_list = []
    for i in range(len(dC_dl_list)):
        dC_dl_full = dC_dl_list[i]  
        C_full = C_full_list[i]
        l0 = l0_list[i]
        d_log_C_d_log_l0 = np.zeros_like(dC_dl_full)
        for k in range(dC_dl_full.shape[0]):
            for l in range(dC_dl_full.shape[1]):
                d_log_C_d_log_l0[k, l] = dC_dl_full[k, l] * l0[l] / C_full[k]
        d_log_C_d_log_l0_list.append(d_log_C_d_log_l0)

    return d_log_C_d_log_l0_list

def get_sign_conditions(tensor_list, cut_off):
    sign_conditions = []
    for i in range(len(tensor_list)):
        sign_conditions.append(clamp_sign(tensor_list[i], cut_off).tolist())
    return sign_conditions


def get_n_unique_signs(data_logger, cut_off, conservative_comparison=False, log_scale=False, dims=None):
    n_unique_signs = []
    
    # Pre-allocate data processing
    if log_scale:
        data_list = [convert_to_log(data_logger, n) for n in range(len(data_logger.network_data))]
    else:
        data_list = [data_logger.network_data[n]['dC_dl_list'] for n in range(len(data_logger.network_data))]
    
    # Apply dimension filtering once if needed
    if dims is not None:
        rows, cols = dims
        data_list = [[data[i][np.ix_(rows, cols)] for i in range(len(data))] for data in data_list]
    
    # Process all data in parallel if possible
    for data in data_list:
        signs_conditions = get_sign_conditions(data, cut_off)
        n_unique, tensor_ids = count_unique_tensors_with_ids(signs_conditions, conservative_comparison)
        n_unique_signs.append(n_unique)
    
    return n_unique_signs

def get_n_unique_signs_index(index, data_logger, cut_off, conservative_comparison = False, log_scale = False, dims = None):
    data = convert_to_log(data_logger, index) if log_scale else data_logger.network_data[index]['dC_dl_list']
    if dims is not None:
        rows, cols = dims
        data = [data[i][np.ix_(rows, cols)] for i in range(len(data))]
    signs_conditions = get_sign_conditions(data, cut_off)
    n_unique, tensor_ids = count_unique_tensors_with_ids(signs_conditions, conservative_comparison)
    return n_unique, tensor_ids

# Timeout handler function
def timeout_handler(signum, frame):
    raise TimeoutError("ODE integration timed out")

class TimeProfiler:
    """
    A class for profiling execution time of different code sections.
    
    This class provides a clean interface for timing various operations
    and generating summary reports. Supports nested timing to avoid double-counting.
    """
    
    def __init__(self):
        """Initialize the time profiler with empty timing data."""
        self.timers = {}
        self.total_start_time = None
        self.total_end_time = None
        self.active_timer_stack = []  # Stack to track nested timers
        
    def start_total_timer(self):
        """Start the overall timer for the entire process."""
        self.total_start_time = time.time()
        
    def end_total_timer(self):
        """End the overall timer for the entire process."""
        self.total_end_time = time.time()
        
        # End any remaining active timers
        for timer_name in list(self.active_timer_stack):
            self.end_timer(timer_name)
        
    def start_timer(self, timer_name: str):
        """Start timing a specific operation.
        
        Args:
            timer_name: Name of the timer/category to start
        """
        if timer_name not in self.timers:
            self.timers[timer_name] = {'total_time': 0.0, 'start_time': None}
        
        # If there's an active timer, pause it before starting the new one
        if self.active_timer_stack:
            current_timer = self.active_timer_stack[-1]
            if self.timers[current_timer]['start_time'] is not None:
                elapsed_time = time.time() - self.timers[current_timer]['start_time']
                self.timers[current_timer]['total_time'] += elapsed_time
                self.timers[current_timer]['start_time'] = None
        
        # Start the new timer
        self.timers[timer_name]['start_time'] = time.time()
        self.active_timer_stack.append(timer_name)
        
    def end_timer(self, timer_name: str):
        """End timing a specific operation and accumulate the time.
        
        Args:
            timer_name: Name of the timer/category to end
        """
        if timer_name not in self.timers or self.timers[timer_name]['start_time'] is None:
            return
            
        # End the current timer
        elapsed_time = time.time() - self.timers[timer_name]['start_time']
        self.timers[timer_name]['total_time'] += elapsed_time
        self.timers[timer_name]['start_time'] = None
        
        # Remove from active stack
        if timer_name in self.active_timer_stack:
            self.active_timer_stack.remove(timer_name)
        
        # Resume the previous timer if there was one
        if self.active_timer_stack:
            previous_timer = self.active_timer_stack[-1]
            self.timers[previous_timer]['start_time'] = time.time()
            
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
        
        # Separate adaptive and non-adaptive timers
        adaptive_timers = []
        non_adaptive_timers = []
        
        for timer_name, timer_data in sorted_timers:
            if timer_name.startswith("adaptive_") and timer_name != "adaptive_sampling":
                adaptive_timers.append((timer_name, timer_data))
            else:
                non_adaptive_timers.append((timer_name, timer_data))

        # Print adaptive timers under a header
        if adaptive_timers:
            print(f"\nAdaptive Sampling Breakdown:")
            for timer_name, timer_data in adaptive_timers:
                timer_time = timer_data['total_time']
                percentage = (timer_time / total_time) * 100
                print(f"  {timer_name}: {timer_time:.2f} seconds ({percentage:.1f}%)")
        
        # Print non-adaptive timers first
        for timer_name, timer_data in non_adaptive_timers:
            timer_time = timer_data['total_time']
            percentage = (timer_time / total_time) * 100
            print(f"{timer_name}: {timer_time:.2f} seconds ({percentage:.1f}%)")
        
        
        
        # Calculate "other" time (now should be accurate with nested timing)
        tracked_time = sum(timer_data['total_time'] for timer_data in self.timers.values())
        other_time = total_time - tracked_time
        if other_time > 0:
            other_percentage = (other_time / total_time) * 100
            print(f"\nOther operations: {other_time:.2f} seconds ({other_percentage:.1f}%)")
            
        print(f"=====================\n")
        
    def reset(self):
        """Reset all timing data."""
        self.timers = {}
        self.total_start_time = None
        self.total_end_time = None
        self.active_timer_stack = []
        
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
        
    def log_network(self, **kwargs):
        """
        Log network data with arbitrary parameters.
        
        All provided keyword arguments will be stored in the data entry.
        No parameters are required - you can store any data you want.
        
        Common parameters you might want to include:
        - r_n: ReactionNetwork instance (will extract network_params if provided)
        - interaction_matrix: Interaction matrix data
        - cycles: Cycle data
        - sign_conditions: Sign condition data
        - C_full_list: Concentration data
        - l0_list: Conservation law data
        - iteration: Iteration number
        - seed: Random seed
        - Any other custom data you want to store
        """
        # Create data entry with all provided kwargs except r_n
        data_entry = {k: v for k, v in kwargs.items() if k != 'r_n'}
        
        # If r_n is provided, extract network parameters for reconstruction
        if 'r_n' in kwargs:
            r_n = kwargs['r_n']
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
                'subset_group_ind': r_n.subset_group_ind,
                'reaction_strings': r_n.get_reaction_strings_simple(include_reverse=False),
                'species_names': r_n.species_names,
                'seed': r_n.seed
            }
            data_entry['network_params'] = network_params
        
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
    
    def reconstruct_network(self, index: int, random_rates: bool = True):
        """
        Reconstruct a ReactionNetwork instance using reaction strings.
        This uses the from_reaction_strings constructor.
        
        Args:
            index: Index of the network to reconstruct
            random_rates: Whether to assign random rates (True) or preserve original rates (False)
            
        Returns:
            ReactionNetwork instance
            
        Raises:
            IndexError: If index is out of range
            ImportError: If ReactionNetwork class is not available
            KeyError: If reaction_strings not found in network_params
        """
        try:
            from CRNs.reaction_network import ReactionNetwork
        except ImportError:
            raise ImportError("ReactionNetwork class not available. Make sure CRNs.reaction_network is importable.")
        
        if 0 <= index < len(self.network_data):
            network_params = self.network_data[index]['network_params']
            
            if 'reaction_strings' not in network_params:
                raise KeyError("reaction_strings not found in network_params. Make sure to log networks with reaction strings.")
            
            reaction_strings = network_params['reaction_strings']
            L = network_params['L']
            seed = network_params.get('seed', 42)
            force_reverse = network_params.get('force_reverse', True)
            subset_group_ind = network_params.get('subset_group_ind')
            species_names = network_params.get('species_names')
            rates = [r[2] for r in network_params.get('reactions')]
            r_n = ReactionNetwork.from_reaction_strings(
                reaction_strings, L, seed, force_reverse, subset_group_ind, 
                random_rates, species_names
            )
            r_n.update_rates(rates)
            
            return r_n
        else:
            raise IndexError(f"Network index {index} out of range (0-{len(self.network_data)-1})")

class AdaptiveSampler:
    """
    Adaptive sampler for sign conditions using Good-Turing-like convergence criteria.
    
    This class implements adaptive sampling that continues until convergence
    is reached based on the rate of discovery of new sign conditions.
    """
    
    def __init__(self, 
                 input_dims: List[int],
                 default_l0: np.ndarray,
                 sc_grad_dims: List[List[int]] = None,
                 min_samples: int = 20,
                 max_samples: int = 1000,
                 convergence_window: int = 20,
                 convergence_threshold: float = 0.02,
                 timeout_seconds: int = 30,
                 l0_range: tuple = (0.0001, 1000.0),
                 profiler: TimeProfiler = None,
                 round_decimals: int = 3,
                 use_signal_alarms: bool = True,
                 steady_state_method: str = 'integration'):
        """
        Initialize the adaptive sampler.
        
        Args:
            input_dims: Array of dimensions of l0 from which to sample new values randomly
            default_l0: Default l0 values for dimensions not in input_dims
            sc_grad_dims: List of [rows, columns] specifying which dimensions of dC_dl_func to use for sign conditions
            min_samples: Minimum number of samples before checking convergence
            max_samples: Maximum number of samples to prevent infinite loops
            convergence_window: Number of recent samples to check for convergence
            convergence_threshold: Fraction of new sign conditions below which to stop
            timeout_seconds: Timeout for individual sample generation
            l0_range: Range for sampling l0 values
            profiler: Optional TimeProfiler instance for timing
            round_decimals: Number of decimal places for rounding
            use_signal_alarms: Whether to use signal alarms for timeout handling
            steady_state_method: Method for finding steady state ('integration' or 'root_finding')
        """
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.convergence_window = convergence_window
        self.convergence_threshold = convergence_threshold
        self.timeout_seconds = timeout_seconds
        self.profiler = profiler
        self.l0_range = l0_range
        self.round_decimals = round_decimals
        self.use_signal_alarms = use_signal_alarms
        self.input_dims = input_dims
        self.default_l0 = np.array(default_l0)  # Ensure it's a numpy array
        self.sc_grad_dims = sc_grad_dims
        # Sampling state
        self.sign_conditions = []
        self.C_full_list = []
        self.dC_dl_list = []
        self.sample_count = 0
        self.new_sign_count = 0
        self.convergence_history = []
        self.l0_list = []
        self.steady_state_method = steady_state_method
    def _check_convergence(self) -> bool:
        """
        Check if sampling has converged based on recent discovery rate.
        
        Returns:
            True if converged, False otherwise
        """
        if self.sample_count < self.min_samples:
            return False
            
        if self.sample_count >= self.max_samples:
            return True
            
        # Calculate discovery rate in recent window
        if len(self.convergence_history) >= self.convergence_window:
            recent_discovery_rate = sum(self.convergence_history[-self.convergence_window:]) / self.convergence_window
            return recent_discovery_rate <= self.convergence_threshold
            
        return False
    
    def _get_sign_sample(self, sim, L, t_span, num_points, int_method, r_tol, a_tol, precomputed_derivatives):
        """
        Generate a single sign sample with all necessary computations.
        
        Args:
            sim: ReactionNetworkSimulator instance
            L: Conservation law matrix
            t_span: Integration time span
            num_points: Number of integration points
            int_method: Integration method
            r_tol: Relative tolerance
            a_tol: Absolute tolerance
            precomputed_derivatives: Precomputed derivative functions (dR_dC_func, dR_dl_func, etc.)
            
        Returns:
            tuple: (signs, C_full, success) where success is boolean
        """
        try:
            # Sample initial conditions
            
            l0 = self.default_l0.copy()
            # Sample only the specified input dimensions
            for dim in self.input_dims:
                l0[dim] = np.exp(np.random.uniform(np.log(self.l0_range[0]), np.log(self.l0_range[1])))
        
            if self.profiler:
                self.profiler.start_timer("adaptive_nnls")
            
            # Generate initial concentrations
            C_full = generate_positive_initial_concentrations_nnls(L, l0)
            
            if self.profiler:
                self.profiler.end_timer("adaptive_nnls")
                self.profiler.start_timer("adaptive_initial_conditions")
            
            # Get reduced initial conditions
            _, C_reduced_init = sim.get_const_and_reduced_init(C_full)
            
            if self.profiler:
                self.profiler.end_timer("adaptive_initial_conditions")
                self.profiler.start_timer("adaptive_integration")
            
            # Set up timeout for integration
            if self.use_signal_alarms:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.timeout_seconds)
            
            try:
                # Create flexible RHS function
                flexible_reduced_ode_rhs = sim.make_reduced_rhs_with_conservation_flexible()
                
                # Integrate ODE
                if self.steady_state_method == 'root_finding':
                    _, C_reduced_final = sim.minimize_to_steady_state(flexible_reduced_ode_rhs, C_reduced_init, l0)
                else:
                    sol_reduced, C_reduced_final = sim.integrate(
                        lambda C: flexible_reduced_ode_rhs(C, l0), 
                        C_reduced_init, 
                        t_span=t_span,
                        num_points=num_points, 
                        method=int_method, 
                        rtol=r_tol, 
                        atol=a_tol
                    )
                
                if self.use_signal_alarms:
                    signal.alarm(0)  # Cancel timeout
                
                if self.profiler:
                    self.profiler.end_timer("adaptive_integration")
                    self.profiler.start_timer("adaptive_species_recovery")
                
                # Recover full species concentrations
                C_full = sim.recover_eliminated_species(l0, C_reduced_final)
                
                if self.profiler:
                    self.profiler.end_timer("adaptive_species_recovery")
                    self.profiler.start_timer("adaptive_sensitivity_analysis")
                
                # Compute sensitivity derivatives (use precomputed if available)
                rates = np.array([sim.r_n.reactions[r_idx][2] for r_idx in range(len(sim.r_n.reactions))])
                
                dR_dC_func, dR_dl_func = precomputed_derivatives
                
                dC_dl = sim.dC_dl_func(C_reduced_final, l0, rates, dR_dC_func, dR_dl_func)
                dC_dl_full = sim.compute_dC_dk_full(dC_dl, l_bool = True)
                
                if self.profiler:
                    self.profiler.end_timer("adaptive_sensitivity_analysis")
                    self.profiler.start_timer("adaptive_sign_processing")
                
                # Extract sign conditions
                if self.sc_grad_dims is not None:
                    # Use only specified dimensions of dC_dl_full
                    rows, cols = self.sc_grad_dims
                    dC_dl_subset = dC_dl_full[np.ix_(rows, cols)]
                    signs = np.sign(np.round(dC_dl_subset, decimals=self.round_decimals)).tolist()
                else:
                    # Use all dimensions (original behavior)
                    signs = np.sign(np.round(dC_dl_full, decimals=self.round_decimals)).tolist()
                
                if self.profiler:
                    self.profiler.end_timer("adaptive_sign_processing")
                
                return signs, C_full, dC_dl_full, l0, True
                
            except TimeoutError:
                if self.use_signal_alarms:
                    signal.alarm(0)  # Cancel timeout
                if self.profiler:
                    self.profiler.end_timer("adaptive_integration")
                return None, None, None, None, False
                
        except Exception as e:
            print(e)
            if self.profiler:
                # End any active timers
                for timer_name in ["adaptive_nnls", "adaptive_initial_conditions", "adaptive_integration", "adaptive_species_recovery", "adaptive_sensitivity_analysis", "adaptive_sign_processing"]:
                    if timer_name in self.profiler.timers and self.profiler.timers[timer_name]['start_time'] is not None:
                        self.profiler.end_timer(timer_name)
            return None, None, None, None, False
    
    def sample_sign_conditions(self, sim, L, t_span, num_points, int_method, r_tol, a_tol, precomputed_derivatives):
        """
        Perform adaptive sampling of sign conditions.
        
        Args:
            sim: ReactionNetworkSimulator instance
            L: Conservation law matrix
            t_span: Integration time span
            num_points: Number of integration points
            int_method: Integration method
            r_tol: Relative tolerance
            a_tol: Absolute tolerance
            precomputed_derivatives: Precomputed derivative functions (dR_dC_func, dR_dl_func, etc.)
            
        Returns:
            tuple: (sign_conditions, C_full_list, sample_count, convergence_reached)
        """
        # print(f"Starting adaptive sampling (min: {self.min_samples}, max: {self.max_samples})")
        
        while not self._check_convergence():
            
            # Generate sign sample
            signs, C_full, dC_dl_full, l0, success = self._get_sign_sample(sim, L, t_span, num_points, int_method, r_tol, a_tol, precomputed_derivatives)
            
            if not success:
                continue

            self.sample_count += 1
            
            # Check if this is a new sign condition
            signs_str = str(signs)
            is_new = True
            for existing_signs in self.sign_conditions:
                if str(existing_signs) == signs_str:
                    is_new = False
                    break

            # Update tracking
            self.convergence_history.append(1 if is_new else 0)
            if is_new:
                self.new_sign_count += 1

            self.sign_conditions.append(signs)
            self.C_full_list.append(C_full)
            self.dC_dl_list.append(dC_dl_full)
            self.l0_list.append(l0)
        
        convergence_reached = self._check_convergence()
        
        return self.sign_conditions, self.C_full_list, self.dC_dl_list, self.l0_list, self.sample_count, convergence_reached
    
    def reset(self):
        """Reset the sampler state for reuse."""
        self.sign_conditions = []
        self.C_full_list = []
        self.dC_dl_list = []
        self.l0_list = []
        self.sample_count = 0
        self.new_sign_count = 0
        self.convergence_history = []
    
    def get_statistics(self):
        """Get sampling statistics."""
        unique_sign_conditions = len(set(str(s) for s in self.sign_conditions))
        discovery_rate = self.new_sign_count / max(self.sample_count, 1)
        
        return {
            'total_samples': self.sample_count,
            'unique_sign_conditions': unique_sign_conditions,
            'discovery_rate': discovery_rate,
            'convergence_reached': self._check_convergence()
        }
    
class GridSampler:
    """
    Grid sampler for sign conditions.
    
    This class implements grid sampling of sign conditions.
    """
    
    def __init__(self, 
                 input_dims: List[int],
                 default_l0: np.ndarray,
                 sc_grad_dims: List[List[int]] = None,
                 l0_range: tuple = (0.0001, 1000.0),
                 l0_grid_size: int = 10,
                 grid_dim: int = 2,
                 timeout_seconds: int = 30,
                 profiler: TimeProfiler = None,
                 round_decimals: int = 3,
                 use_signal_alarms: bool = True,
                 steady_state_method: str = 'integration',
                 use_contour_integration: bool = False):
        """
        Initialize the grid sampler.
        
        Args:
            input_dims: Array of dimensions of l0 from which to sample new values randomly
            default_l0: Default l0 values for dimensions not in input_dims
            sc_grad_dims: List of [rows, columns] specifying which dimensions of dC_dl_func to use for sign conditions
            l0_range: Range for sampling l0 values
            l0_grid_size: Size of the grid for l0 values
            grid_dim: Dimension of the grid
            timeout_seconds: Timeout for individual sample generation
            profiler: Optional TimeProfiler instance for timing
            round_decimals: Number of decimal places for rounding
            use_signal_alarms: Whether to use signal alarms for timeout handling
            steady_state_method: Method for finding steady state ('integration' or 'root_finding')
            use_contour_integration: Whether to use contour integration for 1D cases
        """
        self.l0_range = l0_range
        self.l0_grid_size = l0_grid_size
        self.profiler = profiler
        self.l0_grid = np.logspace(np.log10(l0_range[0]), np.log10(l0_range[1]), l0_grid_size)
        self.grid_dim = grid_dim
        self.timeout_seconds = timeout_seconds
        self.round_decimals = round_decimals
        self.use_signal_alarms = use_signal_alarms
        self.input_dims = input_dims
        self.default_l0 = np.array(default_l0)  # Ensure it's a numpy array
        self.sc_grad_dims = sc_grad_dims
        # Sampling state
        self.sign_conditions = []
        self.C_full_list = []
        self.dC_dl_list = []
        self.sample_count = 0
        self.new_sign_count = 0
        self.convergence_history = []
        self.l0_list = []
        self.steady_state_method = steady_state_method
        self.use_contour_integration = use_contour_integration
    
    def _get_sign_sample(self, l0, sim, L, t_span, num_points, int_method, r_tol, a_tol, precomputed_derivatives):
        """
        Generate a single sign sample with all necessary computations.
        
        Args:
            sim: ReactionNetworkSimulator instance
            L: Conservation law matrix
            t_span: Integration time span
            num_points: Number of integration points
            int_method: Integration method
            r_tol: Relative tolerance
            a_tol: Absolute tolerance
            precomputed_derivatives: Precomputed derivative functions (dR_dC_func, dR_dl_func, etc.)
            
        Returns:
            tuple: (signs, C_full, success) where success is boolean
        """
        try:
            
            if self.profiler:
                self.profiler.start_timer("adaptive_nnls")
            
            # Generate initial concentrations
            C_full = generate_positive_initial_concentrations_nnls(L, l0)
            
            if self.profiler:
                self.profiler.end_timer("adaptive_nnls")
                self.profiler.start_timer("adaptive_initial_conditions")
            
            # Get reduced initial conditions
            _, C_reduced_init = sim.get_const_and_reduced_init(C_full)
            
            if self.profiler:
                self.profiler.end_timer("adaptive_initial_conditions")
                self.profiler.start_timer("adaptive_integration")
            
            # Set up timeout for integration
            if self.use_signal_alarms:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.timeout_seconds)
            
            try:
                # Create flexible RHS function
                flexible_reduced_ode_rhs = sim.make_reduced_rhs_with_conservation_flexible()
                
                # Integrate ODE
                if self.steady_state_method == 'root_finding':
                    _, C_reduced_final = sim.minimize_to_steady_state(flexible_reduced_ode_rhs, C_reduced_init, l0)
                else:
                    sol_reduced, C_reduced_final = sim.integrate(
                        lambda C: flexible_reduced_ode_rhs(C, l0), 
                        C_reduced_init, 
                        t_span=t_span,
                        num_points=num_points, 
                        method=int_method, 
                        rtol=r_tol, 
                        atol=a_tol
                    )
                
                if self.use_signal_alarms:
                    signal.alarm(0)  # Cancel timeout
                
                if self.profiler:
                    self.profiler.end_timer("adaptive_integration")
                    self.profiler.start_timer("adaptive_species_recovery")
                
                # Recover full species concentrations
                C_full = sim.recover_eliminated_species(l0, C_reduced_final)
                
                if self.profiler:
                    self.profiler.end_timer("adaptive_species_recovery")
                    self.profiler.start_timer("adaptive_sensitivity_analysis")
                
                # Compute sensitivity derivatives (use precomputed if available)
                rates = np.array([sim.r_n.reactions[r_idx][2] for r_idx in range(len(sim.r_n.reactions))])
                
                dR_dC_func, dR_dl_func = precomputed_derivatives
                
                dC_dl = sim.dC_dl_func(C_reduced_final, l0, rates, dR_dC_func, dR_dl_func)
                dC_dl_full = sim.compute_dC_dk_full(dC_dl, l_bool = True)
                
                if self.profiler:
                    self.profiler.end_timer("adaptive_sensitivity_analysis")
                    self.profiler.start_timer("adaptive_sign_processing")
                
                # Extract sign conditions
                if self.sc_grad_dims is not None:
                    # Use only specified dimensions of dC_dl_full
                    rows, cols = self.sc_grad_dims
                    dC_dl_subset = dC_dl_full[np.ix_(rows, cols)]
                    signs = np.sign(np.round(dC_dl_subset, decimals=self.round_decimals)).tolist()
                else:
                    # Use all dimensions (original behavior)
                    signs = np.sign(np.round(dC_dl_full, decimals=self.round_decimals)).tolist()
                
                if self.profiler:
                    self.profiler.end_timer("adaptive_sign_processing")
                
                return signs, C_full, dC_dl_full, l0, True
                
            except TimeoutError:
                if self.use_signal_alarms:
                    signal.alarm(0)  # Cancel timeout
                if self.profiler:
                    self.profiler.end_timer("adaptive_integration")
                return None, None, None, None, False
                
        except Exception as e:
            print(e)
            if self.profiler:
                # End any active timers
                for timer_name in ["adaptive_nnls", "adaptive_initial_conditions", "adaptive_integration", "adaptive_species_recovery", "adaptive_sensitivity_analysis", "adaptive_sign_processing"]:
                    if timer_name in self.profiler.timers and self.profiler.timers[timer_name]['start_time'] is not None:
                        self.profiler.end_timer(timer_name)
            return None, None, None, None, False
    
    def _sample_sign_conditions_contour_1d(self, sim, L, t_span, num_points, int_method, r_tol, a_tol, precomputed_derivatives):
        """
        Sample sign conditions using contour integration for 1D parameter space.
        This method performs a single contour integration along the 1D parameter space
        and extracts results at each grid point.
        
        Args:
            sim: ReactionNetworkSimulator instance
            L: Conservation law matrix
            t_span: Integration time span
            num_points: Number of integration points
            int_method: Integration method
            r_tol: Relative tolerance
            a_tol: Absolute tolerance
            precomputed_derivatives: Precomputed derivative functions (dR_dC_func, dR_dl_func, etc.)
            
        Returns:
            tuple: (sign_conditions, C_full_list, dC_dl_list, l0_list, sample_count, convergence_reached)
        """
        if len(self.input_dims) != 1:
            raise ValueError("Contour integration is only supported for 1D parameter spaces")
        
        try:
            # Get grid bounds (use actual grid points, not theoretical bounds)
            l0_min = self.l0_grid[0]
            l0_max = self.l0_grid[-1]
            dim = self.input_dims[0]
            
            # Set up initial conditions at leftmost point
            l0_init = self.default_l0.copy()
            l0_init[dim] = l0_min
            
            if self.profiler:
                self.profiler.start_timer("contour_nnls")
            
            # Generate initial concentrations at leftmost point
            C_full_init = generate_positive_initial_concentrations_nnls(L, l0_init)
            
            if self.profiler:
                self.profiler.end_timer("contour_nnls")
                self.profiler.start_timer("contour_initial_conditions")
            
            # Get reduced initial conditions
            _, C_reduced_init = sim.get_const_and_reduced_init(C_full_init)
            
            if self.profiler:
                self.profiler.end_timer("contour_initial_conditions")
                self.profiler.start_timer("contour_steady_state")
            
            # Integrate to steady state at the leftmost point
            try:
                # Create flexible RHS function
                flexible_reduced_ode_rhs = sim.make_reduced_rhs_with_conservation_flexible()
                
                # Integrate to steady state
                if self.steady_state_method == 'root_finding':
                    _, C_reduced_steady = sim.minimize_to_steady_state(flexible_reduced_ode_rhs, C_reduced_init, l0_init)
                else:
                    sol_reduced, C_reduced_steady = sim.integrate(
                        lambda C: flexible_reduced_ode_rhs(C, l0_init), 
                        C_reduced_init, 
                        t_span=t_span,
                        num_points=num_points, 
                        method=int_method, 
                        rtol=r_tol, 
                        atol=a_tol
                    )
                
                if self.profiler:
                    self.profiler.end_timer("contour_steady_state")
                    self.profiler.start_timer("contour_integration")
                
            except Exception as e:
                print(f"Error in steady state integration: {e}")
                if self.profiler:
                    self.profiler.end_timer("contour_steady_state")
                return [], [], [], [], 0, False
            
            # Set up timeout for integration
            if self.use_signal_alarms:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.timeout_seconds)
            
            try:
                # Get precomputed derivatives
                dR_dC_func, dR_dl_func = precomputed_derivatives
                rates = np.array([sim.r_n.reactions[r_idx][2] for r_idx in range(len(sim.r_n.reactions))])
                
                # Interpolate l0 at current t (t goes from 0 to 1) using log spacing
                l0_current = self.default_l0.copy()
                l0_current[dim] = l0_min


                l0_vec = np.zeros_like(self.default_l0)
                l0_vec[dim] = l0_max - l0_min
                l0_norm = np.linalg.norm(l0_vec)
                l0_unit_vec = l0_vec / l0_norm


                def contour_ode_func(t, C_reduced):
                    return sim.dC_func(
                        t, C_reduced,
                        rates,
                        l0_unit_vec,
                        l0_current,
                        dR_dC_func,
                        dR_dl_func
                    )
                
  
                t_eval = self.l0_grid - l0_min                              
                # Perform contour integration
                sol = solve_ivp(
                    contour_ode_func,
                    t_span=(0, l0_norm),
                    y0=C_reduced_steady,
                    t_eval=t_eval,
                    method=int_method, 
                    rtol=r_tol,
                    atol=a_tol
                )
                
                if self.use_signal_alarms:
                    signal.alarm(0)  # Cancel timeout
                
                if self.profiler:
                    self.profiler.end_timer("contour_integration")
                    self.profiler.start_timer("contour_processing")
                
                # Process results at each grid point
                sign_conditions = []
                C_full_list = []
                dC_dl_list = []
                l0_list = []
                
                for i, t_val in enumerate(t_eval):
                    # Get l0 value at this grid point
                    l0_current = self.default_l0.copy()
                    l0_current[dim] = self.l0_grid[i]
                    
                    # Get reduced concentrations at this point
                    C_reduced_current = sol.y[:, i]
                    
                    # Recover full species concentrations
                    C_full_current = sim.recover_eliminated_species(l0_current, C_reduced_current)
                    
                    # Compute sensitivity derivatives
                    dC_dl = sim.dC_dl_func(C_reduced_current, l0_current, rates, dR_dC_func, dR_dl_func)
                    dC_dl_full = sim.compute_dC_dk_full(dC_dl, l_bool=True)
                    
                    # Extract sign conditions
                    if self.sc_grad_dims is not None:
                        # Use only specified dimensions of dC_dl_full
                        rows, cols = self.sc_grad_dims
                        dC_dl_subset = dC_dl_full[np.ix_(rows, cols)]
                        signs = np.sign(np.round(dC_dl_subset, decimals=self.round_decimals)).tolist()
                    else:
                        # Use all dimensions (original behavior)
                        signs = np.sign(np.round(dC_dl_full, decimals=self.round_decimals)).tolist()
                    
                    # Store results
                    sign_conditions.append(signs)
                    C_full_list.append(C_full_current)
                    dC_dl_list.append(dC_dl_full)
                    l0_list.append(l0_current)
                    self.sample_count += 1
                
                if self.profiler:
                    self.profiler.end_timer("contour_processing")
                
                # Store results in class attributes
                self.sign_conditions = sign_conditions
                self.C_full_list = C_full_list
                self.dC_dl_list = dC_dl_list
                self.l0_list = l0_list
                
                return sign_conditions, C_full_list, dC_dl_list, l0_list, self.sample_count, True
                
            except TimeoutError:
                if self.use_signal_alarms:
                    signal.alarm(0)  # Cancel timeout
                if self.profiler:
                    self.profiler.end_timer("contour_integration")
                return [], [], [], [], 0, False
                
        except Exception as e:
            print(f"Error in contour integration: {e}")
            if self.profiler:
                # End any active timers
                for timer_name in ["contour_nnls", "contour_initial_conditions", "contour_steady_state", "contour_integration", "contour_processing"]:
                    if timer_name in self.profiler.timers and self.profiler.timers[timer_name]['start_time'] is not None:
                        self.profiler.end_timer(timer_name)
            return [], [], [], [], 0, False
        
    # def _get_sign_sample_contour(self, l0_init, C_reduced_init, l0, sim, L, t_span, num_points, int_method, r_tol, a_tol, precomputed_derivatives):
    #     """
    #     Generate a single sign sample with all necessary computations.
        
    #     Args:
    #         sim: ReactionNetworkSimulator instance
    #         L: Conservation law matrix
    #         t_span: Integration time span
    #         num_points: Number of integration points
    #         int_method: Integration method
    #         r_tol: Relative tolerance
    #         a_tol: Absolute tolerance
    #         precomputed_derivatives: Precomputed derivative functions (dR_dC_func, dR_dl_func, etc.)
            
    #     Returns:
    #         tuple: (signs, C_full, success) where success is boolean
    #     """
    #     try:
            
    #         if self.profiler:
    #             self.profiler.start_timer("adaptive_nnls")
            
    #         l0_vec = l0 - l0_init
    #         l0_norm = np.linalg.norm(l0_vec)
    #         l0_unit_vec = l0_vec / l0_norm
            
    #         if self.profiler:
    #             self.profiler.end_timer("adaptive_nnls")
    #             self.profiler.start_timer("adaptive_integration")
            
            
    #         # Set up timeout for integration
    #         if self.use_signal_alarms:
    #             signal.signal(signal.SIGALRM, timeout_handler)
    #             signal.alarm(self.timeout_seconds)

    #         dR_dC_func, dR_dl_func = precomputed_derivatives
    #         rates = np.array([sim.r_n.reactions[r_idx][2] for r_idx in range(len(sim.r_n.reactions))])

    #         def ode_func(t, C):
    #             return sim.dC_func(
    #                 t, C,
    #                 rates,
    #                 l0_unit_vec,
    #                 l0_init,
    #                 dR_dC_func,
    #                 dR_dl_func
    #             )

            
    #         try:
    #             sol = solve_ivp(
    #             ode_func,
    #             t_span=(0, l0_norm),
    #             y0=C_reduced_init,
    #             method=int_method, 
    #             rtol=r_tol,
    #             atol=a_tol
    #             )
                
    #             if self.use_signal_alarms:
    #                 signal.alarm(0)  # Cancel timeout
                
    #             if self.profiler:
    #                 self.profiler.end_timer("adaptive_integration")
    #                 self.profiler.start_timer("adaptive_species_recovery")
                
    #             C_reduced_final = sol.y[:, -1]

    #             # Recover full species concentrations
    #             C_full = sim.recover_eliminated_species(l0, C_reduced_final)
                
    #             if self.profiler:
    #                 self.profiler.end_timer("adaptive_species_recovery")
    #                 self.profiler.start_timer("adaptive_sensitivity_analysis")
                
    #             # Compute sensitivity derivatives (use precomputed if available)
    #             rates = np.array([sim.r_n.reactions[r_idx][2] for r_idx in range(len(sim.r_n.reactions))])
                
                
    #             dC_dl = sim.dC_dl_func(C_reduced_final, l0, rates, dR_dC_func, dR_dl_func)
    #             dC_dl_full = sim.compute_dC_dk_full(dC_dl, l_bool = True)
                
    #             if self.profiler:
    #                 self.profiler.end_timer("adaptive_sensitivity_analysis")
    #                 self.profiler.start_timer("adaptive_sign_processing")
                
    #             # Extract sign conditions
    #             if self.sc_grad_dims is not None:
    #                 # Use only specified dimensions of dC_dl_full
    #                 rows, cols = self.sc_grad_dims
    #                 dC_dl_subset = dC_dl_full[np.ix_(rows, cols)]
    #                 signs = np.sign(np.round(dC_dl_subset, decimals=self.round_decimals)).tolist()
    #             else:
    #                 # Use all dimensions (original behavior)
    #                 signs = np.sign(np.round(dC_dl_full, decimals=self.round_decimals)).tolist()
                
    #             if self.profiler:
    #                 self.profiler.end_timer("adaptive_sign_processing")
                
    #             return signs, C_full, l0, True
                
    #         except TimeoutError:
    #             if self.use_signal_alarms:
    #                 signal.alarm(0)  # Cancel timeout
    #             if self.profiler:
    #                 self.profiler.end_timer("adaptive_integration")
    #             return None, None, None, False
                
    #     except Exception as e:
    #         print(e)
    #         if self.profiler:
    #             # End any active timers
    #             for timer_name in ["adaptive_nnls", "adaptive_initial_conditions", "adaptive_integration", "adaptive_species_recovery", "adaptive_sensitivity_analysis", "adaptive_sign_processing"]:
    #                 if timer_name in self.profiler.timers and self.profiler.timers[timer_name]['start_time'] is not None:
    #                     self.profiler.end_timer(timer_name)
    #         return None, None, None, False
    
    def sample_sign_conditions(self, sim, L, t_span, num_points, int_method, r_tol, a_tol, precomputed_derivatives):
        """
        Perform grid sampling of sign conditions over specified input dimensions.
        
        Args:
            sim: ReactionNetworkSimulator instance
            L: Conservation law matrix
            t_span: Integration time span
            num_points: Number of integration points
            int_method: Integration method
            r_tol: Relative tolerance
            a_tol: Absolute tolerance
            precomputed_derivatives: Precomputed derivative functions (dR_dC_func, dR_dl_func, etc.)
            
        Returns:
            tuple: (sign_conditions, C_full_list, dC_dl_list, l0_list, sample_count, convergence_reached)
        """
        
        # Check if we should use contour integration for 1D case
        if self.use_contour_integration:
            if len(self.input_dims) == 1:
                return self._sample_sign_conditions_contour_1d(sim, L, t_span, num_points, int_method, r_tol, a_tol, precomputed_derivatives)
            else:
                print("Warning: Contour integration is only supported for 1D parameter spaces. Falling back to grid sampling.")
        
        # Create grid over the specified input dimensions
        l0_grid_iter = itertools.product(*[self.l0_grid] * len(self.input_dims))
        
        for l0_grid_vals in l0_grid_iter:
            # Start with default values for all dimensions
            l0 = self.default_l0.copy()
            
            # Set grid values for the specified input dimensions
            for i, dim in enumerate(self.input_dims):
                l0[dim] = l0_grid_vals[i]
            
            # Generate sign sample
            signs, C_full, dC_dl_full, l0, success = self._get_sign_sample(l0, sim, L, t_span, num_points, int_method, r_tol, a_tol, precomputed_derivatives)
            
            if not success:
                continue

            self.sample_count += 1
            
            # # Check if this is a new sign condition
            # signs_str = str(signs)
            # is_new = True
            # for existing_signs in self.sign_conditions:
            #     if str(existing_signs) == signs_str:
            #         is_new = False
            #         break

            self.sign_conditions.append(signs)
            self.C_full_list.append(C_full)
            self.dC_dl_list.append(dC_dl_full)
            self.l0_list.append(l0)
        
        return self.sign_conditions, self.C_full_list, self.dC_dl_list, self.l0_list, self.sample_count, True
    
    def reset(self):
        """Reset the sampler state for reuse."""
        self.sign_conditions = []
        self.C_full_list = []
        self.dC_dl_list = []
        self.l0_list = []
        self.sample_count = 0
    
    def get_statistics(self):
        """Get sampling statistics."""
        unique_sign_conditions = len(set(str(s) for s in self.sign_conditions))
        
        return {
            'total_samples': self.sample_count,
            'unique_sign_conditions': unique_sign_conditions
        }