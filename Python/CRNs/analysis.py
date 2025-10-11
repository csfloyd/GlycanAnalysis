"""
Analysis module for reaction networks.

This module contains functions for analyzing reaction networks, including
conservation law computation and validation.
"""

import numpy as np
import sympy
import itertools
from typing import List, Dict, Tuple


def count_conservation_group_changes(r_n):
    # Get conservation groups
    conservation_groups = r_n.get_conservation_groups()

    M_0t1 = 0
    M_1t0 = 0 
    M_0b1 = 0 # symmetric

    # Loop through reactions and check conservation group membership
    for i, (c1_idx, c2_idx, _) in enumerate(r_n.reactions):
        c1 = r_n.all_complexes[c1_idx]
        c2 = r_n.all_complexes[c2_idx]
        
        for pairing_group in r_n.within_pairing_groups:
            if c1 in pairing_group and c2 in pairing_group.keys():
                break

        for pairing_group in r_n.between_pairing_groups:
            if c1 in pairing_group and c2 in pairing_group.keys():
                
                # Check stoichiometry changes for each conservation group
                cg_change_bool_vec = []
                for group_idx, group in enumerate(conservation_groups):
                    # Count species from this group on each side
                    # Get species in this group from each side
                    c1_species = set(species for species in (c1.split('+') if len(c1)==3 else [c1]) if species in group)
                    c2_species = set(species for species in (c2.split('+') if len(c2)==3 else [c2]) if species in group)
                    # If counts differ, this group has a net change
                    cg_change_bool_vec.append(c1_species != c2_species)
                
                if cg_change_bool_vec == [True, False]:
                    M_1t0 += 1
                elif cg_change_bool_vec == [False, True]:
                    M_0t1 += 1
                elif cg_change_bool_vec == [True, True]:
                    M_0b1 += 1
    
                break

    return M_0t1, M_1t0, M_0b1

def find_sublists_containing_element(big_list, target_element):
    """Return indices of sublists containing the target element."""
    return [i for i, sublist in enumerate(big_list) if target_element in sublist]

def get_interaction_matrix(r_n):
    # Get conservation groups
    conservation_groups = r_n.get_conservation_groups()
    interaction_matrix = np.zeros((len(conservation_groups), len(conservation_groups)))
    S = r_n.get_stoichiometric_matrix()
    species_names = r_n.species_names

    # Loop through reactions and check conservation group membership
    for i, (cr_idx, cp_idx, _) in enumerate(r_n.reactions):
        cr = r_n.all_complexes[cr_idx]
        cp = r_n.all_complexes[cp_idx]

        cr_species = cr.split('+') if '+' in cr else [cr]
        cp_species = cp.split('+') if '+' in cp else [cp]
        #print(cr_species, cp_species)

        # Get species involved (non-zero elements)
        stoich_change_speices = [species_names[j] for j in range(len(species_names)) if S[j,i] != 0]
        lhs_species = cr_species 

        up_stream_indices = list(set().union(*[find_sublists_containing_element(conservation_groups, lhs) for lhs in lhs_species]))
        down_stream_indices = list(set().union(*[find_sublists_containing_element(conservation_groups, stoich) for stoich in stoich_change_speices]))

        # print(up_stream_indices, down_stream_indices)

        for k in up_stream_indices:
            for l in down_stream_indices:
                if k != l:
                    interaction_matrix[k,l] += 1

    return interaction_matrix.T

def compute_cycles(r_n):
        """
        Compute cycles in the reaction network by finding the nullspace of the stoichiometric matrix.
        
        Returns:
            Z: Matrix of cycles
            cycles: List of cycles where each cycle is a list of reaction strings
        """
        """
        Compute cycles in the reaction network by finding the nullspace of the stoichiometric matrix.
        
        Args:
            r_n: ReactionNetwork instance
            force_reverse: Boolean indicating if reactions are forced to be reversible
            
        Returns:
            Z: Matrix of cycles
            cycles: List of cycles where each cycle is a list of reaction strings
        """
        S = r_n.get_stoichiometric_matrix()
 
        if r_n.force_reverse:
            S_red = S[:,0:-1:2] 
        else:
            S_red = S

        ns = sympy.Matrix(S_red).nullspace()  # list of column vectors
        Z = sympy.Matrix.hstack(*ns) if ns else sympy.Matrix.zeros(S_red.shape[1], 0)
        
        
        cycles = []
        reaction_strings = r_n.get_reaction_strings(include_rates=False)
        
        for col in range(Z.shape[1]):
            cycle = Z[:,col]
            cycle_reactions = []
            for ind in range(len(cycle)):
                if cycle[ind] == 1:
                    cycle_reactions.append(reaction_strings[2*ind])
                if cycle[ind] == -1:
                    cycle_reactions.append(reaction_strings[2*ind+1])
            cycles.append(cycle_reactions)
            
        return Z, cycles

###################################################
########### Results analysis functions ############
###################################################


def calculate_finite_differences(y, x, log_y=False, log_x=False):
    """
    Calculate finite difference approximation of dy/dx using central difference.
    Optional log transformation of x and/or y before calculating differences.
    
    Args:
        y: List of y values
        x: List of x values
        log_y: Boolean indicating whether to take log10 of y values
        log_x: Boolean indicating whether to take log10 of x values
        
    Returns:
        List of finite difference approximations
    """
    d_vals_fd = []
    
    # Transform values if needed
    x_vals = np.log10(x) if log_x else x
    y_vals = np.log10(y) if log_y else y
    
    # First point uses forward difference
    dy = y_vals[1] - y_vals[0]
    dx = x_vals[1] - x_vals[0]
    d_vals_fd.append(dy/dx)
    
    # Central difference for middle points
    for i in range(1, len(x)-1):
        dy = y_vals[i+1] - y_vals[i-1]
        dx = x_vals[i+1] - x_vals[i-1]
        d_vals_fd.append(dy/dx)
        
    # Last point uses backward difference
    dy = y_vals[-1] - y_vals[-2]
    dx = x_vals[-1] - x_vals[-2]
    d_vals_fd.append(dy/dx)
    
    return d_vals_fd

def check_list_consistency(base_list, comparison_list, threshold=0.1, epsilon=0, pad = 0):
    # Convert lists to numpy arrays for vectorized operations
    base_arr = np.array(base_list)
    comp_arr = np.array(comparison_list)
    
    # Calculate differences ratio array
    diff_ratios = np.abs(comp_arr[pad:-pad] - base_arr[pad:-pad])/(np.max(np.abs(base_arr[pad:-pad]) + epsilon))
    
    # Check if any ratio exceeds threshold
    return not (diff_ratios > threshold).any()

def normalized_arc_length(x_vals, y_vals):
    """
    Compute the range-normalized arc length of a 1D function sampled at (x_vals, y_vals).

    Normalization:
        - y is linearly rescaled to [0, 1]
        - x is assumed monotonic (domain scaling is implicit)
        - returns arc length normalized by domain length (minus 1 for convenience)
    """
    x = np.asarray(x_vals)
    y = np.asarray(y_vals)

    if x.size < 2:
        return 0.0
    # Segment-wise arc length
    dx = np.diff(x)
    dy = np.diff(y)
    segment_lengths = np.sqrt(dx**2 + dy**2)
    L = np.sum(segment_lengths)

    # Straight-line chord between endpoints (in the same normalized coordinate system)
    chord_dx = x[-1] - x[0]
    chord_dy = y[-1] - y[0]
    chord_length = np.hypot(chord_dx, chord_dy)

    return L / chord_length





import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def fit_mlp(x_data, y_data, hidden_layer_sizes, normalize_x=True, random_state=42):
    """
    Fit a multilayer perceptron to the data
    """
    x_data = np.array(x_data).reshape(-1, 1)
    y_data = np.array(y_data)
    
    if normalize_x:
        x_min, x_max = x_data.min(), x_data.max()
        x_norm = (x_data - x_min) / (x_max - x_min)
    else:
        x_norm = x_data
        x_min, x_max = None, None
    
    # Standardize the data
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    x_scaled = scaler_x.fit_transform(x_norm)
    y_scaled = scaler_y.fit_transform(y_data.reshape(-1, 1)).ravel()
    
    
    mlp = MLPRegressor(
    solver='lbfgs',  # Much faster than Adam for small data
    activation='tanh',
    hidden_layer_sizes=hidden_layer_sizes,
    max_iter=2000,
    # Remove early stopping for speed
    early_stopping=False,
    validation_fraction=0.1,
    n_iter_no_change=20
)
    
    # Fit the model
    mlp.fit(x_scaled, y_scaled)
    
    # Predict on training data
    y_pred_scaled = mlp.predict(x_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    
    # Calculate various scaled metrics
    residuals = y_data - y_pred
    ss_res = np.sum(residuals**2)
    mse = np.mean(residuals**2)
    rmse = np.sqrt(mse)
    ss_tot = np.sum((y_data - np.mean(y_data))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Scaled metrics
    data_variance = np.var(y_data)
    data_range = y_data.max() - y_data.min()
    normalized_ss_res = ss_res / data_variance
    normalized_mse = mse / data_variance
    cv_residuals = np.std(residuals) / np.mean(np.abs(y_data))
    
    # Count parameters (degrees of freedom)
    n_params = 0
    for i in range(len(mlp.coefs_)):
        n_params += mlp.coefs_[i].size  # weights
        n_params += mlp.intercepts_[i].size  # biases
    
    return {
        'mlp': mlp,
        'scaler_x': scaler_x,
        'scaler_y': scaler_y,
        'y_predicted': y_pred,
        'residuals': residuals,
        'r_squared': r_squared,
        'ss_res': ss_res,
        'mse': mse,
        'rmse': rmse,
        'normalized_ss_res': normalized_ss_res,
        'normalized_mse': normalized_mse,
        'cv_residuals': cv_residuals,
        'n_params': n_params,
        'hidden_layer_sizes': hidden_layer_sizes,
        'x_min': x_min,
        'x_max': x_max
    }
    

def scan_mlp_widths(x_data, y_data, max_width=20, normalize_x=True):
    """
    Scan over different hidden layer widths
    """
    widths = range(1, max_width + 1)
    results = {}
    squared_residuals = []
    r_squared_values = []
    n_params_list = []
    
    for width in widths:
        try:
            result = fit_mlp(x_data, y_data, (width,), normalize_x)
            results[width] = result
            squared_residuals.append(result['ss_res'])
            r_squared_values.append(result['r_squared'])
            n_params_list.append(result['n_params'])
            print(f"MLP width {width}: SS_res = {result['ss_res']:.6f}, R² = {result['r_squared']:.4f}, params = {result['n_params']}")
        except Exception as e:
            print(f"MLP width {width} failed: {e}")
            squared_residuals.append(np.inf)
            r_squared_values.append(0)
            n_params_list.append(0)
    
    return results, squared_residuals, r_squared_values, n_params_list, widths

def plot_mlp_fits(x_data, y_data, results, max_show=9):
    """
    Plot grid of MLP fits
    """
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    
    n_results = min(len(results), max_show)
    n_cols = 3
    n_rows = (n_results + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, 5 * n_rows))
    
    for i, (width, result) in enumerate(list(results.items())[:max_show]):
        plt.subplot(n_rows, n_cols, i + 1)
        
        plt.scatter(x_data, y_data, alpha=0.6, color='blue', s=20, label='Data')
        
        # Plot fitted curve
        x_plot = np.linspace(x_data.min(), x_data.max(), 1000).reshape(-1, 1)
        
        if result['x_min'] is not None:
            x_plot_norm = (x_plot - result['x_min']) / (result['x_max'] - result['x_min'])
        else:
            x_plot_norm = x_plot
        
        x_plot_scaled = result['scaler_x'].transform(x_plot_norm)
        y_plot_scaled = result['mlp'].predict(x_plot_scaled)
        y_plot = result['scaler_y'].inverse_transform(y_plot_scaled.reshape(-1, 1)).ravel()
        
        plt.plot(x_plot.ravel(), y_plot, 'r-', linewidth=2, label=f'MLP width {width}')
        
        plt.title(f'MLP Width {width}\nR² = {result["r_squared"]:.3f}, Params = {result["n_params"]}')
        plt.grid(True, alpha=0.3)
        if i == 0:
            plt.legend(fontsize=8)
    
    plt.tight_layout()
    plt.show()


def count_critical_points(network_data, target_node_idx=None, l0_list=None, fd_comparison = False, threshold=0.2, pad=4):
    """
    Count critical points (sign changes) in derivatives for a given network.
    
    Parameters:
    network_data: Dictionary containing network data with keys:
                  - 'dC_dl_list': List of derivatives dC/dl0
                  - 'C_full_list': List of concentration values
                  - 'l0_list': List of l0 values (optional, can be provided separately)
                  - 'network_params': Network parameters including 'species_names' and 'NS'
    target_node_idx: Index of target node to analyze (default: last species, NS-1)
    l0_list: Optional list of l0 values (overrides network_data['l0_list'])
    threshold: Threshold for finite difference consistency check
    pad: Number of points to skip at beginning and end for consistency check
    
    Returns:
    int: Number of sign changes (critical points) if consistent, None if inconsistent
    """
    try:
        # Extract data from network_data
        d_C_d_l0_list = network_data['dC_dl_list']
        C_full_list = network_data['C_full_list']
        
        # Get l0_list from provided argument or network_data
        if l0_list is None:
            l0_list = network_data.get('l0_list')
            if l0_list is None:
                raise ValueError("l0_list must be provided either as argument or in network_data")
        
        # Determine target node index
        if target_node_idx is None:
            # Default to last species
            NS = network_data['network_params']['NS']
            target_node_idx = NS - 1
        
        # Extract derivative values for target node
        # Assuming d_C_d_l0_list contains derivatives for all species
        # If it's already a 1D list, use it directly; otherwise extract target species
        if isinstance(d_C_d_l0_list[0], (list, np.ndarray)) and len(np.shape(d_C_d_l0_list[0])) > 0:
            # Multi-dimensional: extract target node
            d_vals = [d_C_d_l0_list[i][target_node_idx] if isinstance(d_C_d_l0_list[i][target_node_idx], (int, float, np.number)) 
                     else d_C_d_l0_list[i][target_node_idx][0] 
                     for i in range(len(d_C_d_l0_list))]
        else:
            # Already 1D
            d_vals = d_C_d_l0_list
        
        # Extract concentration values for target node
        if isinstance(C_full_list[0], (list, np.ndarray)) and len(np.shape(C_full_list[0])) > 0:
            C_vals = [C_full_list[i][target_node_idx] if isinstance(C_full_list[i][target_node_idx], (int, float, np.number))
                     else C_full_list[i][target_node_idx][0]
                     for i in range(len(C_full_list))]
        else:
            C_vals = C_full_list
            
        # Create log scale x-axis
        log_l0_x = [np.log10(l0[0]) if isinstance(l0, (list, np.ndarray)) else np.log10(l0) 
                for l0 in l0_list]
        
        # Calculate finite differences for consistency check
        if fd_comparison:
            d_vals_fd = calculate_finite_differences(C_vals, log_l0_x, log_y=False, log_x=True)
            if not check_list_consistency(d_vals_fd, d_vals, threshold=threshold, pad=pad):
                print("Inconsistent derivatives found")
                return None, None  # Inconsistent derivatives
        
        max_log_deriv = np.max(np.abs(d_vals * np.array(log_l0_x)))

        # Find sign changes
        sign_change_indices = []
        for i in range(len(d_vals) - 1):
            if d_vals[i] * d_vals[i + 1] < 0:  # Sign change occurs
                sign_change_indices.append(i)
        
        return (len(sign_change_indices), max_log_deriv)
        
    except Exception as e:
        print(f"Error in count_critical_points: {e}")
        return None, None


def select_best_mlp_width(x_vals, y_vals, width_range=(2, 20), normalize_x=True, random_state=42, r2_threshold = 0.99, quiet = True):
    """
    Select the smallest MLP width that best fits the data
    
    Parameters:
    x_vals: input data
    y_vals: output data
    width_range: tuple of (min_width, max_width)
    normalize_x: whether to normalize x to [0,1]
    random_state: random seed for reproducibility
    
    Returns:
    dict with best_width, metrics, and all results
    """
    
    def fit_mlp_fast(x_data, y_data, hidden_layer_sizes, normalize_x=True, random_state=42):
        """Fast MLP fitting function with proper scaling"""
        x_data = np.array(x_data).reshape(-1, 1)
        y_data = np.array(y_data)
        
        if normalize_x:
            x_min, x_max = x_data.min(), x_data.max()
            x_norm = (x_data - x_min) / (x_max - x_min)
        else:
            x_norm = x_data
            x_min, x_max = None, None
        
        # Use StandardScaler for proper scaling
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        
        x_scaled = scaler_x.fit_transform(x_norm)
        y_scaled = scaler_y.fit_transform(y_data.reshape(-1, 1)).ravel()
        
        mlp = MLPRegressor(
            solver='lbfgs',  # Much faster than Adam for small data
            activation='tanh',
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=2000,
            # Remove early stopping for speed
            early_stopping=False,
            validation_fraction=0.1,
            n_iter_no_change=5, 
            random_state=random_state
        )
        
        mlp.fit(x_scaled, y_scaled)
        y_pred_scaled = mlp.predict(x_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        residuals = y_data - y_pred
        mse = np.mean(residuals**2)
        rmse = np.sqrt(mse)
        ss_tot = np.sum((y_data - np.mean(y_data))**2)
        r_squared = 1 - (np.sum(residuals**2) / ss_tot)
        
        n_params = sum(coef.size for coef in mlp.coefs_) + sum(intercept.size for intercept in mlp.intercepts_)
        
        return {
            'mlp': mlp,
            'scaler_x': scaler_x,
            'scaler_y': scaler_y,
            'y_predicted': y_pred,
            'r_squared': r_squared,
            'mse': mse,
            'rmse': rmse,
            'n_params': n_params,
            'x_min': x_min,
            'x_max': x_max
        }
    
    # Scan over widths
    min_width, max_width = width_range
    widths = range(min_width, max_width + 1)
    results = {}
    metrics = []
    
    if not quiet:
        print(f"Scanning MLP widths from {min_width} to {max_width}...")
    
    for width in widths:
        try:
            result = fit_mlp_fast(x_vals, y_vals, (width,), normalize_x, random_state)
            results[width] = result
            
            metrics.append({
                'width': width,
                'r_squared': result['r_squared'],
                'mse': result['mse'],
                'rmse': result['rmse'],
                'n_params': result['n_params']
            })
            
            if not quiet:
                print(f"Width {width}: R² = {result['r_squared']:.4f}, MSE = {result['mse']:.6f}, Params = {result['n_params']}")
            
        except Exception as e:
            if not quiet:
                print(f"Width {width} failed: {e}")
            metrics.append({
                'width': width,
                'r_squared': 0,
                'mse': np.inf,
                'rmse': np.inf,
                'n_params': 0
            })
    
    # Find best width using R² threshold
    valid_metrics = [m for m in metrics if m['mse'] != np.inf]
    
    if not valid_metrics:
        raise ValueError("No valid fits found!")
    
    # Find smallest width meeting R² threshold
    candidates = [m for m in valid_metrics if m['r_squared'] >= r2_threshold]
    
    if candidates:
        best_width = min(candidates, key=lambda x: x['width'])['width']
    else:
        # Fallback to best R² if no width meets threshold
        best_r2_idx = np.argmax([m['r_squared'] for m in valid_metrics])
        best_width = valid_metrics[best_r2_idx]['width']
    
    # Summary
    summary = {
        'best_width': best_width,
        'r2_threshold': r2_threshold,
        'threshold_met': len(candidates) > 0,
        'candidates_count': len(candidates)
    }
    
    if not quiet:
        print(f"\n=== Results Summary ===")
        print(f"R² threshold: {r2_threshold}")
        print(f"Best width meeting threshold: {best_width}")
        print(f"Threshold met: {len(candidates) > 0}")
        print(f"Number of candidates: {len(candidates)}")
    
    return {
        'best_width': best_width,
        'summary': summary,
        'all_metrics': metrics,
        'all_results': results,
        'recommended_result': results[best_width] if best_width in results else None
    }
