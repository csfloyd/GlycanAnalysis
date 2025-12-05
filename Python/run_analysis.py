from CRNs import *
from CRNs.utils import NetworkDataLogger
import numpy as np
import argparse
import time
import copy
import pickle
import ast
import os
import glob

#analysis_function = "arc_length"
analysis_function = "total_absolute_curvature"
#analysis_function = "critical_points"
#analysis_function = "mlp_width"
#analysis_function = "pca"

# Create argument parser
parser = argparse.ArgumentParser(description="Analyze existing data from input directory and save results to output directory.")

arg1_type = str
# Define command-line arguments
parser.add_argument("--param1", type=arg1_type, required=True, help="An integer parameter")
parser.add_argument("--param2", type=int, required=False, help="An integer parameter")
parser.add_argument("--input", type=str, required=True, help="Input directory containing existing data")
parser.add_argument("--output", type=str, required=True, help="Output directory for analysis results")

# Parse arguments
args = parser.parse_args()

input_dir = args.input
output_dir = args.output

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Initialize results logger
results_logger = NetworkDataLogger()

# Load input data
data_logger = NetworkDataLogger(input_dir + "/SavedData.pkl")
n_networks = len(data_logger.network_data)

print(f"Found {n_networks} networks to analyze")

whole_data_functions = ["pca"]

time_start = time.time()
if analysis_function in whole_data_functions:
    (target_node, target_node_idx, n_paths) = count_paths_to_target(data_logger, 0, source_node='R0')
    network_data = data_logger.get_network_by_index(0)
    species_names = network_data['network_params']['species_names']
    
    analysis_functions = {
        "pca": lambda: fit_pca_on_networks(data_logger, n_points=100, n_samples=50000, target_species_idx = species_names.index(target_node))
    }

    result = analysis_functions[analysis_function]()
    results_logger.log_network(
        network_index=0,
        analysis_type=analysis_function,
        result_value=result,
        param1=args.param1,
        param2=args.param2 if args.param2 is not None else None
    )

else:
    for network_index in range(n_networks):
        if network_index % 100 == 0:
            print(f"Analyzing network {network_index + 1}/{n_networks}")
        
        network_data = data_logger.get_network_by_index(network_index)
        network_params = network_data['network_params']
        species_names = network_params['species_names']
        NR =int(sum('R' in name for name in species_names))
        NS =int(sum('S' in name for name in species_names) // 2)
        target_node = 'S'+str(NS-1)
        (target_node, target_node_idx, n_paths) = count_paths_to_target(data_logger, network_index, source_node='R0')
        adjacency_matrix = network_data['adjacency_matrix']
        input_substrates_list = network_data['input_substrates_list']
        C_full_list = network_data['C_full_list']
        l0_list = network_data['l0_list']
        log_l0_x = [np.log10(l0[0]) for l0 in l0_list]
        
        # Perform analysis based on the specified function
        # Map analysis functions to their implementations
        analysis_functions = {
            "arc_length": lambda: normalized_arc_length(log_l0_x, C_full_list),
            "total_absolute_curvature": lambda: total_absolute_curvatures(log_l0_x, C_full_list),
            "num_sign_changes": lambda: count_conservation_group_changes(network_data),
            "mlp_width": lambda: select_best_mlp_width(log_l0_x, C_full_list, width_range=(2, 10), normalize_x=True, random_state=42, r2_threshold = 0.99, quiet = True)['best_width'],
            "sign_conditions": lambda: count_sign_conditions(network_data),
            "critical_points": lambda: count_critical_points(network_data, target_node_idx = species_names.index(target_node), l0_list = l0_list, fd_comparison = False, eps = 1e-10) 
        }

        #if analysis_function in analysis_functions:
        result = analysis_functions[analysis_function]()
        results_logger.log_network(
            network_index=network_index,
            analysis_type=analysis_function,
            result_value=result,
            param1=args.param1,
            param2=args.param2 if args.param2 is not None else None
        )
        
time_end = time.time()
print(f"Time taken: {time_end - time_start} seconds")

# save one network copy
results_logger.log_network(**data_logger.get_network_by_index(0))

# Save analysis results
results_file = os.path.join(output_dir,analysis_function + ".pkl")
results_logger.save_data(results_file)
print(f"\nAnalysis complete! Results saved to: {results_file}")

