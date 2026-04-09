from CRNs import *
from CRNs.utils import NetworkDataLogger
import numpy as np
import argparse
import time
import copy
import pickle
import ast


# Create argument parser
parser = argparse.ArgumentParser(description="SLURM job script with arguments.")

arg1_type = int
# Define command-line arguments
parser.add_argument("--param1", type=int, required=True, help="An integer parameter")
parser.add_argument("--param2", type=float, required=False, help="An integer parameter")
parser.add_argument("--param3", type=int, required=False, help="An integer parameter")
parser.add_argument("--output", type=str, required=True, help="A string parameter")

# Parse arguments
args = parser.parse_args()

output_dir = args.output


seed = args.param3
n_classes = 5
input_dim = 10
center_variance = 10.0
log_variance = 1
proj_dim = args.param1
NR = proj_dim
hidden_dim = 5 #args.param2


num_batches=1000
batch_size=25
T_start=0.1
T_end=0.1
T_decay=0.99
noise_start=0.0
noise_end=0.0
noise_decay=0.99
print_every=250

np.random.seed(seed)
random.seed(seed)

#################################################
############### Data generation #################
#################################################


n_classes, data_list, log_means = generate_lognormal_mixture_random_centers(
    n_classes=n_classes,
    n_samples_per_class=10000,
    input_dim=input_dim,
    center_variance=center_variance,   # Total variance spread across dimensions
    log_variance=log_variance,      # How tight each class cluster is
    center_offset=0.0,     # Shift so medians are around exp(4) ≈ 55
    random_state=seed
)


data_list = project_data_list(data_list, d=proj_dim)
input_data = InputData(n_classes, data_list)


#################################################
############### Data generation #################
#################################################


NS_vec = [hidden_dim,n_classes]
NS = sum(NS_vec)
input_nodes = [f'R{i}' for i in range(NR)]
target_nodes = [f'S{NS - n_classes + i}' for i in range(n_classes)]

p_r = 0.0
p_f = args.param2
species_names, reaction_strings, L, adjacency_matrix, input_substrates_list = generate_layered_feedforward_signaling_network(NR, NS_vec, p_r, p_f = p_f, include_reverse=False, include_uncatalyzed=True)

### add recurrences -comment out if not using
# reaction_strings, adjacency_matrix = add_recurrent_connections(adjacency_matrix, reaction_strings, False, input_substrates_list, NR, NS, 1.0, seed=seed)

target_node_idxs = [species_names.index(node) for node in target_nodes]

G = get_digraph_from_adjacency_matrix(adjacency_matrix, input_substrates_list, NR, NS)
n_paths = count_simple_paths(G, f'R0', f'S{NS-1}')


################################################
############### Create network #################
################################################

r_n = ReactionNetwork.from_reaction_strings(
    reaction_strings=reaction_strings,
    L=L,
    seed=seed,
    species_names=species_names,
    force_reverse=True
)

sim = ReactionNetworkSimulator(r_n)
symbolic_rhs, species, rates = sim.get_symbolic_rhs()
reduced_rhs, remaining_syms, const_syms, rate_syms = sim.get_symbolic_reduced_rhs()
dR_dC, dR_dC_func, dR_dl, dR_dl_func, dR_dk, dR_dk_func, remaining_syms = sim.get_first_order_derivatives()
sim.solve_conservation_laws()
rates_orig = np.array([r_n.reactions[r_idx][2] for r_idx in range(len(r_n.reactions))])


###############################################
############### Train network #################
###############################################

n_rates = len(r_n.get_rates())
rates = np.exp(np.random.randn(n_rates) * 1.0)
r_n.update_rates(rates)

# ============== GRAPH COMPUTATION ==============
use_graph_forward = True  # Set False to use ODE integration instead
if use_graph_forward:
    graph_comp = GraphComputation(G, input_nodes, target_nodes)
    graph_comp.build_r_n_maps(r_n)
    n_nodes = len(graph_comp.nodes)
    forward_method = 'graph'
else:
    n_nodes = len(L)
    graph_comp = None
    forward_method = 'ode'

default_l0 = np.ones(n_nodes)
# Target node indices in species list
target_node_idxs = [r_n.species_names.index(node) for node in target_nodes]
n_inputs = len(input_nodes)

# ============== MODEL ==============
crn_model = CRNModel(
    r_n=r_n,
    sim=sim,
    L=L,
    class_ids=target_node_idxs,
    n_inputs=n_inputs,
    default_l0=default_l0,
    forward_method=forward_method,
    graph_comp=graph_comp,
    dR_dC_func=dR_dC_func,
    dR_dk_func=dR_dk_func,
    dR_dl_func=dR_dl_func,
    generate_init_func=generate_positive_initial_concentrations_nnls,
)

# ============== TRAINER ==============
trainer = UnifiedTrainer(
    crn_model,
    optimizer_type='adam',
    lr_dict={'log_rates': 0.005, 'log_l0': 0.005},
    max_grad_norm=50.0,
    frozen_params=['log_l0']
)

# ============== TRAIN ==============
history = run_training_crn(
    trainer=trainer,
    input_data=input_data,
    n_classes=n_classes,
    num_batches=num_batches,
    batch_size=batch_size,
    T_start=T_start,
    T_end=T_end,
    T_decay=T_decay,
    noise_start=noise_start,
    noise_end=noise_end,
    noise_decay=noise_decay,
    print_every=print_every
)

# ============== SAVE EVERYTHING ==============
save_data = {
    # Model parameters (learned)
    'model_params': trainer.model.get_params(),
    
    # Network structure
    'reaction_strings': reaction_strings,
    'species_names': species_names,
    'L': L,
    'adjacency_matrix': adjacency_matrix,
    'input_substrates_list': input_substrates_list,
    
    # Network configuration
    'seed': seed,
    'n_classes': n_classes,
    'NR': NR,
    'NS_vec': NS_vec,
    'NS': NS,
    'p_r': p_r,
    'hidden_dim': hidden_dim,
    'input_nodes': input_nodes,
    'target_nodes': target_nodes,
    'target_node_idxs': target_node_idxs,
    'n_inputs': n_inputs,
    'default_l0': default_l0,
    
    # Data generation parameters
    'input_dim': input_dim,
    'proj_dim': proj_dim,
    'center_variance': center_variance,
    'log_variance': log_variance,
    'center_offset': 0.0,
    'n_samples_per_class': 10000,
    'log_means': log_means,
    
    # Training hyperparameters
    'num_batches': num_batches,
    'batch_size': batch_size,
    'T_start': T_start,
    'T_end': T_end,
    'T_decay': T_decay,
    'noise_start': noise_start,
    'noise_end': noise_end,
    'noise_decay': noise_decay,
    'lr_dict': {'log_rates': 0.001, 'log_l0': 0.001},
    'max_grad_norm': 50.0,
    
    # Training history
    'history': history,
}

with open(f'{output_dir}/training_results.pkl', 'wb') as f:
    pickle.dump(save_data, f)
