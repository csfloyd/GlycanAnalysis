"""
Training module for reaction networks.

This module contains functions for training reaction networks.
"""

import numpy as np
import networkx as nx
import signal
from itertools import product
from typing import List, Dict, Tuple, Iterator, Optional
from abc import ABC, abstractmethod
import sympy
from scipy.optimize import nnls
from scipy.optimize import minimize

from CRNs.regularizers import _active_form_index, _group_substrate_forms, _parse_substrate_species
from CRNs.utils import timeout_handler


# ============== ABSTRACT BASE CLASS ==============

class ForwardModel(ABC):
    """Abstract base class for trainable forward models."""
    
    @abstractmethod
    def forward(self, inputs) -> np.ndarray:
        """Compute class outputs from inputs.
        
        Args:
            inputs: input values (model-specific format)
            
        Returns:
            class_outputs: array of shape (n_classes,)
        """
        pass
    
    @abstractmethod
    def backward(self, probs: np.ndarray, target_idx: int, temperature: float = 1.0) -> Dict[str, np.ndarray]:
        """Compute parameter gradients for cross-entropy loss.
        
        Args:
            probs: softmax probabilities (n_classes,)
            target_idx: index of correct class
            temperature: softmax temperature
            
        Returns:
            dict mapping parameter names to gradient arrays
        """
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, np.ndarray]:
        """Return dict of parameter_name -> parameter array."""
        pass
    
    @abstractmethod
    def set_params(self, params_dict: Dict[str, np.ndarray]):
        """Set parameters from dict."""
        pass
    
    @abstractmethod
    def get_param_shapes(self) -> Dict[str, Tuple]:
        """Return dict of parameter_name -> shape tuple."""
        pass


# ============== MLP MODEL ==============

class MLPModel(ForwardModel):
    """MLP-based forward model with backprop gradients."""
    
    def __init__(self, mlp: 'SimpleMLP', n_classes: int):
        """
        Args:
            mlp: SimpleMLP instance
            n_classes: number of output classes
        """
        self.mlp = mlp
        self.n_classes = n_classes
    
    def forward(self, inputs) -> np.ndarray:
        """Forward pass returning logits."""
        x = np.asarray(inputs).flatten()
        logits = self.mlp.forward(x)
        return logits
    
    def backward(self, probs: np.ndarray, target_idx: int, temperature: float = 1.0) -> Dict[str, np.ndarray]:
        """Backprop gradients for cross-entropy loss."""
        grad_weights, grad_biases = self.mlp.backward(probs, target_idx, temperature)
        
        # Flatten to single gradient vector
        grads = []
        for dW, db in zip(grad_weights, grad_biases):
            grads.append(dW.flatten())
            grads.append(db.flatten())
        
        return {'params': np.concatenate(grads)}
    
    def backward_mse(self, probs: np.ndarray, target_vec: np.ndarray, temperature: float = 1.0) -> Dict[str, np.ndarray]:
        """Backprop gradients for MSE loss."""
        grad_weights, grad_biases = self.mlp.backward_mse(probs, target_vec, temperature)
        
        grads = []
        for dW, db in zip(grad_weights, grad_biases):
            grads.append(dW.flatten())
            grads.append(db.flatten())
        
        return {'params': np.concatenate(grads)}
    
    def get_params(self) -> Dict[str, np.ndarray]:
        return {'params': self.mlp.get_flat_params()}
    
    def set_params(self, params_dict: Dict[str, np.ndarray]):
        self.mlp.set_flat_params(params_dict['params'])
    
    def get_param_shapes(self) -> Dict[str, Tuple]:
        return {'params': (self.mlp.get_param_count(),)}

    def get_regularization_state(self) -> Dict:
        """
        Extract state for MLP regularizers (call after forward()).

        Returns hidden post-activation vector from the penultimate layer.
        """
        if not hasattr(self.mlp, 'activations') or len(self.mlp.activations) < 3:
            raise RuntimeError("Must call forward() before getting regularization state")
        if self.mlp.n_layers < 2:
            raise RuntimeError("MLP must have at least one hidden layer for regularization")
        return {
            'hidden_activations': self.mlp.activations[-2].copy(),
            'penultimate_index': len(self.mlp.activations) - 2,
            'mlp': self.mlp,
        }
    
    def get_penultimate_activations(self) -> np.ndarray:
        """
        Extract penultimate layer activations under sigmoid (for MI training).
        
        Returns h ∈ [0,1]^d suitable for use as Bernoulli parameters.
        Must be called after forward().
        """
        if not hasattr(self.mlp, 'activations') or len(self.mlp.activations) < 3:
            raise RuntimeError("Must call forward() before getting penultimate activations")
        if self.mlp.n_layers < 2:
            raise RuntimeError("MLP must have at least one hidden layer")
        
        # Get pre-activation (linear output) of penultimate layer
        penultimate_pre = self.mlp.pre_activations[-2]
        
        # Apply sigmoid to get h ∈ [0,1]
        h = 1.0 / (1.0 + np.exp(-np.clip(penultimate_pre, -500, 500)))
        
        return h


# ============== LINEAR READOUT ==============

def identify_hidden_species_indices(
    species_names: List[str],
    class_ids: List[int],
    n_inputs: int,
) -> List[int]:
    """Species indices for hidden substrates (not receptors or class outputs)."""
    output_names = {species_names[idx] for idx in class_ids}
    hidden_indices = []
    for i, name in enumerate(species_names):
        parsed = _parse_substrate_species(name)
        if parsed is None:
            continue
        base_name = name.rstrip('s')
        if base_name in output_names or name in output_names:
            continue
        hidden_indices.append(i)
    return hidden_indices


def build_hidden_readout_groups(
    species_names: List[str],
    hidden_indices: List[int],
) -> List[Dict]:
    """
    Ordered hidden substrate groups for linear readout.

    Each group is one hidden node (active + inactive phosphoforms).
    """
    substrate_groups = _group_substrate_forms(species_names, list(hidden_indices))
    groups = []
    for base_id in sorted(substrate_groups.keys()):
        forms = sorted(substrate_groups[base_id], key=lambda x: x[0])
        if len(forms) < 2:
            continue
        active_idx = _active_form_index(forms)
        groups.append({
            'base_id': base_id,
            'active_idx': active_idx,
            'form_indices': [idx for _, idx, _ in forms],
        })
    return groups


def extract_hidden_vector(
    C_full: np.ndarray,
    hidden_groups: List[Dict],
) -> np.ndarray:
    """Build h from raw active-form concentrations at hidden substrates."""
    h = np.zeros(len(hidden_groups), dtype=float)
    for m, group in enumerate(hidden_groups):
        h[m] = C_full[group['active_idx']]
    return h


def hidden_jacobian_wrt_C(
    n_species: int,
    hidden_groups: List[Dict],
) -> np.ndarray:
    """Jacobian dh/dC with shape (n_hidden, n_species) for raw active concentrations."""
    dh_dC = np.zeros((len(hidden_groups), n_species), dtype=float)
    for m, group in enumerate(hidden_groups):
        dh_dC[m, group['active_idx']] = 1.0
    return dh_dC


# ============== CRN MODEL ==============

def _softmax_jacobian(probs: np.ndarray, temperature: float) -> np.ndarray:
    return (np.diag(probs) - np.outer(probs, probs)) / temperature


def _cross_entropy_logit_grad(probs: np.ndarray, target_idx: int, temperature: float) -> np.ndarray:
    """dL/dy for softmax + cross-entropy with temperature T."""
    one_hot = np.zeros_like(probs)
    one_hot[target_idx] = 1.0
    return (probs - one_hot) / temperature


class CRNModel(ForwardModel):
    """CRN-based forward model with analytical gradients.
    
    Supports multiple forward computation methods:
    - 'ode': Full ODE integration to steady state (accurate, slow)
    - 'graph': Fast graph-based approximation
    
    Readout options:
    - 'biochemical': class logits = steady-state output species concentrations
    - 'linear': class logits = W @ h + b from hidden active concentrations h
    
    Both methods use the same analytical gradient computation.
    """
    
    def __init__(self, 
                 r_n,
                 sim,
                 L: np.ndarray,
                 class_ids: List[int],
                 n_inputs: int,
                 default_l0: np.ndarray,
                 forward_method: str = 'ode',
                 graph_comp: Optional['GraphComputation'] = None,
                 dR_dC_func=None,
                 dR_dk_func=None,
                 dR_dl_func=None,
                 generate_init_func=None,
                 readout_type: str = 'biochemical',
                 readout_init_std: Optional[float] = None,
                 readout_seed: Optional[int] = None,
                 t_span: Tuple[float, float] = (0, 100000),
                 num_points: int = 10000,
                 rtol: float = 1e-12,
                 atol: float = 1e-12,
                 timeout_seconds: Optional[int] = 5):
        """
        Args:
            r_n: ReactionNetwork instance
            sim: Simulator instance
            L: Conservation law matrix
            class_ids: indices of output species (target nodes) for biochemical readout,
                or class count anchors for linear readout (hidden nodes excluded from h)
            n_inputs: number of input species (receptors)
            default_l0: default conservation constants
            forward_method: 'ode' or 'graph'
            graph_comp: GraphComputation instance (required if forward_method='graph')
            dR_dC_func: Jacobian dR/dC function
            dR_dk_func: Jacobian dR/dk function  
            dR_dl_func: Jacobian dR/dl function
            generate_init_func: function to generate initial concentrations from l0
            readout_type: 'biochemical' or 'linear'
            readout_init_std: std dev for initializing W (default: Xavier-like)
            readout_seed: RNG seed for linear readout initialization
            t_span: ODE integration time span
            num_points: number of ODE integration points
            rtol, atol: ODE integration tolerances
            timeout_seconds: max seconds for a single ODE integration; None or 0 disables.
                Timed-out samples are skipped (same as sampling). SIGALRM, so Linux/SLURM only.
        """
        if readout_type not in ('biochemical', 'linear'):
            raise ValueError(f"Unknown readout_type: {readout_type}")

        self.r_n = r_n
        self.sim = sim
        self.L = L
        self.class_ids = class_ids
        self.n_classes = len(class_ids)
        self.n_inputs = n_inputs
        self.readout_type = readout_type
        
        self.forward_method = forward_method
        self.graph_comp = graph_comp
        
        # Jacobian functions for analytical gradients
        self.dR_dC_func = dR_dC_func
        self.dR_dk_func = dR_dk_func
        self.dR_dl_func = dR_dl_func
        self.generate_init_func = generate_init_func
        
        # ODE settings
        self.t_span = t_span
        self.num_points = num_points
        self.rtol = rtol
        self.atol = atol
        self.timeout_seconds = timeout_seconds
        self.n_integration_timeouts = 0
        
        # Parameters
        self._rates = np.array(r_n.get_rates())
        self._default_l0 = default_l0.copy()
        
        # Indices for trainable l0 (typically exclude inputs and outputs)
        self.l0_train_range = (n_inputs, len(default_l0) - len(class_ids))

        # Linear readout: h -> y = W h + b
        self._hidden_groups = []
        self._W = None
        self._b = None
        self._h = None
        if readout_type == 'linear':
            hidden_indices = identify_hidden_species_indices(
                r_n.species_names, class_ids, n_inputs
            )
            self._hidden_groups = build_hidden_readout_groups(
                r_n.species_names, hidden_indices
            )
            n_hidden = len(self._hidden_groups)
            if n_hidden == 0:
                raise ValueError(
                    "linear readout requires at least one hidden substrate node"
                )
            rng = np.random.default_rng(readout_seed)
            if readout_init_std is None:
                readout_init_std = np.sqrt(2.0 / (n_hidden + self.n_classes))
            self._W = rng.normal(0.0, readout_init_std, size=(self.n_classes, n_hidden))
            self._b = np.zeros(self.n_classes, dtype=float)
        
        # Cached forward pass results
        self._C_full = None
        self._C_reduced_final = None
        self._current_l0 = None
    
    def set_forward_method(self, method: str):
        """Switch forward computation method ('ode' or 'graph')."""
        if method not in ('ode', 'graph'):
            raise ValueError(f"Unknown forward method: {method}")
        if method == 'graph' and self.graph_comp is None:
            raise ValueError("GraphComputation not provided")
        self.forward_method = method
    
    def forward(self, inputs) -> np.ndarray:
        """Compute steady-state class outputs.
        
        Args:
            inputs: array of input values (receptor concentrations)
            
        Returns:
            class_outputs: array of shape (n_classes,)
        """
        # Build l0 with inputs
        l0 = self._default_l0.copy()
        l0[:self.n_inputs] = inputs
        self._current_l0 = l0
        
        if self.forward_method == 'ode':
            C_full, C_reduced = self._forward_ode(l0)
        elif self.forward_method == 'graph':
            C_full, C_reduced = self._forward_graph(l0)
        else:
            raise ValueError(f"Unknown forward method: {self.forward_method}")
        
        self._C_full = C_full
        self._C_reduced_final = C_reduced
        
        if self.readout_type == 'biochemical':
            return np.array([C_full[idx] for idx in self.class_ids])

        self._h = extract_hidden_vector(C_full, self._hidden_groups)
        return self._W @ self._h + self._b
    
    def _forward_ode(self, l0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass via ODE integration."""
        C_full = self.generate_init_func(self.L, l0)
        _, C_reduced_init = self.sim.get_const_and_reduced_init(C_full)
        self.sim.make_reduced_rhs_with_conservation(l0)

        use_timeout = (
            self.timeout_seconds is not None
            and self.timeout_seconds > 0
            and hasattr(signal, 'SIGALRM')
        )
        if use_timeout:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(self.timeout_seconds))
        try:
            sol_reduced, C_reduced_final = self.sim.integrate(
                self.sim.reduced_ode_rhs, C_reduced_init,
                t_span=self.t_span, num_points=self.num_points,
                method='LSODA', rtol=self.rtol, atol=self.atol
            )
        except TimeoutError:
            self.n_integration_timeouts += 1
            raise
        finally:
            if use_timeout:
                signal.alarm(0)
        
        C_full = self.sim.recover_eliminated_species(l0, C_reduced_final)
        return C_full, C_reduced_final
    
    def _forward_graph(self, l0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass via graph computation."""
        # Graph forward returns C_full directly
        C_full = self.graph_comp.forward(self._rates, l0)
        
        # Derive C_reduced from C_full for gradient computation
        _, C_reduced = self.sim.get_const_and_reduced_init(C_full)
        
        return C_full, C_reduced
    
    def _compute_sensitivity_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return full dC/dk and dC/dl matrices at the cached steady state."""
        dC_dk = self.sim.dC_dk_func(
            self._C_reduced_final, self._current_l0, self._rates,
            self.dR_dC_func, self.dR_dk_func
        )
        dC_dk_full = self.sim.compute_dC_dk_full(dC_dk, l_bool=False)

        dC_dl = self.sim.dC_dl_func(
            self._C_reduced_final, self._current_l0, self._rates,
            self.dR_dC_func, self.dR_dl_func
        )
        dC_dl_full = self.sim.compute_dC_dk_full(dC_dl, l_bool=True)
        return dC_dk_full, dC_dl_full

    def _readout_jacobian_wrt_C(self) -> np.ndarray:
        """
        dy/dC with shape (n_classes, n_species).

        Biochemical readout: rows are unit vectors at class species indices.
        Linear readout: W @ dh/dC.
        """
        n_species = len(self.r_n.species_names)
        if self.readout_type == 'biochemical':
            dy_dC = np.zeros((self.n_classes, n_species), dtype=float)
            for i, idx in enumerate(self.class_ids):
                dy_dC[i, idx] = 1.0
            return dy_dC

        dh_dC = hidden_jacobian_wrt_C(n_species, self._hidden_groups)
        return self._W @ dh_dC

    def _backward_from_dy_dparams(
        self,
        dL_dy: np.ndarray,
        dC_dk_full: np.ndarray,
        dC_dl_full: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Chain dL/dy through readout and sensitivities to trainable parameters."""
        dy_dC = self._readout_jacobian_wrt_C()
        dL_dC = dL_dy @ dy_dC
        grad_k = dL_dC @ dC_dk_full
        grad_l = dL_dC @ dC_dl_full
        grads = {
            'log_rates': grad_k * self._rates,
            'log_l0': grad_l * self._default_l0,
        }
        if self.readout_type == 'linear':
            grads['W'] = np.outer(dL_dy, self._h)
            grads['b'] = dL_dy.copy()
        return grads
    
    def backward(self, probs: np.ndarray, target_idx: int, temperature: float = 1.0) -> Dict[str, np.ndarray]:
        """Compute analytical gradients w.r.t. rates and l0.
        
        Uses dC_dk and dC_dl sensitivity functions evaluated at steady state.
        """
        dC_dk_full, dC_dl_full = self._compute_sensitivity_matrices()

        if self.readout_type == 'biochemical':
            softmax_jacobian = _softmax_jacobian(probs, temperature)
            dC_dk_class = np.array([dC_dk_full[idx] for idx in self.class_ids])
            dC_dl_class = np.array([dC_dl_full[idx] for idx in self.class_ids])
            dprobs_dk = softmax_jacobian @ dC_dk_class
            dprobs_dl = softmax_jacobian @ dC_dl_class

            eps = 1e-8
            grad_k = -dprobs_dk[target_idx] / (probs[target_idx] + eps)
            grad_l = -dprobs_dl[target_idx] / (probs[target_idx] + eps)
            return {
                'log_rates': grad_k * self._rates,
                'log_l0': grad_l * self._default_l0
            }

        dL_dy = _cross_entropy_logit_grad(probs, target_idx, temperature)
        return self._backward_from_dy_dparams(dL_dy, dC_dk_full, dC_dl_full)
    
    def backward_mse(self, probs: np.ndarray, target_vec: np.ndarray, temperature: float = 1.0) -> Dict[str, np.ndarray]:
        """Compute analytical gradients for MSE loss."""
        dC_dk_full, dC_dl_full = self._compute_sensitivity_matrices()

        if self.readout_type == 'biochemical':
            softmax_jacobian = _softmax_jacobian(probs, temperature)
            dC_dk_class = np.array([dC_dk_full[idx] for idx in self.class_ids])
            dC_dl_class = np.array([dC_dl_full[idx] for idx in self.class_ids])
            dprobs_dk = softmax_jacobian @ dC_dk_class
            dprobs_dl = softmax_jacobian @ dC_dl_class

            diff = probs - target_vec
            grad_k = np.sum(diff[:, None] * dprobs_dk, axis=0)
            grad_l = np.sum(diff[:, None] * dprobs_dl, axis=0)
            return {
                'log_rates': grad_k * self._rates,
                'log_l0': grad_l * self._default_l0
            }

        softmax_jacobian = _softmax_jacobian(probs, temperature)
        dL_dy = softmax_jacobian.T @ (probs - target_vec)
        return self._backward_from_dy_dparams(dL_dy, dC_dk_full, dC_dl_full)
    
    def get_params(self) -> Dict[str, np.ndarray]:
        params = {
            'log_rates': np.log(self._rates),
            'log_l0': np.log(self._default_l0),
        }
        if self.readout_type == 'linear':
            params['W'] = self._W.copy()
            params['b'] = self._b.copy()
        return params
    
    def set_params(self, params_dict: Dict[str, np.ndarray]):
        self._rates = np.exp(params_dict['log_rates'])
        self._default_l0 = np.exp(params_dict['log_l0'])
        self.r_n.update_rates(self._rates)
        if self.readout_type == 'linear':
            self._W = np.array(params_dict['W'], dtype=float)
            self._b = np.array(params_dict['b'], dtype=float)
    
    def get_param_shapes(self) -> Dict[str, Tuple]:
        shapes = {
            'log_rates': self._rates.shape,
            'log_l0': self._default_l0.shape,
        }
        if self.readout_type == 'linear':
            shapes['W'] = self._W.shape
            shapes['b'] = self._b.shape
        return shapes
    
    def get_C_full(self) -> np.ndarray:
        """Return the full concentration vector from last forward pass."""
        return self._C_full
    
    def get_regularization_state(self) -> Dict:
        """
        Extract state information needed for regularization computations.
        
        Must be called after forward() to have valid state.
        
        Returns:
            state_dict: Dictionary containing:
                - 'C_full': Full concentration vector (all species)
                - 'C_reduced': Reduced concentration vector (remaining species)
                - 'l0': Conservation constants
                - 'rates': Rate constants
                - 'species_names': List of species names
                - 'hidden_indices': Indices of hidden substrate nodes in C_full
                - 'class_ids': Indices of output nodes
                - 'n_inputs': Number of input nodes
        """
        if self._C_full is None:
            raise RuntimeError("Must call forward() before getting regularization state")
        
        # Identify hidden node indices
        hidden_indices = identify_hidden_species_indices(
            self.r_n.species_names, self.class_ids, self.n_inputs
        )
        
        state = {
            'C_full': self._C_full.copy(),
            'C_reduced': self._C_reduced_final.copy() if self._C_reduced_final is not None else None,
            'l0': self._current_l0.copy() if self._current_l0 is not None else self._default_l0.copy(),
            'rates': self._rates.copy(),
            'species_names': self.r_n.species_names.copy(),
            'hidden_indices': hidden_indices,
            'class_ids': self.class_ids,
            'n_inputs': self.n_inputs,
            'readout_type': self.readout_type,
        }
        if self.readout_type == 'linear' and self._h is not None:
            state['hidden_activations'] = self._h.copy()
            state['readout_W'] = self._W.copy()
            state['readout_b'] = self._b.copy()
        return state


# ============== UNIFIED TRAINER ==============

class UnifiedTrainer:
    """Unified training loop for any ForwardModel."""
    
    def __init__(self, 
                 model: ForwardModel,
                 optimizer_type: str = 'adam',
                 lr: float = 0.1,
                 lr_dict: Optional[Dict[str, float]] = None,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 eps: float = 1e-8,
                 max_grad_norm: float = 50.0,
                 loss_type: str = 'cross_entropy',
                 frozen_params: Optional[List[str]] = None,
                 regularizers: Optional[List] = None,
                 reg_schedule_type: str = 'none',
                 reg_schedule_params: Optional[Dict] = None):
        """
        Args:
            model: ForwardModel instance (MLPModel or CRNModel)
            optimizer_type: 'adam' or 'sgd'
            lr: default learning rate
            lr_dict: optional per-parameter learning rates {'param_name': lr}
            beta1, beta2, eps: Adam hyperparameters
            max_grad_norm: gradient clipping threshold
            loss_type: 'cross_entropy' or 'mse'
            frozen_params: list of parameter names to freeze (e.g. ['log_l0'])
            regularizers: list of Regularizer objects (optional)
            reg_schedule_type: Regularization weight scheduling type:
                - 'none': No scheduling (constant weight)
                - 'linear_warmup': Linear increase from 0 to 1
                - 'exponential_warmup': Exponential increase to 1
                - 'delayed': Step function (0 until delay_batches, then 1)
                - 'cosine_warmup': Cosine-based smooth warmup
            reg_schedule_params: Parameters for scheduling:
                - 'warmup_batches': Number of batches for warmup (linear, exponential, cosine)
                - 'delay_batches': Number of batches before activation (delayed)
                - 'tau': Time constant for exponential (default: warmup_batches/3)
        """
        self.model = model
        self.optimizer_type = optimizer_type
        self.lr = lr
        self.lr_dict = lr_dict or {}
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.max_grad_norm = max_grad_norm
        self.loss_type = loss_type
        self.frozen_params = set(frozen_params) if frozen_params else set()
        self.regularizers = regularizers if regularizers else []
        
        # Regularization scheduling
        self.reg_schedule_type = reg_schedule_type
        self.reg_schedule_params = reg_schedule_params or {}
        
        # Initialize Adam state for each parameter group
        self.m = {}  # first moment
        self.v = {}  # second moment
        self.t = 0   # timestep
        
        for name, shape in model.get_param_shapes().items():
            self.m[name] = np.zeros(shape)
            self.v[name] = np.zeros(shape)
    
    def get_regularizer_schedule_factor(self, batch_num: int) -> float:
        """
        Compute regularization weight multiplier based on schedule.
        
        Args:
            batch_num: Current batch number (0-indexed)
            
        Returns:
            factor: Multiplier in [0, 1] to apply to regularizer weights
        """
        if self.reg_schedule_type == 'none':
            return 1.0
        
        elif self.reg_schedule_type == 'linear_warmup':
            warmup_batches = self.reg_schedule_params.get('warmup_batches', 500)
            return min(1.0, batch_num / max(1, warmup_batches))
        
        elif self.reg_schedule_type == 'exponential_warmup':
            warmup_batches = self.reg_schedule_params.get('warmup_batches', 500)
            tau = self.reg_schedule_params.get('tau', warmup_batches / 3.0)
            return 1.0 - np.exp(-batch_num / max(1, tau))
        
        elif self.reg_schedule_type == 'delayed':
            delay_batches = self.reg_schedule_params.get('delay_batches', 500)
            return 1.0 if batch_num >= delay_batches else 0.0
        
        elif self.reg_schedule_type == 'cosine_warmup':
            warmup_batches = self.reg_schedule_params.get('warmup_batches', 500)
            if batch_num >= warmup_batches:
                return 1.0
            progress = batch_num / max(1, warmup_batches)
            return 0.5 * (1.0 - np.cos(np.pi * progress))
        
        else:
            raise ValueError(f"Unknown reg_schedule_type: {self.reg_schedule_type}")
    
    def softmax(self, outputs: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Softmax with temperature scaling."""
        scaled = outputs / temperature
        shifted = scaled - np.max(scaled)
        exp_vals = np.exp(shifted)
        return exp_vals / np.sum(exp_vals)
    
    def compute_loss(self, probs: np.ndarray, target_idx: int) -> float:
        """Compute loss value."""
        if self.loss_type == 'cross_entropy':
            return -np.log(probs[target_idx] + 1e-8)
        elif self.loss_type == 'mse':
            target_vec = np.zeros(len(probs))
            target_vec[target_idx] = 1.0
            return 0.5 * np.sum((probs - target_vec) ** 2)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
    
    def clip_gradient(self, grad: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """Clip gradient by global norm."""
        grad_norm = np.linalg.norm(grad)
        if grad_norm > self.max_grad_norm:
            return grad * (self.max_grad_norm / grad_norm), grad_norm, True
        return grad, grad_norm, False
    
    def train_step(self, inputs, target_idx: int, temperature: float = 1.0,
                   noise_scale: float = 0.0) -> Tuple[float, np.ndarray, Dict[str, float], Dict[str, float]]:
        """Single training step with optional regularization.
        
        Args:
            inputs: input values for the model
            target_idx: index of correct class
            temperature: softmax temperature
            noise_scale: gradient noise scale (0 = no noise)
            
        Returns:
            loss: scalar loss value (classification + regularization)
            probs: probability vector
            grad_norms: dict of gradient norms per parameter
            reg_values: dict of raw regularizer metric values (name -> raw metric, not weighted penalty)
        """
        # Forward pass
        outputs = self.model.forward(inputs)
        probs = self.softmax(outputs, temperature)
        
        # Compute classification loss
        class_loss = self.compute_loss(probs, target_idx)
        
        # Compute regularization penalties (if any)
        reg_loss = 0.0
        reg_breakdown = {}  # For logging individual regularizer contributions
        reg_metrics = {}  # For logging the raw metric values (not penalties)
        
        if len(self.regularizers) > 0 and isinstance(self.model, CRNModel):
            try:
                # Get model state for regularization
                state_dict = self.model.get_regularization_state()
                
                # Get schedule factor (0 to 1 based on training progress)
                schedule_factor = self.get_regularizer_schedule_factor(self.t)
                
                # Compute penalty from each regularizer
                for reg in self.regularizers:
                    try:
                        penalty = reg.compute_penalty(state_dict)
                        # Apply both regularizer weight AND schedule factor
                        weighted_penalty = reg.weight * schedule_factor * penalty
                        reg_loss += weighted_penalty
                        reg_breakdown[reg.name] = weighted_penalty
                        
                        # Compute raw metric value
                        metric_value = reg.compute_metric(state_dict)
                        reg_metrics[reg.name] = metric_value
                    except NotImplementedError:
                        # Regularizer not yet implemented, skip
                        pass
                    except Exception as e:
                        print(f"Warning: Regularizer {reg.name} failed: {e}")
            except Exception as e:
                print(f"Warning: Could not compute regularization: {e}")

        elif len(self.regularizers) > 0 and isinstance(self.model, MLPModel):
            try:
                state_dict = self.model.get_regularization_state()
                schedule_factor = self.get_regularizer_schedule_factor(self.t)

                for reg in self.regularizers:
                    if not hasattr(reg, 'compute_gradients_mlp'):
                        continue
                    try:
                        penalty = reg.compute_penalty(state_dict)
                        weighted_penalty = reg.weight * schedule_factor * penalty
                        reg_loss += weighted_penalty
                        reg_breakdown[reg.name] = weighted_penalty
                        reg_metrics[reg.name] = reg.compute_metric(state_dict)
                    except NotImplementedError:
                        pass
                    except Exception as e:
                        print(f"Warning: Regularizer {reg.name} failed: {e}")
            except Exception as e:
                print(f"Warning: Could not compute regularization: {e}")
        
        # Total loss
        total_loss = class_loss + reg_loss
        
        # Backward pass: classification gradients
        if self.loss_type == 'cross_entropy':
            grads = self.model.backward(probs, target_idx, temperature)
        elif self.loss_type == 'mse':
            target_vec = np.zeros(len(probs))
            target_vec[target_idx] = 1.0
            grads = self.model.backward_mse(probs, target_vec, temperature)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        
        # Add regularization gradients (if applicable and implemented)
        if len(self.regularizers) > 0 and isinstance(self.model, CRNModel):
            try:
                # Get sensitivity matrices for gradient computation
                C_reduced = state_dict['C_reduced']
                l0 = state_dict['l0']
                rates = state_dict['rates']
                
                # Compute sensitivity matrices
                dC_dk = self.model.sim.dC_dk_func(
                    C_reduced, l0, rates, 
                    self.model.dR_dC_func, self.model.dR_dk_func
                )
                dC_dl = self.model.sim.dC_dl_func(
                    C_reduced, l0, rates,
                    self.model.dR_dC_func, self.model.dR_dl_func
                )
                
                # Compute full dC/dk including eliminated species
                dC_dk_full = self.model.sim.compute_dC_dk_full(dC_dk, l_bool=False)
                dC_dl_full = self.model.sim.compute_dC_dk_full(dC_dl, l_bool=True)
                
                # Add gradients from each regularizer (with schedule factor applied)
                for reg in self.regularizers:
                    try:
                        reg_grads = reg.compute_gradients(state_dict, dC_dk_full, dC_dl_full)
                        
                        # Apply schedule factor to regularizer gradients
                        # (regularizers already include their weight, we multiply by schedule)
                        for param_name, reg_grad in reg_grads.items():
                            if param_name in grads:
                                grads[param_name] = grads[param_name] + schedule_factor * reg_grad
                    except NotImplementedError:
                        # Regularizer not yet implemented, skip
                        pass
                    except Exception as e:
                        print(f"Warning: Regularizer {reg.name} gradient failed: {e}")
            except Exception as e:
                print(f"Warning: Could not compute regularization gradients: {e}")

        elif len(self.regularizers) > 0 and isinstance(self.model, MLPModel):
            try:
                state_dict = self.model.get_regularization_state()
                schedule_factor = self.get_regularizer_schedule_factor(self.t)

                for reg in self.regularizers:
                    if not hasattr(reg, 'compute_gradients_mlp'):
                        continue
                    try:
                        reg_grads = reg.compute_gradients_mlp(state_dict)
                        for param_name, reg_grad in reg_grads.items():
                            if param_name in grads:
                                grads[param_name] = grads[param_name] + schedule_factor * reg_grad
                    except NotImplementedError:
                        pass
                    except Exception as e:
                        print(f"Warning: Regularizer {reg.name} gradient failed: {e}")
            except Exception as e:
                print(f"Warning: Could not compute regularization gradients: {e}")
        
        # Get current params
        params = self.model.get_params()
        grad_norms = {}
        clipped = False
        
        # Process each parameter group (skip frozen params)
        trainable_params = [name for name in params if name not in self.frozen_params]
        
        for name in trainable_params:
            g = grads[name]
            
            # Gradient clipping
            g, grad_norm, was_clipped = self.clip_gradient(g)
            grad_norms[name] = grad_norm
            clipped = clipped or was_clipped
            
            # Add gradient noise
            if noise_scale > 0:
                noise = np.random.randn(*g.shape) * noise_scale * (np.abs(g).mean() + 1e-8)
                g = g + noise
            
            grads[name] = g
        
        # Optimizer step
        self.t += 1
        lr_effective = {}
        
        for name in trainable_params:
            g = grads[name]
            lr_param = self.lr_dict.get(name, self.lr)
            lr_effective[name] = lr_param
            
            if self.optimizer_type == 'adam':
                self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * g
                self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (g ** 2)
                m_hat = self.m[name] / (1 - self.beta1 ** self.t)
                v_hat = self.v[name] / (1 - self.beta2 ** self.t)
                params[name] = params[name] - lr_param * m_hat / (np.sqrt(v_hat) + self.eps)
            elif self.optimizer_type == 'sgd':
                params[name] = params[name] - lr_param * g
            else:
                raise ValueError(f"Unknown optimizer_type: {self.optimizer_type}")
        
        self.model.set_params(params)
        
        return total_loss, probs, grad_norms, reg_metrics
    
    def compute_accuracy(self, probs: np.ndarray, target_idx: int) -> bool:
        """Check if prediction is correct."""
        return np.argmax(probs) == target_idx
    
    def reset_optimizer_state(self):
        """Reset Adam momentum variables."""
        self.t = 0
        for name in self.m:
            self.m[name] = np.zeros_like(self.m[name])
            self.v[name] = np.zeros_like(self.v[name])
    
    def freeze_params(self, param_names: List[str]):
        """Freeze parameters (stop updating them)."""
        self.frozen_params.update(param_names)
    
    def unfreeze_params(self, param_names: List[str]):
        """Unfreeze parameters (resume updating them)."""
        self.frozen_params.difference_update(param_names)
    
    def get_trainable_params(self) -> List[str]:
        """Return list of currently trainable parameter names."""
        return [name for name in self.model.get_param_shapes().keys() 
                if name not in self.frozen_params]


def run_training(trainer: UnifiedTrainer,
                 input_data,
                 n_classes: int,
                 num_batches: int,
                 batch_size: int,
                 T_start: float = 1.0,
                 T_end: float = 0.2,
                 T_decay: float = 0.99,
                 noise_start: float = 1.0,
                 noise_end: float = 0.0,
                 noise_decay: float = 0.9,
                 print_every: int = 50,
                 history: 'TrainingHistory' = None,
                 verbose: bool = True) -> 'TrainingHistory':
    """
    Run a complete training loop.
    
    Args:
        trainer: UnifiedTrainer instance
        input_data: object with get_next_training_sample(class_idx) method
        n_classes: number of classes
        num_batches: number of batches to train
        batch_size: samples per batch
        T_start: initial softmax temperature
        T_end: final softmax temperature
        T_decay: temperature decay rate per batch
        noise_start: initial gradient noise scale
        noise_end: final gradient noise scale
        noise_decay: noise decay rate per batch
        print_every: print diagnostics every N batches
        history: TrainingHistory instance (created if None)
        verbose: whether to print progress
        
    Returns:
        TrainingHistory with recorded metrics
    """
    import time
    import random
    
    if history is None:
        history = TrainingHistory(n_classes)
    
    start_time = time.time()
    
    if verbose:
        print(f"Starting training: {num_batches} batches, batch_size={batch_size}")
        print("=" * 80)
    
    for batch in range(num_batches):
        # Annealing schedules
        temperature = max(T_end, T_start * (T_decay ** batch))
        noise_scale = max(noise_end, noise_start * (noise_decay ** batch))
        
        batch_loss = 0.0
        batch_correct = 0
        batch_valid = 0
        batch_grad_norms = None
        batch_reg_values = {}  # Accumulate regularizer values
        
        for sample in range(batch_size):
            target_idx = random.randrange(n_classes)
            inputs = input_data.get_next_training_sample(target_idx)
            
            try:
                loss, probs, grad_norms, reg_values = trainer.train_step(
                    inputs=np.array(inputs).flatten(),
                    target_idx=target_idx,
                    temperature=temperature,
                    noise_scale=noise_scale
                )
                
                # Check for numerical issues
                if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
                    if verbose:
                        print(f"  Warning: Invalid probs at batch {batch}, sample {sample}")
                    continue
                
                batch_loss += loss
                history.record_sample_loss(target_idx, loss)
                batch_correct += int(trainer.compute_accuracy(probs, target_idx))
                batch_valid += 1
                batch_grad_norms = grad_norms
                
                # Accumulate regularizer values (MLP version)
                for reg_name, reg_value in reg_values.items():
                    if reg_name not in batch_reg_values:
                        batch_reg_values[reg_name] = 0.0
                    batch_reg_values[reg_name] += reg_value
                
            except Exception as e:
                if verbose:
                    print(f"  Warning: Training step failed at batch {batch}, sample {sample}: {e}")
                continue
        
        # Skip if no valid samples
        if batch_valid == 0:
            if verbose:
                print(f"  Batch {batch}: No valid samples, skipping")
            continue
        
        # Get param stats
        params = trainer.model.get_params()
        param_values = np.concatenate([p.flatten() for p in params.values()])
        param_stats = (param_values.min(), param_values.max(), param_values.mean())
        
        # Average regularizer values
        avg_reg_values = {name: val / batch_valid for name, val in batch_reg_values.items()}
        
        # Get current regularization schedule factor
        reg_schedule_factor = trainer.get_regularizer_schedule_factor(batch)
        
        # Record batch metrics
        history.record_batch(
            avg_loss=batch_loss / batch_valid,
            accuracy=batch_correct / batch_valid,
            grad_norms=batch_grad_norms if batch_grad_norms else {},
            temperature=temperature,
            noise_scale=noise_scale,
            param_stats=param_stats,
            regularizer_values=avg_reg_values,
            reg_schedule_factor=reg_schedule_factor
        )
        
        # Print diagnostics
        if verbose and (batch % print_every == 0 or batch == num_batches - 1):
            recent_loss = history.get_recent_avg('loss')
            recent_acc = history.get_recent_avg('accuracy')
            
            grad_str = ""
            if batch_grad_norms:
                if len(batch_grad_norms) == 1:
                    grad_str = f"GradNorm: {list(batch_grad_norms.values())[0]:.2e}"
                else:
                    grad_str = " ".join([f"{k}:{v:.2e}" for k, v in batch_grad_norms.items()])
            
            # Build regularizer string
            reg_str = ""
            if avg_reg_values:
                reg_str = " | " + ", ".join([f"{k}:{v:.4f}" for k, v in avg_reg_values.items()])
            
            print(f"Batch {batch:4d}/{num_batches} | "
                  f"Loss: {batch_loss/batch_valid:.4f} (avg: {recent_loss:.4f}) | "
                  f"Acc: {batch_correct/batch_valid:.1%} (avg: {recent_acc:.1%}) | "
                  f"{grad_str} | "
                  f"T: {temperature:.2f}"
                  f"{reg_str}")
    
    training_time = time.time() - start_time
    
    if verbose:
        print("=" * 80)
        history.print_summary(training_time)
    
    return history


def run_training_crn(trainer: UnifiedTrainer,
                     input_data,
                     n_classes: int,
                     num_batches: int,
                     batch_size: int,
                     T_start: float = 1.0,
                     T_end: float = 0.2,
                     T_decay: float = 0.99,
                     noise_start: float = 5.0,
                     noise_end: float = 0.0,
                     noise_decay: float = 0.99,
                     print_every: int = 50,
                     history: 'TrainingHistory' = None,
                     verbose: bool = True) -> 'TrainingHistory':
    """
    Run training loop for CRN models (handles potential ODE integration failures).
    
    Same interface as run_training but with CRN-specific defaults and error handling.
    """
    import time
    import random
    
    if history is None:
        history = TrainingHistory(n_classes)
    
    start_time = time.time()
    
    if verbose:
        model_type = "CRN"
        if hasattr(trainer.model, 'forward_method'):
            model_type = f"CRN ({trainer.model.forward_method})"
        print(f"Starting {model_type} training: {num_batches} batches, batch_size={batch_size}")
        timeout_seconds = getattr(trainer.model, 'timeout_seconds', None)
        if timeout_seconds:
            print(f"ODE integration timeout: {timeout_seconds}s per sample")
        print("=" * 80)
    
    for batch in range(num_batches):
        # Annealing schedules
        temperature = max(T_end, T_start * (T_decay ** batch))
        noise_scale = max(noise_end, noise_start * (noise_decay ** batch))
        
        batch_loss = 0.0
        batch_correct = 0
        batch_valid = 0
        batch_grad_norms = None
        batch_reg_values = {}  # Accumulate regularizer values
        
        for sample in range(batch_size):
            target_idx = random.randrange(n_classes)
            inputs = input_data.get_next_training_sample(target_idx)
            
            try:
                loss, probs, grad_norms, reg_values = trainer.train_step(
                    inputs=inputs,  # CRN models handle input format internally
                    target_idx=target_idx,
                    temperature=temperature,
                    noise_scale=noise_scale
                )
                
                # Check for numerical issues
                if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
                    if verbose:
                        print(f"  Warning: Invalid probs at batch {batch}, sample {sample}")
                    continue
                
                # Check for invalid gradients
                has_invalid_grad = any(
                    np.any(np.isnan(g)) or np.any(np.isinf(g)) 
                    for g in grad_norms.values()
                )
                if has_invalid_grad:
                    if verbose:
                        print(f"  Warning: Invalid gradient at batch {batch}, sample {sample}")
                    continue
                
                batch_loss += loss
                history.record_sample_loss(target_idx, loss)
                batch_correct += int(trainer.compute_accuracy(probs, target_idx))
                batch_valid += 1
                batch_grad_norms = grad_norms
                
                # Accumulate regularizer values
                for reg_name, reg_value in reg_values.items():
                    if reg_name not in batch_reg_values:
                        batch_reg_values[reg_name] = 0.0
                    batch_reg_values[reg_name] += reg_value
                
            except TimeoutError:
                continue
            except Exception as e:
                if verbose:
                    print(f"  Warning: Training step failed at batch {batch}, sample {sample}: {e}")
                continue
        
        # Skip if no valid samples
        if batch_valid == 0:
            if verbose:
                print(f"  Batch {batch}: No valid samples, skipping")
            continue
        
        # Get param stats (for CRN, show rate range)
        params = trainer.model.get_params()
        if 'log_rates' in params:
            rates = np.exp(params['log_rates'])
            param_stats = (rates.min(), rates.max(), rates.mean())
        else:
            param_values = np.concatenate([p.flatten() for p in params.values()])
            param_stats = (param_values.min(), param_values.max(), param_values.mean())
        
        # Average regularizer values
        avg_reg_values = {name: val / batch_valid for name, val in batch_reg_values.items()}
        
        # Get current regularization schedule factor
        reg_schedule_factor = trainer.get_regularizer_schedule_factor(batch)
        
        # Record batch metrics
        history.record_batch(
            avg_loss=batch_loss / batch_valid,
            accuracy=batch_correct / batch_valid,
            grad_norms=batch_grad_norms if batch_grad_norms else {},
            temperature=temperature,
            noise_scale=noise_scale,
            param_stats=param_stats,
            regularizer_values=avg_reg_values,
            reg_schedule_factor=reg_schedule_factor
        )
        
        # Print diagnostics
        if verbose and (batch % print_every == 0 or batch == num_batches - 1):
            recent_loss = history.get_recent_avg('loss')
            recent_acc = history.get_recent_avg('accuracy')
            
            grad_str = ""
            if batch_grad_norms:
                grad_str = " ".join([f"{k[:6]}:{v:.2e}" for k, v in batch_grad_norms.items()])
            
            # Build regularizer string
            reg_str = ""
            if avg_reg_values:
                reg_str = " | " + ", ".join([f"{k}:{v:.4f}" for k, v in avg_reg_values.items()])
            
            # Add regularization schedule info if using schedule
            if trainer.reg_schedule_type != 'none' and avg_reg_values:
                reg_str += f" | RegSched: {reg_schedule_factor:.2f}"
            
            pmin, pmax, _ = param_stats
            print(f"Batch {batch:4d}/{num_batches} | "
                  f"Loss: {batch_loss/batch_valid:.4f} (avg: {recent_loss:.4f}) | "
                  f"Acc: {batch_correct/batch_valid:.1%} (avg: {recent_acc:.1%}) | "
                  f"{grad_str} | "
                  f"Rates: [{pmin:.2e}, {pmax:.2e}]"
                  f"{reg_str}")
    
    training_time = time.time() - start_time
    if hasattr(trainer.model, 'n_integration_timeouts'):
        history.n_integration_timeouts = trainer.model.n_integration_timeouts
    
    if verbose:
        print("=" * 80)
        history.print_summary(training_time)
    
    return history


class GraphComputation:
    """Computational graph for reaction networks using NumPy."""
    
    def __init__(self, G, input_nodes, output_nodes):
        self.G = G
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.topo_order = list(nx.topological_sort(G))
        self.edges = list(G.edges)
        self.nodes = list(G.nodes)
        self.n_nodes = len(self.nodes)
        self.n_edges = len(self.edges)
        
        # Create node index mappings
        self.node_to_idx = {n: i for i, n in enumerate(self.nodes)}
        self.edge_to_idx = {e: i for i, e in enumerate(self.edges)}
        
        # Pre-compute indices for fast access
        self.input_idxs = np.array([self.node_to_idx[n] for n in input_nodes])
        self.output_idxs = np.array([self.node_to_idx[n] for n in output_nodes])
        
        # Pre-compute NON-INPUT nodes in topological order
        input_set = set(input_nodes)
        self.compute_node_idxs = [
            self.node_to_idx[n] for n in self.topo_order if n not in input_set
        ]
        
        # Pre-compute predecessor structure
        self._build_predecessor_lists()
    
    def _build_predecessor_lists(self):
        """Pre-compute predecessor indices for each node."""
        self.pred_info = {}  # node_idx -> list of (pred_node_idx, edge_idx)
        
        for node in self.nodes:
            node_idx = self.node_to_idx[node]
            preds = list(self.G.predecessors(node))
            self.pred_info[node_idx] = [
                (self.node_to_idx[pred], self.edge_to_idx[(pred, node)])
                for pred in preds
            ]

    def build_r_n_maps(self, r_n):
        """Build index mappings from reaction network."""
        self.node_f_idx = np.zeros(self.n_nodes, dtype=np.int32)
        self.node_r_idx = np.zeros(self.n_nodes, dtype=np.int32)
        self.edge_f_idx = np.zeros(self.n_edges, dtype=np.int32)
        self.edge_r_idx = np.zeros(self.n_edges, dtype=np.int32)

        for (i, reaction) in enumerate(r_n.reactions):
            src = r_n.all_complexes[reaction[0]].split('+')
            dst = r_n.all_complexes[reaction[1]].split('+')
            num_src, num_dst = len(src), len(dst)
            
            # Unimolecular reactions: Xs <-> X
            if num_src == 1 and num_dst == 1:
                if src[0].endswith('s'):
                    node = src[0][:-1]
                    if node in self.node_to_idx:
                        self.node_f_idx[self.node_to_idx[node]] = i
                if dst[0].endswith('s'):
                    node = dst[0][:-1]
                    if node in self.node_to_idx:
                        self.node_r_idx[self.node_to_idx[node]] = i

            # Bimolecular reactions: A+Bs -> A+B or A+B -> A+Bs
            if num_src == 2 and num_dst == 2:
                src_set, dst_set = set(src), set(dst)
                common = src_set & dst_set
                
                if len(common) == 1:
                    upstream_node = common.pop()
                    src_only = (src_set - {upstream_node}).pop()
                    dst_only = (dst_set - {upstream_node}).pop()
                    
                    if src_only.endswith('s') and not dst_only.endswith('s'):
                        downstream_node = dst_only
                        is_forward = True
                    elif dst_only.endswith('s') and not src_only.endswith('s'):
                        downstream_node = src_only
                        is_forward = False
                    else:
                        continue
                    
                    edge_key = (upstream_node, downstream_node)
                    if edge_key in self.edge_to_idx:
                        edge_idx = self.edge_to_idx[edge_key]
                        if is_forward:
                            self.edge_f_idx[edge_idx] = i
                        else:
                            self.edge_r_idx[edge_idx] = i

        self.species_node_map = {}
        self.num_species = len(r_n.species_names)
        for node in self.nodes:
            self.species_node_map[node] = r_n.species_names.index(node)
            if node[0] == 'S':
                self.species_node_map[node+'s'] = r_n.species_names.index(node+'s')



    def forward(self, rates, l0):
        """
        Compute steady-state node values.
        
        Args:
            rates: (n_reactions,) array of rate constants
            l0: (n_nodes,) array of conservation constants
            input_vals: (n_inputs,) array of input values
            
        Returns:
            (n_outputs,) array of output node values
        """
        rates = np.asarray(rates)
        l0 = np.asarray(l0)
        # Initialize node values
        node_values = np.zeros(self.n_nodes)
        node_values[self.input_idxs] = l0[self.input_idxs]
        # Get rate parameters via indexing
        node_kf = rates[self.node_f_idx]
        node_kr = rates[self.node_r_idx]
        edge_kf = rates[self.edge_f_idx]
        edge_kr = rates[self.edge_r_idx]
        
        # Process nodes in topological order (skip inputs)
        for node_idx in self.compute_node_idxs:
            kf_node = node_kf[node_idx]
            kr_node = node_kr[node_idx]
            numerator = kf_node
            denominator = kf_node + kr_node
            
            # Sum contributions from predecessors
            for pred_idx, edge_idx in self.pred_info[node_idx]:
                kf_edge = edge_kf[edge_idx]
                kr_edge = edge_kr[edge_idx]
                pred_val = node_values[pred_idx]
                
                numerator += kf_edge * pred_val
                denominator += (kf_edge + kr_edge) * pred_val
            
            node_values[node_idx] = l0[node_idx] * numerator / (denominator + 1e-10)

        # Recover full concentrations
        C_full = np.zeros(self.num_species)
        for (i, node) in enumerate(self.nodes):
            C_full[self.species_node_map[node]] = node_values[i]
            if node[0] == 'S':
                C_full[self.species_node_map[node+'s']] = l0[i] - node_values[i]
        
        return C_full
    
    def loss(self, rates, l0, input_vals, targets):
        """MSE loss between predictions and targets."""
        preds = self.forward(rates, l0, input_vals)
        return np.mean((preds - targets) ** 2)


# ============== MLP IMPLEMENTATION ==============
class SimpleMLP:
    """
    Simple MLP with manual backpropagation for full control over gradients.
    """
    def __init__(self, layer_sizes, activation='relu'):
        """
        layer_sizes: list of ints, e.g. [input_dim, hidden1, hidden2, n_classes]
        """
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        self.activation = activation
        
        # Initialize weights and biases (Xavier initialization)
        self.weights = []
        self.biases = []
        for i in range(self.n_layers):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            W = np.random.randn(fan_in, fan_out) * std
            b = np.zeros(fan_out)
            self.weights.append(W)
            self.biases.append(b)
    
    def _activation(self, x):
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        else:
            return x  # linear
    
    def _activation_derivative(self, x):
        if self.activation == 'relu':
            return (x > 0).astype(float)
        elif self.activation == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation == 'sigmoid':
            s = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
            return s * (1 - s)
        else:
            return np.ones_like(x)
    
    def forward(self, x):
        """Forward pass, storing activations for backprop."""
        self.activations = [x]
        self.pre_activations = []
        
        for i in range(self.n_layers):
            z = self.activations[-1] @ self.weights[i] + self.biases[i]
            self.pre_activations.append(z)
            
            if i < self.n_layers - 1:
                # Hidden layer: apply activation
                a = self._activation(z)
            else:
                # Output layer: no activation (we apply softmax separately)
                a = z
            self.activations.append(a)
        
        return self.activations[-1]  # logits
    
    def softmax(self, logits, temperature=1.0):
        """Softmax with temperature."""
        scaled = logits / temperature
        shifted = scaled - np.max(scaled)
        exp_vals = np.exp(shifted)
        return exp_vals / np.sum(exp_vals)
    
    def backward(self, probs, target_idx, temperature=1.0):
        """
        Backward pass for cross-entropy loss with softmax output.
        Returns gradients for weights and biases.
        """
        grad_weights = []
        grad_biases = []
        
        # Gradient of cross-entropy + softmax: (p - y) / temperature
        # where y is one-hot target
        delta = probs.copy()
        delta[target_idx] -= 1.0
        delta /= temperature
        
        # Backpropagate through layers
        for i in range(self.n_layers - 1, -1, -1):
            # Gradient w.r.t. weights and biases
            a_prev = self.activations[i]
            dW = np.outer(a_prev, delta)
            db = delta.copy()
            
            grad_weights.insert(0, dW)
            grad_biases.insert(0, db)
            
            if i > 0:
                # Backprop through weights
                delta = self.weights[i] @ delta
                # Backprop through activation
                delta = delta * self._activation_derivative(self.pre_activations[i - 1])
        
        return grad_weights, grad_biases
    
    def backward_mse(self, probs, target_vec, temperature=1.0):
        """
        Backward pass for MSE loss with softmax output.
        """
        grad_weights = []
        grad_biases = []
        
        # d(MSE)/d(logits) via chain rule through softmax
        # d(MSE)/dp = (p - y)
        # dp/dz (softmax jacobian) = diag(p) - outer(p, p)
        diff = probs - target_vec
        softmax_jacobian = (np.diag(probs) - np.outer(probs, probs)) / temperature
        delta = softmax_jacobian.T @ diff
        
        for i in range(self.n_layers - 1, -1, -1):
            a_prev = self.activations[i]
            dW = np.outer(a_prev, delta)
            db = delta.copy()
            
            grad_weights.insert(0, dW)
            grad_biases.insert(0, db)
            
            if i > 0:
                delta = self.weights[i] @ delta
                delta = delta * self._activation_derivative(self.pre_activations[i - 1])
        
        return grad_weights, grad_biases
    
    def backprop_from_post_activation_grad(self, activation_index: int, dL_da: np.ndarray):
        """
        Backprop from gradient w.r.t. post-activation at activations[activation_index].

        Layers above activation_index (e.g. readout) receive zero gradient.
        Returns grad_weights, grad_biases aligned with self.weights/self.biases.
        """
        layer_idx = activation_index - 1
        if layer_idx < 0 or layer_idx >= self.n_layers:
            raise ValueError(
                f"activation_index={activation_index} invalid for n_layers={self.n_layers}"
            )

        grad_weights = [np.zeros_like(W) for W in self.weights]
        grad_biases = [np.zeros_like(b) for b in self.biases]
        delta = np.asarray(dL_da, dtype=float).copy()

        for i in range(layer_idx, -1, -1):
            a_prev = self.activations[i]
            grad_weights[i] = np.outer(a_prev, delta)
            grad_biases[i] = delta.copy()
            if i > 0:
                delta = self.weights[i] @ delta
                delta = delta * self._activation_derivative(self.pre_activations[i - 1])

        return grad_weights, grad_biases
    
    def get_flat_params(self):
        """Flatten all parameters into a single vector."""
        params = []
        for W, b in zip(self.weights, self.biases):
            params.append(W.flatten())
            params.append(b.flatten())
        return np.concatenate(params)
    
    def set_flat_params(self, flat_params):
        """Set parameters from a flat vector."""
        idx = 0
        for i in range(self.n_layers):
            W_shape = self.weights[i].shape
            W_size = np.prod(W_shape)
            self.weights[i] = flat_params[idx:idx + W_size].reshape(W_shape)
            idx += W_size
            
            b_size = self.biases[i].shape[0]
            self.biases[i] = flat_params[idx:idx + b_size]
            idx += b_size
    
    def get_param_count(self):
        """Return total number of parameters."""
        return sum(W.size + b.size for W, b in zip(self.weights, self.biases))


# ============== TRAINING HISTORY & DIAGNOSTICS ==============

class TrainingHistory:
    """Track and store training metrics."""
    
    def __init__(self, n_classes: int):
        self.n_classes = n_classes
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.loss_history = []
        self.accuracy_history = []
        self.grad_norm_history = []  # Can be dict or scalar
        self.temperature_history = []
        self.noise_scale_history = []
        self.loss_by_class = [[] for _ in range(self.n_classes)]
        self.param_stats_history = []  # (min, max, mean) tuples
        self.regularizer_history = []  # List of dicts with regularizer values per batch
        self.reg_schedule_history = []  # Regularization schedule factor per batch
        self.n_integration_timeouts = 0
    
    def record_batch(self, 
                     avg_loss: float,
                     accuracy: float,
                     grad_norms: dict,
                     temperature: float = None,
                     noise_scale: float = None,
                     param_stats: Tuple[float, float, float] = None,
                     regularizer_values: dict = None,
                     reg_schedule_factor: float = None):
        """Record metrics for a batch."""
        self.loss_history.append(avg_loss)
        self.accuracy_history.append(accuracy)
        self.grad_norm_history.append(grad_norms)
        
        if temperature is not None:
            self.temperature_history.append(temperature)
        if noise_scale is not None:
            self.noise_scale_history.append(noise_scale)
        if param_stats is not None:
            self.param_stats_history.append(param_stats)
        if regularizer_values is not None:
            self.regularizer_history.append(regularizer_values)
        if reg_schedule_factor is not None:
            self.reg_schedule_history.append(reg_schedule_factor)
    
    def record_sample_loss(self, class_idx: int, loss: float):
        """Record loss for a specific class sample."""
        self.loss_by_class[class_idx].append(loss)
    
    def get_recent_avg(self, metric: str, window: int = 100) -> float:
        """Get recent average of a metric."""
        history = getattr(self, f'{metric}_history', [])
        if len(history) == 0:
            return 0.0
        window = min(window, len(history))
        return np.mean(history[-window:])
    
    def get_recent_avg_regularizer(self, reg_name: str, window: int = 100) -> float:
        """Get recent average of a regularizer value."""
        if len(self.regularizer_history) == 0:
            return 0.0
        window = min(window, len(self.regularizer_history))
        recent = self.regularizer_history[-window:]
        values = [r.get(reg_name, 0.0) for r in recent]
        return np.mean(values) if values else 0.0

    def get_regularizer_series(self, reg_name: str = None) -> Dict[str, List[float]]:
        """Return per-batch regularizer metric series for plotting."""
        if len(self.regularizer_history) == 0:
            return {}
        names = [reg_name] if reg_name else sorted({
            k for batch in self.regularizer_history for k in batch.keys()
        })
        return {
            name: [batch.get(name, np.nan) for batch in self.regularizer_history]
            for name in names
        }
    
    def print_summary(self, training_time: float = None):
        """Print training summary statistics."""
        print(f"\n{'='*60}")
        print(f"Training Summary")
        print(f"{'='*60}")
        
        if training_time is not None:
            print(f"Training time: {training_time:.1f} seconds")
        
        if len(self.loss_history) > 0:
            print(f"Final loss (last 100 batches): {self.get_recent_avg('loss'):.4f}")
            print(f"Min loss achieved: {min(self.loss_history):.4f}")
        
        if len(self.accuracy_history) > 0:
            print(f"Final accuracy (last 100 batches): {self.get_recent_avg('accuracy'):.1%}")
            print(f"Best accuracy: {max(self.accuracy_history):.1%}")
        
        if len(self.param_stats_history) > 0:
            pmin, pmax, pmean = self.param_stats_history[-1]
            print(f"Final param range: [{pmin:.2e}, {pmax:.2e}], mean: {pmean:.2e}")
        
        if len(self.regularizer_history) > 0:
            print(f"\nRegularizer Metrics (raw values, not weighted):")
            # Get all regularizer names
            all_reg_names = set()
            for reg_dict in self.regularizer_history:
                all_reg_names.update(reg_dict.keys())
            
            for reg_name in sorted(all_reg_names):
                recent_avg = self.get_recent_avg_regularizer(reg_name, window=100)
                print(f"  {reg_name}: {recent_avg:.4f} (avg last 100 batches)")
        
        n_timeouts = getattr(self, 'n_integration_timeouts', 0)
        print(f"ODE integration timeouts: {n_timeouts}")


def plot_regularizer_history(history: TrainingHistory,
                             reg_names: List[str] = None,
                             title: str = "Regularizer Metrics During Training",
                             target_lines: Dict[str, float] = None,
                             figsize: Tuple[int, int] = (8, 4),
                             smoothing_window: int = None,
                             ax=None,
                             show: bool = True):
    """
    Plot raw regularizer metric values recorded each batch (e.g. hidden IPR).

    Args:
        history: TrainingHistory with regularizer_history populated
        reg_names: subset of regularizer names to plot (default: all)
        title: plot title
        target_lines: optional {reg_name: y_value} horizontal reference lines
        figsize: figure size when ax is None
        smoothing_window: moving-average window (optional)
        ax: existing matplotlib axis
        show: call plt.show()

    Returns:
        fig, ax
    """
    import matplotlib.pyplot as plt

    series = history.get_regularizer_series()
    if not series:
        print("No regularizer history to plot.")
        return None, None

    if reg_names is None:
        reg_names = sorted(series.keys())
    else:
        reg_names = [name for name in reg_names if name in series]

    if len(reg_names) == 0:
        print("No matching regularizer names in history.")
        return None, None

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    batches = np.arange(len(history.regularizer_history))
    colors = plt.cm.tab10(np.linspace(0, 1, len(reg_names)))

    for i, name in enumerate(reg_names):
        values = np.asarray(series[name], dtype=float)
        ax.plot(batches, values, alpha=0.25, color=colors[i])

        if smoothing_window is not None and len(values) >= smoothing_window:
            kernel = np.ones(smoothing_window) / smoothing_window
            smoothed = np.convolve(values, kernel, mode='valid')
            ax.plot(
                batches[smoothing_window - 1:],
                smoothed,
                color=colors[i],
                linewidth=2,
                label=name,
            )
        else:
            ax.plot(batches, values, color=colors[i], linewidth=1.5, label=name)

        if target_lines and name in target_lines:
            ax.axhline(
                target_lines[name],
                color=colors[i],
                linestyle='--',
                alpha=0.7,
                label=f'{name} target',
            )

    ax.set_xlabel('Batch')
    ax.set_ylabel('Metric value')
    ax.set_title(title)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def plot_training_diagnostics(history: TrainingHistory,
                               title: str = "Training Diagnostics",
                               max_grad_norm: float = None,
                               figsize: Tuple[int, int] = (12, 10),
                               smoothing_window: int = None,
                               show: bool = True):
    """
    Plot comprehensive training diagnostics.
    
    Args:
        history: TrainingHistory object with recorded metrics
        title: Overall figure title
        max_grad_norm: Gradient clipping threshold (for reference line)
        figsize: Figure size
        smoothing_window: Window size for smoothing (auto if None)
        show: Whether to call plt.show()
    
    Returns:
        fig, axes: matplotlib figure and axes
    """
    import matplotlib.pyplot as plt
    
    n_batches = len(history.loss_history)
    if smoothing_window is None:
        smoothing_window = min(50, n_batches // 10 + 1)
    
    # Determine layout based on available data
    has_grad_norms = len(history.grad_norm_history) > 0
    has_class_loss = any(len(losses) > 0 for losses in history.loss_by_class)
    has_annealing = len(history.temperature_history) > 0 or len(history.noise_scale_history) > 0
    has_regularizers = len(history.regularizer_history) > 0 and any(
        len(batch) > 0 for batch in history.regularizer_history
    )
    
    n_plots = 2 + int(has_grad_norms) + int(has_class_loss or has_annealing) + int(has_regularizers)
    n_rows = (n_plots + 1) // 2
    
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
    axes = axes.flatten() if n_plots > 2 else [axes] if n_plots == 1 else axes.flatten()
    
    plot_idx = 0
    
    # ===== Plot 1: Loss over training =====
    ax = axes[plot_idx]
    ax.plot(history.loss_history, alpha=0.3, label='Batch loss', color='blue')
    
    if n_batches >= smoothing_window:
        smoothed = np.convolve(history.loss_history, 
                               np.ones(smoothing_window)/smoothing_window, mode='valid')
        ax.plot(range(smoothing_window-1, n_batches), smoothed, 
                'r-', linewidth=2, label=f'Smoothed (w={smoothing_window})')
    
    ax.set_xlabel('Batch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_idx += 1
    
    # ===== Plot 2: Accuracy over training =====
    ax = axes[plot_idx]
    ax.plot(history.accuracy_history, alpha=0.3, label='Batch accuracy', color='green')
    
    if n_batches >= smoothing_window:
        smoothed = np.convolve(history.accuracy_history,
                               np.ones(smoothing_window)/smoothing_window, mode='valid')
        ax.plot(range(smoothing_window-1, n_batches), smoothed,
                'darkgreen', linewidth=2, label=f'Smoothed (w={smoothing_window})')
    
    ax.axhline(y=1/history.n_classes, color='k', linestyle='--', 
               alpha=0.5, label='Random chance')
    ax.set_xlabel('Batch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training Accuracy')
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_idx += 1
    
    # ===== Plot 3: Gradient norm =====
    if has_grad_norms:
        ax = axes[plot_idx]
        
        # Handle both dict and scalar grad norms
        grad_norms = history.grad_norm_history
        if isinstance(grad_norms[0], dict):
            # Plot each parameter group
            param_names = list(grad_norms[0].keys())
            colors = plt.cm.tab10(np.linspace(0, 1, len(param_names)))
            
            for i, name in enumerate(param_names):
                norms = [g.get(name, 0) for g in grad_norms]
                ax.semilogy(norms, alpha=0.7, color=colors[i], label=name)
        else:
            # Single scalar
            ax.semilogy(grad_norms, alpha=0.7, color='purple')
        
        if max_grad_norm is not None:
            ax.axhline(y=max_grad_norm, color='r', linestyle='--', 
                       alpha=0.7, label='Clip threshold')
        
        ax.set_xlabel('Batch')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Norm')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # ===== Plot 4: Loss by class OR Annealing schedules =====
    if has_class_loss:
        ax = axes[plot_idx]
        colors = plt.cm.tab10(np.linspace(0, 1, history.n_classes))
        
        for k in range(history.n_classes):
            losses = history.loss_by_class[k]
            if len(losses) > 0:
                ax.plot(losses, alpha=0.2, color=colors[k])
                
                class_window = min(20, len(losses) // 5 + 1)
                if len(losses) >= class_window:
                    smoothed = np.convolve(losses, 
                                          np.ones(class_window)/class_window, mode='valid')
                    ax.plot(range(class_window-1, len(losses)), smoothed,
                           color=colors[k], linewidth=2, label=f'Class {k}')
                else:
                    ax.plot(losses, color=colors[k], linewidth=2, label=f'Class {k}')
        
        ax.set_xlabel('Sample (per class)')
        ax.set_ylabel('Loss')
        ax.set_title('Loss by Class')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    elif has_annealing:
        ax = axes[plot_idx]
        
        if len(history.temperature_history) > 0:
            ax.plot(history.temperature_history, 'b-', linewidth=2, label='Temperature')
        
        if len(history.noise_scale_history) > 0:
            ax2 = ax.twinx()
            ax2.plot(history.noise_scale_history, 'orange', linewidth=2, label='Noise scale')
            ax2.set_ylabel('Noise Scale', color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')
        
        ax.set_xlabel('Batch')
        ax.set_ylabel('Temperature', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        ax.set_title('Annealing Schedules')
        ax.grid(True, alpha=0.3)
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        if len(history.noise_scale_history) > 0:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        else:
            ax.legend()
        
        plot_idx += 1
    
    # ===== Regularizer metrics (e.g. hidden IPR) =====
    if has_regularizers:
        ax = axes[plot_idx]
        plot_regularizer_history(
            history,
            ax=ax,
            title='Regularizer Metrics',
            show=False,
        )
        plot_idx += 1
    
    # Hide unused axes
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig, axes


def plot_comparison(histories: Dict[str, TrainingHistory],
                    metric: str = 'loss',
                    title: str = None,
                    smoothing_window: int = 50,
                    figsize: Tuple[int, int] = (10, 6),
                    show: bool = True):
    """
    Plot comparison of multiple training runs.
    
    Args:
        histories: dict mapping run name to TrainingHistory
        metric: 'loss' or 'accuracy'
        title: Plot title
        smoothing_window: Smoothing window size
        figsize: Figure size
        show: Whether to call plt.show()
    
    Returns:
        fig, ax: matplotlib figure and axis
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))
    
    for i, (name, history) in enumerate(histories.items()):
        data = history.loss_history if metric == 'loss' else history.accuracy_history
        
        if len(data) == 0:
            continue
        
        ax.plot(data, alpha=0.2, color=colors[i])
        
        if len(data) >= smoothing_window:
            smoothed = np.convolve(data, np.ones(smoothing_window)/smoothing_window, mode='valid')
            ax.plot(range(smoothing_window-1, len(data)), smoothed,
                   color=colors[i], linewidth=2, label=name)
        else:
            ax.plot(data, color=colors[i], linewidth=2, label=name)
    
    ax.set_xlabel('Batch')
    ax.set_ylabel(metric.capitalize())
    ax.set_title(title or f'Training {metric.capitalize()} Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if metric == 'accuracy':
        ax.set_ylim([0, 1.05])
        n_classes = list(histories.values())[0].n_classes
        ax.axhline(y=1/n_classes, color='k', linestyle='--', alpha=0.5, label='Random')
    
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig, ax

def generate_lognormal_mixture(
    n_classes: int,
    n_samples_per_class: int,
    input_dim: int,
    log_means: list,
    log_variances: list,
    random_state: int = None
):
    if random_state is not None:
        np.random.seed(random_state)
    
    data_list = []
    for c in range(n_classes):
        mu = np.array(log_means[c]).flatten()
        var = log_variances[c]
        if np.isscalar(var):
            cov = var * np.eye(input_dim)
        else:
            cov = np.diag(np.array(var).flatten())
        
        log_samples = np.random.multivariate_normal(mu, cov, n_samples_per_class)
        samples = np.power(10, log_samples)  # Changed from np.exp to base-10
        data_list.append([sample for sample in samples])
    
    return n_classes, data_list


def _assign_modes_to_labels(n_modes: int, n_classes: int, rng=None) -> list:
    """Assign modes to class labels as evenly as possible; remainder is random."""
    base, remainder = divmod(n_modes, n_classes)
    mode_labels = []
    for label in range(n_classes):
        mode_labels.extend([label] * base)
    if remainder > 0:
        extra_labels = (rng or np.random).choice(n_classes, size=remainder, replace=False)
        mode_labels.extend(extra_labels.tolist())
    (rng or np.random).shuffle(mode_labels)
    return mode_labels


def _merge_modes_to_labels(mode_data_list: list, mode_labels: list, n_classes: int) -> list:
    """Concatenate per-mode samples into per-label lists."""
    merged = [[] for _ in range(n_classes)]
    for mode_idx, samples in enumerate(mode_data_list):
        merged[mode_labels[mode_idx]].extend(samples)
    return merged


def convert_input_data_to_log_scale(input_data):
    """
    Convert InputData from linear scale to log scale.
    
    Takes an InputData object where data is in linear space (after 10^x transform)
    and returns a new InputData object with log10-transformed data.
    
    Args:
        input_data: InputData object with data in linear scale
        
    Returns:
        InputData object with data in log10 scale
    """
    from CRNs.utils import InputData
    
    log_data_list = []
    for class_data in input_data.data_list:
        log_class_data = []
        for sample in class_data:
            log_sample = np.log10(np.asarray(sample))  # Changed from np.log to np.log10
            log_class_data.append(log_sample)
        log_data_list.append(log_class_data)
    
    return InputData(
        n_classes=input_data.n_classes,
        data_list=log_data_list,
        split_fac=input_data.split_fac
    )


def generate_lognormal_mixture_lattice_centers(
    n_classes: int,
    n_samples_per_class: int,
    input_dim: int = 1,
    center_variance: float = 1.0,
    log_variance: float = 0.3,
    center_offset: float = 3.0,
    n_modes: int = None,
    random_state: int = None,
    p_mode_keep: float = 1.0,
    merge_modes: bool = True,
):
    """
    Generate lognormal mixture with centers arranged in a regular hyperlattice in log10-space.
    
    The number of modes must be a perfect power of the dimension (e.g., 8 = 2^3 for 3D).
    Centers are arranged in a regular grid with spacing determined by center_variance.
    Within each class, points are drawn with log_variance spread.
    Data is sampled in log10-space and then transformed via 10^x.
    
    Args:
        n_classes: Number of output classes
        n_samples_per_class: Samples per mode
        input_dim: Dimensionality of the data
        center_variance: TOTAL variance for lattice spacing in log10-space (divided by input_dim)
        log_variance: TOTAL variance for sampling within each class (divided by input_dim)
        center_offset: Center position in log10-space
        n_modes: Number of Gaussian modes. Must be a perfect power of input_dim
        random_state: Random seed
        p_mode_keep: Fraction of modes to keep (1.0=all, 0.5=randomly keep half)
        merge_modes: if False, return one list per mode instead of merging into n_classes
        
    Returns:
        n_classes, data_list, log_means (log_means are in log10-space)
    """
    if random_state is not None:
        np.random.seed(random_state)

    if n_modes is None:
        n_modes = n_classes
    elif n_modes < n_classes:
        raise ValueError(f"n_modes ({n_modes}) must be >= n_classes ({n_classes})")
    
    # Check that n_modes is a perfect power of input_dim
    grid_size_per_dim = round(n_modes ** (1.0 / input_dim))
    if grid_size_per_dim ** input_dim != n_modes:
        raise ValueError(
            f"n_modes ({n_modes}) must be a perfect {input_dim}-th power. "
            f"For dim={input_dim}, try n_modes={grid_size_per_dim**input_dim} "
            f"(={grid_size_per_dim}^{input_dim})"
        )
    
    # Scale variances by dimension
    per_dim_center_variance = center_variance / input_dim
    per_dim_log_variance = log_variance / input_dim
    
    # Generate lattice coordinates
    # Create grid points from 0 to (grid_size_per_dim - 1) in each dimension
    grid_coords = np.meshgrid(*[np.arange(grid_size_per_dim) for _ in range(input_dim)], indexing='ij')
    grid_coords = np.stack([g.flatten() for g in grid_coords], axis=1)
    
    # Center the grid and scale by center_variance
    # The lattice spacing is sqrt(per_dim_center_variance)
    lattice_spacing = np.sqrt(per_dim_center_variance)
    grid_coords_centered = (grid_coords - (grid_size_per_dim - 1) / 2.0) * lattice_spacing
    
    # Generate log_means at lattice points
    log_means = []
    for i in range(n_modes):
        log_mean = grid_coords_centered[i] + center_offset
        log_means.append(log_mean)
    
    # Randomly delete modes if p_mode_keep < 1.0
    if p_mode_keep < 1.0:
        n_modes_full = len(log_means)
        n_modes_keep = max(1, int(np.round(n_modes_full * p_mode_keep)))
        
        # Randomly select which modes to keep (use random_state + 1 for mode selection)
        rng = np.random.default_rng(random_state + 1 if random_state is not None else None)
        kept_indices = np.sort(rng.choice(n_modes_full, n_modes_keep, replace=False))
        
        # Filter log_means
        log_means = [log_means[i] for i in kept_indices]
        n_modes = len(log_means)
    
    # Use per_dim_log_variance for the spread within each mode
    log_variances = [per_dim_log_variance] * n_modes
    
    _, mode_data_list = generate_lognormal_mixture(
        n_classes=n_modes,
        n_samples_per_class=n_samples_per_class,
        input_dim=input_dim,
        log_means=log_means,
        log_variances=log_variances,
        random_state=None
    )

    if (not merge_modes) or n_modes == n_classes:
        data_list = mode_data_list
    else:
        mode_labels = _assign_modes_to_labels(n_modes, n_classes)
        data_list = _merge_modes_to_labels(mode_data_list, mode_labels, n_classes)
    
    return n_classes, data_list, log_means


def generate_lognormal_mixture_random_centers(
    n_classes: int,
    n_samples_per_class: int,
    input_dim: int = 1,
    center_variance: float = 1.0,
    log_variance: float = 0.3,
    center_offset: float = 3.0,
    n_modes: int = None,
    random_state: int = None,
    merge_modes: bool = True,
):
    """
    Generate lognormal mixture with randomly placed centers in log10-space.
    
    Both center_variance and log_variance are TOTAL variances.
    They are divided by input_dim to get per-dimension variance,
    ensuring comparable spread across different dimensionalities.
    Data is sampled in log10-space and then transformed via 10^x.

    Args:
        n_modes: Number of Gaussian modes (clouds) in log10-space. Defaults to
            n_classes (one mode per label). When n_modes > n_classes, modes are
            assigned to labels as evenly as possible; any remainder modes are
            assigned to randomly chosen labels. n_samples_per_class is per mode.
        merge_modes: if False, return one list per mode instead of merging into n_classes
            
    Returns:
        n_classes, data_list, log_means (log_means are in log10-space)
    """
    if random_state is not None:
        np.random.seed(random_state)

    if n_modes is None:
        n_modes = n_classes
    elif n_modes < n_classes:
        raise ValueError(f"n_modes ({n_modes}) must be >= n_classes ({n_classes})")
    
    # Scale both by dimension for comparable total variance
    per_dim_center_variance = center_variance / input_dim
    per_dim_log_variance = log_variance / input_dim
    
    log_means = []
    for _ in range(n_modes):
        center = np.random.normal(0, np.sqrt(per_dim_center_variance), size=input_dim)
        log_mean = center + center_offset
        log_means.append(log_mean)
    
    log_variances = [per_dim_log_variance] * n_modes
    
    _, mode_data_list = generate_lognormal_mixture(
        n_classes=n_modes,
        n_samples_per_class=n_samples_per_class,
        input_dim=input_dim,
        log_means=log_means,
        log_variances=log_variances,
        random_state=None
    )

    if (not merge_modes) or n_modes == n_classes:
        data_list = mode_data_list
    else:
        mode_labels = _assign_modes_to_labels(n_modes, n_classes)
        data_list = _merge_modes_to_labels(mode_data_list, mode_labels, n_classes)
    
    return n_classes, data_list, log_means


def generate_cartesian_alphabet_data(
    Q: int,
    L: int,
    N: int,
    C: int,
    M: int,
    n_samples_per_class: int,
    log_variance: float = 0.3,
    center_variance: float = 1.0,
    random_state=None,
):
    """
    Build LN-dimensional training data from concatenated alphabet vectors.
    Data is sampled in log10-space and transformed via 10^x.

    Hyperparameters
    ---------------
    Q : alphabet size
    L : number of vector blocks (psi_l for l = 0..L-1)
    N : length of each block vector
    C : number of class labels
    M : distinct vectors sampled per block (M <= Q^N)
    n_samples_per_class : total noisy samples per class label
    log_variance : total log10-space variance for sampling within each center
    center_variance : total variance for alphabet/center separation in log10-space

    Returns
    -------
    C, data_list, log_means, alphabet, psi, mode_labels
        data_list[c]     = n_samples_per_class noisy samples for class c
        log_means[k]     = log10 of the k-th concatenated center (length L*N)
        psi[l]           = (M, N) sampled vectors for block l
        mode_labels[k]   = class label assigned to center k
    """
    rng = np.random.default_rng(random_state)

    input_dim = L * N
    n_possible = Q ** N
    if M > n_possible:
        raise ValueError(f"M={M} exceeds Q^N={n_possible}")

    log_spacing = np.sqrt(center_variance / input_dim)
    log_alphabet = (np.arange(Q) - (Q - 1) / 2.0) * log_spacing
    alphabet = np.power(10, log_alphabet)  # Changed from np.exp to base-10

    def index_to_vector(flat_idx: int) -> np.ndarray:
        digits = np.unravel_index(flat_idx, (Q,) * N)
        return alphabet[np.asarray(digits)]

    psi = []
    for _ in range(L):
        chosen = rng.choice(n_possible, size=M, replace=False)
        psi.append(np.stack([index_to_vector(i) for i in chosen], axis=0))

    centers = []
    for combo in product(range(M), repeat=L):
        blocks = [psi[l][combo[l]] for l in range(L)]
        centers.append(np.concatenate(blocks))
    centers = np.asarray(centers)

    n_modes = len(centers)
    if n_modes < C:
        raise ValueError(f"M^L={n_modes} must be >= C={C} for balanced label assignment")

    mode_labels = _assign_modes_to_labels(n_modes, C, rng)
    log_means = [np.log10(center) for center in centers]  # Changed from np.log to np.log10

    per_dim_var = log_variance / input_dim
    cov = per_dim_var * np.eye(input_dim)

    data_list = [[] for _ in range(C)]
    for c in range(C):
        center_idxs = np.flatnonzero(np.asarray(mode_labels) == c)
        chosen_centers = rng.choice(center_idxs, size=n_samples_per_class, replace=True)
        for k in chosen_centers:
            log_sample = rng.multivariate_normal(log_means[k], cov)
            data_list[c].append(np.power(10, log_sample))  # Changed from np.exp to base-10

    return C, data_list, log_means, alphabet, psi, mode_labels


def project_data_list(data_list, d):

    projected_data_list = []
    for class_data in data_list:
        projected_class = [sample[:d] for sample in class_data]
        projected_data_list.append(projected_class)
    return projected_data_list


# ============== MULTI-TASK LEARNING ==============

class MultiTaskInputData:
    """Container for multiple classification tasks with task-specific l0 values."""
    
    def __init__(self, tasks: Dict[str, dict]):
        """
        Args:
            tasks: dict mapping task_id -> {
                'input_data': InputData instance,
                'l0': np.array of l0 values,
                'n_classes': int,
                'log_means': list (optional, for reproducibility)
            }
        """
        self.tasks = tasks
        self.task_ids = list(tasks.keys())
        self.n_tasks = len(self.task_ids)
    
    def sample_task(self) -> str:
        """Randomly sample a task id."""
        import random
        return random.choice(self.task_ids)
    
    def get_task(self, task_id: str) -> dict:
        """Get task info by id."""
        return self.tasks[task_id]
    
    def get_l0(self, task_id: str) -> np.ndarray:
        """Get l0 values for a task."""
        return self.tasks[task_id]['l0']
    
    def get_input_data(self, task_id: str):
        """Get InputData for a task."""
        return self.tasks[task_id]['input_data']
    
    def get_next_training_sample(self, task_id: str, class_idx: int):
        """Get next training sample from a specific task."""
        return self.tasks[task_id]['input_data'].get_next_training_sample(class_idx)
    
    def get_n_classes(self, task_id: str) -> int:
        """Get number of classes for a task."""
        return self.tasks[task_id]['n_classes']


def generate_multitask_data(
    n_tasks: int,
    n_classes: int,
    n_samples_per_class: int,
    input_dim: int,
    proj_dim: int,
    center_variance: float,
    log_variance: float,
    center_offset: float,
    n_nodes: int,
    NR: int,
    hidden_dim: int,
    input_data_class,
    l0_log_mean: float = 0.0,
    l0_log_std: float = 1.0,
    base_seed: int = None,
    permute_labels_only: bool = False,
    resample_mode_labels: bool = False,
    data_gen_method: str = 'random',
    n_modes: int = None,
    p_mode_keep: float = 1.0,
    log_scale: bool = True,
) -> MultiTaskInputData:
    """
    Generate multi-task data with random l0 values for internal nodes.
    
    Args:
        n_tasks: number of tasks to generate
        n_classes: number of classes per task
        n_samples_per_class: samples per class
        input_dim: original input dimension (before projection)
        proj_dim: projected dimension (= NR)
        center_variance: variance for class center generation
        log_variance: variance within each class
        center_offset: offset for log means
        n_nodes: total number of conservation-law / graph nodes
        NR: number of input (receptor) nodes
        hidden_dim: unused for l0 slicing; kept for API compatibility
        input_data_class: InputData class to use for wrapping data
        l0_log_mean: mean of log(l0) for internal nodes
        l0_log_std: std of log(l0) for internal nodes
        base_seed: random seed for reproducibility
        permute_labels_only: keep the same class clouds and only permute class
            IDs. Mode groups that were merged into a class stay together
            (a class-id swap). Ignored when resample_mode_labels is True.
        resample_mode_labels: keep the same modes but redraw which modes map
            to which class on each task. Modes that shared a label on one
            task may be split or regrouped on another. Only differs from
            permute_labels_only when n_modes > n_classes; if n_modes ==
            n_classes both options are a label permutation.
        data_gen_method: 'random' (GMM) or 'lattice'
        n_modes: number of mixture modes (defaults to n_classes)
        p_mode_keep: fraction of lattice modes to keep
        log_scale: if False, convert samples to log10 space after generation
        
    Returns:
        MultiTaskInputData container
    """
    import random as random_module

    def _generate_clouds(random_state, merge_modes=True):
        kwargs = dict(
            n_classes=n_classes,
            n_modes=n_modes,
            n_samples_per_class=n_samples_per_class,
            input_dim=input_dim,
            center_variance=center_variance,
            log_variance=log_variance,
            center_offset=center_offset,
            random_state=random_state,
            merge_modes=merge_modes,
        )
        if data_gen_method == 'lattice':
            _, data_list, log_means = generate_lognormal_mixture_lattice_centers(
                p_mode_keep=p_mode_keep, **kwargs
            )
        else:
            _, data_list, log_means = generate_lognormal_mixture_random_centers(**kwargs)
        data_list = project_data_list(data_list, d=proj_dim)
        return data_list, log_means

    def _wrap_input_data(data_list):
        input_data = input_data_class(n_classes, data_list)
        if not log_scale:
            input_data = convert_input_data_to_log_scale(input_data)
        return input_data
    
    tasks = {}
    share_clouds = permute_labels_only or resample_mode_labels
    
    # Generate modes/class clouds once; each task only remaps labels
    if share_clouds:
        data_seed = base_seed if base_seed is not None else 42
        np.random.seed(data_seed)
        random_module.seed(data_seed)
        # resample: keep one list per mode so they can be regrouped.
        # permute: merge into classes now, then only swap class IDs.
        base_data_list, base_log_means = _generate_clouds(
            data_seed, merge_modes=not resample_mode_labels
        )
    
    for task_idx in range(n_tasks):
        # Set seed for this task if base_seed provided
        task_seed = base_seed + task_idx if base_seed is not None else None
        
        if task_seed is not None:
            np.random.seed(task_seed)
            random_module.seed(task_seed)
        
        permutation = None
        mode_labels = None

        if resample_mode_labels:
            # Same modes; new partition of modes into classes
            n_modes_actual = len(base_data_list)
            mode_labels = _assign_modes_to_labels(n_modes_actual, n_classes)
            data_list = _merge_modes_to_labels(base_data_list, mode_labels, n_classes)
            log_means = base_log_means
            input_data = _wrap_input_data(data_list)
        elif permute_labels_only:
            # Same class bundles; only the class IDs are shuffled
            permutation = list(range(n_classes))
            random_module.shuffle(permutation)
            
            # permutation[new_label] = old_label: class new_label gets the
            # entire old_label cloud (all modes already merged into that class)
            permuted_data_list = [base_data_list[permutation[i]] for i in range(n_classes)]
            if len(base_log_means) == n_classes:
                log_means = [base_log_means[permutation[i]] for i in range(n_classes)]
            else:
                # Per-mode means (e.g. lattice with n_modes != n_classes)
                log_means = base_log_means
            
            input_data = _wrap_input_data(permuted_data_list)
        else:
            # Fresh clouds (new centers) for this task
            data_list, log_means = _generate_clouds(task_seed)
            input_data = _wrap_input_data(data_list)
        
        # Generate l0 values. Randomize non-input, non-output conservation laws
        # (matches CRNModel.l0_train_range; covers hidden_depth > 1 and MP extras).
        l0 = np.ones(n_nodes)
        internal_start = NR
        internal_end = n_nodes - n_classes
        n_internal = max(0, internal_end - internal_start)
        
        # Log-normal distribution for l0 at internal nodes
        if n_internal > 0:
            log_l0_internal = np.random.randn(n_internal) * l0_log_std + l0_log_mean
            l0[internal_start:internal_end] = np.exp(log_l0_internal)
        
        # Store task
        task_id = f"task_{task_idx}"
        tasks[task_id] = {
            'input_data': input_data,
            'l0': l0,
            'n_classes': n_classes,
            'log_means': log_means,
            'seed': task_seed,
            'permutation': permutation,  # Store permutation for debugging
            'mode_labels': mode_labels,
        }
    
    return MultiTaskInputData(tasks)


def run_training_crn_multitask(
    trainer: UnifiedTrainer,
    multi_task_data: MultiTaskInputData,
    n_classes: int,
    num_batches: int,
    batch_size: int,
    T_start: float = 1.0,
    T_end: float = 0.2,
    T_decay: float = 0.99,
    noise_start: float = 5.0,
    noise_end: float = 0.0,
    noise_decay: float = 0.99,
    print_every: int = 50,
    history: 'TrainingHistory' = None,
    verbose: bool = True,
    debug_l0: bool = False
) -> 'TrainingHistory':
    """
    Run training loop for CRN models with multiple tasks.
    
    Each task has its own data distribution and l0 values.
    The model learns shared rates across all tasks.
    
    Args:
        trainer: UnifiedTrainer instance (should have frozen_params=['log_l0'])
        multi_task_data: MultiTaskInputData with task-specific data and l0
        n_classes: number of classes (assumed same for all tasks)
        num_batches: number of batches to train
        batch_size: samples per batch
        T_start: initial softmax temperature
        T_end: final softmax temperature
        T_decay: temperature decay rate per batch
        noise_start: initial gradient noise scale
        noise_end: final gradient noise scale
        noise_decay: noise decay rate per batch
        print_every: print diagnostics every N batches
        history: TrainingHistory instance (created if None)
        verbose: whether to print progress
        debug_l0: if True, print l0 values for first few samples to verify task switching
        
    Returns:
        TrainingHistory with recorded metrics
    """
    import time
    import random
    
    if history is None:
        history = TrainingHistory(n_classes)
    
    start_time = time.time()
    
    if verbose:
        if hasattr(trainer.model, 'forward_method'):
            model_type = f"CRN ({trainer.model.forward_method})"
        elif hasattr(trainer.model, 'mlp'):
            model_type = "MLP"
        else:
            model_type = "model"
        print(f"Starting {model_type} multi-task training: {num_batches} batches, "
              f"batch_size={batch_size}, n_tasks={multi_task_data.n_tasks}")
        print("=" * 80)
    
    for batch in range(num_batches):
        # Annealing schedules
        temperature = max(T_end, T_start * (T_decay ** batch))
        noise_scale = max(noise_end, noise_start * (noise_decay ** batch))
        
        batch_loss = 0.0
        batch_correct = 0
        batch_valid = 0
        batch_grad_norms = None
        batch_reg_values = {}  # Accumulate regularizer values
        
        for sample in range(batch_size):
            # Sample a task
            task_id = multi_task_data.sample_task()
            
            # Set l0 for this task (context switch). MLP has no per-task l0.
            if hasattr(trainer.model, '_default_l0'):
                trainer.model._default_l0 = multi_task_data.get_l0(task_id).copy()
            
            # Debug: verify l0 is being set correctly
            if debug_l0 and batch == 0 and sample < 5 and hasattr(trainer.model, '_default_l0'):
                n_inputs = trainer.model.n_inputs
                n_classes_model = trainer.model.n_classes
                internal_l0 = trainer.model._default_l0[n_inputs:-n_classes_model] if n_classes_model > 0 else trainer.model._default_l0[n_inputs:]
                print(f"  DEBUG: {task_id}, sample {sample}: internal l0 = {internal_l0[:4]}...")  # First 4 values
            
            # Sample class and get input from this task
            target_idx = random.randrange(n_classes)
            inputs = multi_task_data.get_next_training_sample(task_id, target_idx)
            
            try:
                loss, probs, grad_norms, reg_values = trainer.train_step(
                    inputs=inputs,
                    target_idx=target_idx,
                    temperature=temperature,
                    noise_scale=noise_scale
                )
                
                # Check for numerical issues
                if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
                    if verbose:
                        print(f"  Warning: Invalid probs at batch {batch}, sample {sample}")
                    continue
                
                # Check for invalid gradients
                has_invalid_grad = any(
                    np.any(np.isnan(g)) or np.any(np.isinf(g)) 
                    for g in grad_norms.values()
                )
                if has_invalid_grad:
                    if verbose:
                        print(f"  Warning: Invalid gradient at batch {batch}, sample {sample}")
                    continue
                
                batch_loss += loss
                history.record_sample_loss(target_idx, loss)
                batch_correct += int(trainer.compute_accuracy(probs, target_idx))
                batch_valid += 1
                batch_grad_norms = grad_norms
                
                # Accumulate regularizer values (multitask version)
                for reg_name, reg_value in reg_values.items():
                    if reg_name not in batch_reg_values:
                        batch_reg_values[reg_name] = 0.0
                    batch_reg_values[reg_name] += reg_value
                
            except TimeoutError:
                continue
            except Exception as e:
                if verbose:
                    print(f"  Warning: Training step failed at batch {batch}, sample {sample}: {e}")
                continue
        
        # Skip if no valid samples
        if batch_valid == 0:
            if verbose:
                print(f"  Batch {batch}: No valid samples, skipping")
            continue
        
        # Get param stats (for CRN, show rate range)
        params = trainer.model.get_params()
        if 'log_rates' in params:
            rates = np.exp(params['log_rates'])
            param_stats = (rates.min(), rates.max(), rates.mean())
        else:
            param_values = np.concatenate([p.flatten() for p in params.values()])
            param_stats = (param_values.min(), param_values.max(), param_values.mean())
        
        # Average regularizer values
        avg_reg_values = {name: val / batch_valid for name, val in batch_reg_values.items()}
        
        # Get current regularization schedule factor
        reg_schedule_factor = trainer.get_regularizer_schedule_factor(batch)
        
        # Record batch metrics
        history.record_batch(
            avg_loss=batch_loss / batch_valid,
            accuracy=batch_correct / batch_valid,
            grad_norms=batch_grad_norms if batch_grad_norms else {},
            temperature=temperature,
            noise_scale=noise_scale,
            param_stats=param_stats,
            regularizer_values=avg_reg_values,
            reg_schedule_factor=reg_schedule_factor
        )
        
        # Print diagnostics
        if verbose and (batch % print_every == 0 or batch == num_batches - 1):
            recent_loss = history.get_recent_avg('loss')
            recent_acc = history.get_recent_avg('accuracy')
            
            grad_str = ""
            if batch_grad_norms:
                grad_str = " ".join([f"{k[:6]}:{v:.2e}" for k, v in batch_grad_norms.items()])
            
            # Build regularizer string (multitask version)
            reg_str = ""
            if avg_reg_values:
                reg_str = " | " + ", ".join([f"{k}:{v:.4f}" for k, v in avg_reg_values.items()])
            
            # Add regularization schedule info if using schedule
            if trainer.reg_schedule_type != 'none' and avg_reg_values:
                reg_str += f" | RegSched: {reg_schedule_factor:.2f}"
            
            pmin, pmax, _ = param_stats
            print(f"Batch {batch:4d}/{num_batches} | "
                  f"Loss: {batch_loss/batch_valid:.4f} (avg: {recent_loss:.4f}) | "
                  f"Acc: {batch_correct/batch_valid:.1%} (avg: {recent_acc:.1%}) | "
                  f"{grad_str} | "
                  f"Rates: [{pmin:.2e}, {pmax:.2e}]"
                  f"{reg_str}")
    
    training_time = time.time() - start_time
    if hasattr(trainer.model, 'n_integration_timeouts'):
        history.n_integration_timeouts = trainer.model.n_integration_timeouts
    
    if verbose:
        print("=" * 80)
        history.print_summary(training_time)
    
    return history


# class GraphComputationJIT:
#     """Optimized computational graph using JAX arrays and JIT compilation."""
    
#     def __init__(self, G, input_nodes, output_nodes):
#         self.G = G
#         self.input_nodes = input_nodes
#         self.output_nodes = output_nodes
#         self.topo_order = list(nx.topological_sort(G))
#         self.edges = list(G.edges)
#         self.nodes = list(G.nodes)
#         self.n_nodes = len(self.nodes)
#         self.n_edges = len(self.edges)
        
#         # Create node index mappings
#         self.node_to_idx = {n: i for i, n in enumerate(self.nodes)}
#         self.edge_to_idx = {e: i for i, e in enumerate(self.edges)}
        
#         # Pre-compute indices for fast access
#         self.input_idxs = jnp.array([self.node_to_idx[n] for n in input_nodes])
#         self.output_idxs = jnp.array([self.node_to_idx[n] for n in output_nodes])
        
#         # Pre-compute NON-INPUT nodes in topological order (avoid conditional in JIT)
#         input_set = set(input_nodes)
#         self.compute_node_idxs = tuple(
#             self.node_to_idx[n] for n in self.topo_order if n not in input_set
#         )
        
#         # Pre-compute predecessor structure as arrays
#         self._build_predecessor_arrays()
        
#     def _build_predecessor_arrays(self):
#         """Pre-compute predecessor indices for vectorized access."""
#         max_preds = max((len(list(self.G.predecessors(n))) for n in self.nodes), default=1)
#         max_preds = max(max_preds, 1)
        
#         self.pred_node_idxs = np.full((self.n_nodes, max_preds), 0, dtype=np.int32)
#         self.pred_edge_idxs = np.full((self.n_nodes, max_preds), 0, dtype=np.int32)
#         self.pred_mask = np.zeros((self.n_nodes, max_preds), dtype=bool)
        
#         for node in self.nodes:
#             node_idx = self.node_to_idx[node]
#             preds = list(self.G.predecessors(node))
            
#             for j, pred in enumerate(preds):
#                 pred_idx = self.node_to_idx[pred]
#                 edge_idx = self.edge_to_idx[(pred, node)]
#                 self.pred_node_idxs[node_idx, j] = pred_idx
#                 self.pred_edge_idxs[node_idx, j] = edge_idx
#                 self.pred_mask[node_idx, j] = True
        
#         self.pred_node_idxs = jnp.array(self.pred_node_idxs)
#         self.pred_edge_idxs = jnp.array(self.pred_edge_idxs)
#         self.pred_mask = jnp.array(self.pred_mask)

#     def build_r_n_maps(self, r_n):
#         """Build index mappings from reaction network."""
#         node_f_idx = np.zeros(self.n_nodes, dtype=np.int32)
#         node_r_idx = np.zeros(self.n_nodes, dtype=np.int32)
#         edge_f_idx = np.zeros(self.n_edges, dtype=np.int32)
#         edge_r_idx = np.zeros(self.n_edges, dtype=np.int32)

#         for (i, reaction) in enumerate(r_n.reactions):
#             src = r_n.all_complexes[reaction[0]].split('+')
#             dst = r_n.all_complexes[reaction[1]].split('+')
#             num_src, num_dst = len(src), len(dst)
            
#             if num_src == 1 and num_dst == 1:
#                 if src[0].endswith('s'):
#                     node = src[0][:-1]
#                     if node in self.node_to_idx:
#                         node_f_idx[self.node_to_idx[node]] = i
#                 if dst[0].endswith('s'):
#                     node = dst[0][:-1]
#                     if node in self.node_to_idx:
#                         node_r_idx[self.node_to_idx[node]] = i

#             if num_src == 2 and num_dst == 2:
#                 src_set, dst_set = set(src), set(dst)
#                 common = src_set & dst_set
                
#                 if len(common) == 1:
#                     upstream_node = common.pop()
#                     src_only = (src_set - {upstream_node}).pop()
#                     dst_only = (dst_set - {upstream_node}).pop()
                    
#                     if src_only.endswith('s') and not dst_only.endswith('s'):
#                         downstream_node = dst_only
#                         is_forward = True
#                     elif dst_only.endswith('s') and not src_only.endswith('s'):
#                         downstream_node = src_only
#                         is_forward = False
#                     else:
#                         continue
                    
#                     edge_key = (upstream_node, downstream_node)
#                     if edge_key in self.edge_to_idx:
#                         edge_idx = self.edge_to_idx[edge_key]
#                         if is_forward:
#                             edge_f_idx[edge_idx] = i
#                         else:
#                             edge_r_idx[edge_idx] = i
        
#         self.node_f_idx = jnp.array(node_f_idx)
#         self.node_r_idx = jnp.array(node_r_idx)
#         self.edge_f_idx = jnp.array(edge_f_idx)
#         self.edge_r_idx = jnp.array(edge_r_idx)
        
#         self._compile_forward()

#     def _compile_forward(self):
#         """Create JIT-compiled forward pass."""
        
#         # Static values (known at compile time)
#         compute_node_idxs = self.compute_node_idxs  # tuple = static
#         input_idxs = self.input_idxs
#         output_idxs = self.output_idxs
#         pred_node_idxs = self.pred_node_idxs
#         pred_edge_idxs = self.pred_edge_idxs
#         pred_mask = self.pred_mask
#         node_f_idx = self.node_f_idx
#         node_r_idx = self.node_r_idx
#         edge_f_idx = self.edge_f_idx
#         edge_r_idx = self.edge_r_idx
#         n_nodes = self.n_nodes
        
#         #@jax.jit
#         def _forward_jit(rates, l0, input_vals):
#             """
#             rates: (n_reactions,) array of rate constants
#             l0: (n_nodes,) array of conservation constants
#             input_vals: (n_inputs,) array of input values
#             """
#             # Initialize node values with inputs
#             node_values = jnp.zeros(n_nodes)
#             node_values = node_values.at[input_idxs].set(input_vals)
            
#             # Get rate parameters via indexing
#             node_kf = rates[node_f_idx]
#             node_kr = rates[node_r_idx]
#             edge_kf = rates[edge_f_idx]
#             edge_kr = rates[edge_r_idx]
            
#             # Process only non-input nodes (static tuple, unrolled by JIT)
#             for node_idx in compute_node_idxs:
#                 kf_node = node_kf[node_idx]
#                 kr_node = node_kr[node_idx]
#                 numerator = kf_node
#                 denominator = kf_node + kr_node
                
#                 # Vectorized predecessor contribution
#                 pred_nodes = pred_node_idxs[node_idx]
#                 pred_edges = pred_edge_idxs[node_idx]
#                 mask = pred_mask[node_idx]
                
#                 pred_vals = jnp.where(mask, node_values[pred_nodes], 0.0)
#                 kf_edges = jnp.where(mask, edge_kf[pred_edges], 0.0)
#                 kr_edges = jnp.where(mask, edge_kr[pred_edges], 0.0)
#                 kt_edges = kf_edges + kr_edges
                
#                 numerator = numerator + jnp.sum(kf_edges * pred_vals)
#                 denominator = denominator + jnp.sum(kt_edges * pred_vals)
                
#                 node_val = l0[node_idx] * numerator / (denominator + 1e-10)
#                 node_values = node_values.at[node_idx].set(node_val)
            
#             return node_values[output_idxs]
        
#         self._forward_jit = _forward_jit
    
#     def forward(self, rates, l0, input_vals):
#         return self._forward_jit(
#             jnp.asarray(rates),
#             jnp.asarray(l0),
#             jnp.asarray(input_vals)
#         )


# class GraphComputation:
#     """Turn a NetworkX graph into a differentiable computational graph."""
    
#     def __init__(self, G, input_nodes, output_nodes):
#         self.G = G
#         self.input_nodes = input_nodes
#         self.output_nodes = output_nodes
#         self.topo_order = list(nx.topological_sort(G))
#         self.edges = list(G.edges)
#         self.nodes = list(G.nodes)
#         self.n_edges = len(self.edges)
#         self.edge_to_idx = {e: i for i, e in enumerate(self.edges)}

#     def build_r_n_maps(self, r_n):
#         self.node_params_f_map = {}
#         self.node_params_r_map = {}
#         self.edge_params_f_map = {}
#         self.edge_params_r_map = {}
#         self.node_params_f = {}
#         self.node_params_r = {}
#         self.edge_params_f = {}
#         self.edge_params_r = {}

#         for (i, reaction) in enumerate(r_n.reactions):
#             src = r_n.all_complexes[reaction[0]].split('+')
#             dst = r_n.all_complexes[reaction[1]].split('+')
#             rate = reaction[2]
#             num_src = len(src)
#             num_dst = len(dst)
            
#             # Unimolecular reactions: X <-> Xs
#             if num_src == 1 and num_dst == 1:
#                 if src[0].endswith('s'):
#                     node = src[0][:-1]  # remove 's'
#                     #node_params_f[node] = (rate, i)
#                     self.node_params_f_map[node] = i
#                     self.node_params_f[node] = rate
#                 if dst[0].endswith('s'):
#                     node = dst[0][:-1]  # remove 's'
#                     #node_params_r[node] = (rate, i)
#                     self.node_params_r_map[node] = i
#                     self.node_params_r[node] = rate

#             # Bimolecular reactions: A+Bs -> A+B or A+B -> A+Bs
#             if num_src == 2 and num_dst == 2:
#                 src_set = set(src)
#                 dst_set = set(dst)
                
#                 # Find the species that appears on both sides (upstream/catalyst)
#                 common = src_set & dst_set
                
#                 if len(common) == 1:
#                     upstream_node = common.pop()
                    
#                     # Find the species that changes (has 's' on one side)
#                     src_only = (src_set - {upstream_node}).pop()
#                     dst_only = (dst_set - {upstream_node}).pop()
                    
#                     # Determine downstream node and direction
#                     if src_only.endswith('s') and not dst_only.endswith('s'):
#                         # Xs -> X : forward activation, 's' on left
#                         downstream_node = dst_only
#                         is_forward = True
#                     elif dst_only.endswith('s') and not src_only.endswith('s'):
#                         # X -> Xs : reverse reaction, 's' on right
#                         downstream_node = src_only
#                         is_forward = False
#                     else:
#                         print(f"Warning: Unexpected reaction pattern at {i}: {src} -> {dst}")
#                         continue
                    
#                     edge_key = (upstream_node, downstream_node)
                    
#                     if is_forward:
#                         self.edge_params_f_map[edge_key] = i
#                         self.edge_params_f[edge_key] = rate
#                     else:
#                         self.edge_params_r_map[edge_key] = i
#                         self.edge_params_r[edge_key] = rate
                    
#                     # print(f"Reaction {i}: {'+'.join(src)} -> {'+'.join(dst)}")
#                     # print(f"  Upstream: {upstream_node}, Downstream: {downstream_node}, Forward: {is_forward}")
#                 else:
#                     # Both species change or neither - different reaction type
#                     print(f"Reaction {i}: Non-catalytic bimolecular: {'+'.join(src)} -> {'+'.join(dst)}")

#     def build_node_params_l0(self, l0):
#         self.node_params_l0 = {}
#         for (i, node) in enumerate(self.nodes):
#             self.node_params_l0[node] = l0[i]
    
#     def build_rate_params(self, rates):
#         for key in self.node_params_f_map:
#             self.node_params_f[key] = rates[self.node_params_f_map[key]]
#         for key in self.node_params_r_map:
#             self.node_params_r[key] = rates[self.node_params_r_map[key]]
#         for key in self.edge_params_f_map:
#             self.edge_params_f[key] = rates[self.edge_params_f_map[key]]
#         for key in self.edge_params_r_map:
#             self.edge_params_r[key] = rates[self.edge_params_r_map[key]]
            

#     def forward(self, inputs):
#         """
#         params: array of shape (n_edges,) - one parameter per edge
#         inputs: dict mapping input_node -> value
#         """
#         node_values = {n: inputs[n] for n in self.input_nodes}
        
#         for node in self.topo_order:
#             if node in self.input_nodes:
#                 continue
            
#             # Aggregate incoming edges
#             numerator = self.node_params_f[node]
#             denominator = self.node_params_f[node] + self.node_params_r[node]
#             for pred in self.G.predecessors(node):
#                 edge_idx = self.edge_to_idx[(pred, node)]
#                 edge = self.edges[edge_idx]
#                 kf_couple = self.edge_params_f[edge]
#                 kr_couple = self.edge_params_r[edge]
#                 kt_couple = kf_couple + kr_couple
#                 numerator += kf_couple * node_values[pred]
#                 denominator += kt_couple * node_values[pred]
            
#             # Node activation (customize as needed)
#             node_values[node] = self.node_params_l0[node] * numerator / denominator
        
#         return jnp.array([node_values[n] for n in self.output_nodes])

    
#     def loss(self, params, inputs, targets):
#         preds = self.forward(params, inputs)
#         return jnp.mean((preds - targets) ** 2)