"""
Mutual Information training for CRN models.

This module provides training infrastructure for maximizing I(R; z)
where z ~ Bernoulli(h(R)) and h are CRN activation parameters.

Reuses existing training infrastructure while replacing classification
loss with MI objective.
"""

import numpy as np
import random
import time
from typing import Dict, Optional, List, Tuple

from CRNs.training import CRNModel, MLPModel, TrainingHistory
from CRNs.utils import InputData
from CRNs.mi_estimators import (
    MutualInformationEstimator, create_mi_estimator, LogNormalChannelMI,
)
from CRNs.regularizers import compute_activation_fractions, compute_fractional_ipr


class MITrainingHistory(TrainingHistory):
    """
    Extended training history for MI optimization.
    
    Tracks MI values in addition to standard training metrics.
    """
    
    def __init__(self, n_classes: int = 1):
        """
        Initialize MI-specific history tracking.
        
        Args:
            n_classes: Number of classes (for compatibility, default=1 for MI)
        """
        super().__init__(n_classes=n_classes)
        self.mi_values = []
        self.h_statistics = []  # Track h distribution statistics
        self.ipr_values = []  # Track IPR (sparsity) of activations
    
    def record_batch_mi(self, mi_value: float, h_mean: np.ndarray, h_std: np.ndarray, h_ipr: float):
        """Record MI value, h statistics, and IPR for a batch."""
        self.mi_values.append(mi_value)
        self.h_statistics.append({
            'mean': h_mean.copy(),
            'std': h_std.copy()
        })
        self.ipr_values.append(h_ipr)
    
    def get_recent_mi(self, window: int = 10) -> float:
        """Get average MI over recent batches."""
        if len(self.mi_values) == 0:
            return 0.0
        return np.mean(self.mi_values[-window:])


class MITrainer:
    """
    Trainer for maximizing I(R; z) where z ~ Bernoulli(h(R)).
    
    Supports both CRNModel and MLPModel.
    Reuses optimizer infrastructure from UnifiedTrainer but replaces
    classification loss with MI objective.
    """
    
    def __init__(self,
                 model,  # CRNModel or MLPModel
                 mi_estimator: MutualInformationEstimator,
                 layer_indices: List[int],
                 optimizer_type: str = 'adam',
                 lr: float = 0.01,
                 lr_dict: Optional[Dict[str, float]] = None,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 eps: float = 1e-8,
                 max_grad_norm: float = 50.0,
                 frozen_params: Optional[List[str]] = None,
                 log_input_space: bool = False):
        """
        Args:
            model: CRNModel or MLPModel instance (outputs h ∈ [0,1]^d)
            mi_estimator: MutualInformationEstimator instance
            layer_indices: List of species indices to compute MI over (e.g., hidden_indices or class_ids)
            optimizer_type: 'adam', 'sgd', or 'fletcher_reeves'
            lr: Default learning rate
            lr_dict: Per-parameter learning rates
            beta1, beta2, eps: Adam hyperparameters
            max_grad_norm: Gradient clipping threshold
            frozen_params: Parameters to freeze (e.g., ['log_l0'] for CRN)
            log_input_space: If True, optimize I(ln(R); z) by scaling gradients
                           w.r.t. input species by 1/R. Useful for exp-scale data.
                           (Only affects CRNModel, ignored for MLPModel)
        """
        if optimizer_type not in ('adam', 'sgd', 'fletcher_reeves'):
            raise ValueError(f"Unknown optimizer_type: {optimizer_type}. "
                           f"Must be 'adam', 'sgd', or 'fletcher_reeves'")
        
        self.model = model
        self.mi_estimator = mi_estimator
        self.layer_indices = layer_indices
        self.optimizer_type = optimizer_type
        self.lr = lr
        self.lr_dict = lr_dict or {}
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.max_grad_norm = max_grad_norm
        self.frozen_params = set(frozen_params) if frozen_params else set()
        self.log_input_space = log_input_space

        # Channel type determines what "code" we extract from the forward pass.
        # 'bernoulli': binary code z ~ Bernoulli(h), h = activation fraction / hidden.
        # 'lognormal': continuous code C ~ LogNormal around mass-action mean Cbar.
        self.channel = getattr(mi_estimator, 'channel_type', 'bernoulli')
        
        # Initialize optimizer state
        self.m = {}  # First moment (Adam) or previous gradient (Fletcher-Reeves)
        self.v = {}  # Second moment (Adam only)
        self.d = {}  # Search direction (Fletcher-Reeves only)
        self.t = 0   # Timestep
        
        for name, shape in model.get_param_shapes().items():
            self.m[name] = np.zeros(shape)
            self.v[name] = np.zeros(shape)
            self.d[name] = np.zeros(shape)
    
    def extract_code_from_indices(self, inputs, indices: List[int]) -> np.ndarray:
        """
        Extract code from arbitrary layer specified by indices.
        
        Handles both Bernoulli channel (activation fractions) and log-normal channel
        (active-form concentrations), dispatching on self.channel.
        
        For CRNModel with biochemical readout:
            - Bernoulli: activation fractions of substrate groups at indices
            - Log-normal: active-form concentrations of substrate groups at indices
        
        For CRNModel with linear readout:
            - Bernoulli: hidden activations clipped to [0,1]
            - Log-normal: hidden concentrations (raw)
        
        For MLPModel:
            - Bernoulli: penultimate layer under sigmoid
            - Log-normal: not supported
        
        Args:
            inputs: Input values for the model
            indices: List of species indices to extract code from
        
        Returns:
            code: shape (code_dim,) where code_dim = # substrate groups at indices
        """
        # Forward pass
        _ = self.model.forward(inputs)
        
        # MLP model: use penultimate layer
        if isinstance(self.model, MLPModel):
            if self.channel == 'lognormal':
                raise NotImplementedError(
                    "Log-normal channel is only supported for CRNModel, not MLPModel."
                )
            return self.model.get_penultimate_activations()
        
        # CRN model
        if not hasattr(self.model, 'readout_type'):
            raise ValueError("Model must be CRNModel or MLPModel")
        
        state = self.model.get_regularization_state()
        
        if self.model.readout_type == 'biochemical':
            from CRNs.regularizers import (
                _group_substrate_forms,
                _active_form_index,
                _activation_fraction_for_group,
                _bound_mass_by_base_id,
            )
            
            substrate_groups = _group_substrate_forms(state['species_names'], indices)
            if len(substrate_groups) == 0:
                raise RuntimeError(f"No substrate groups found at indices {indices}")
            
            C_full = state['C_full']
            bound = _bound_mass_by_base_id(C_full, state['species_names'])
            code = []
            
            if self.channel == 'lognormal':
                # Active-form concentrations
                for _, forms in sorted(substrate_groups.items()):
                    active_idx = _active_form_index(list(forms))
                    code.append(C_full[active_idx])
            else:
                # Activation fractions (Bernoulli)
                for base_id, forms in sorted(substrate_groups.items()):
                    result = _activation_fraction_for_group(
                        C_full, forms, bound_mass=bound.get(base_id, 0.0)
                    )
                    if result is not None:
                        activation, _, _, _ = result
                        code.append(activation)
                    else:
                        code.append(0.0)
            
            return np.array(code, dtype=float)
        
        elif self.model.readout_type == 'linear':
            # Linear readout uses the hidden vector h
            h = self.model._h.copy()
            if self.channel == 'bernoulli':
                h = np.clip(h, 0.0, 1.0)
            return h
        
        else:
            raise ValueError(f"Unknown readout_type: {self.model.readout_type}")

    def _extract_code(self, inputs) -> np.ndarray:
        """Extract the per-sample code from the configured layer."""
        return self.extract_code_from_indices(inputs, self.layer_indices)
    
    def train_step_batch(self, 
                        R_batch: List,
                        compute_diagnostics: bool = False) -> Tuple[float, Dict]:
        """
        Single training step: maximize I(R; z) over a batch.
        
        Unlike classification, MI requires joint (R, h) samples, so we:
        1. Collect h samples from all R in batch
        2. Estimate I(R; z) over the batch
        3. Compute gradients via chain rule
        4. Update parameters (gradient ascent to maximize)
        
        Args:
            R_batch: List of input samples
            compute_diagnostics: Whether to compute detailed diagnostics
        
        Returns:
            mi_value: Estimated I(R; z)
            diagnostics: Dictionary with grad norms, h statistics, etc.
        """
        batch_size = len(R_batch)
        
        # Step 1: Collect codes (deterministic given R).
        # Bernoulli channel: h = activation fractions in [0,1].
        # Log-normal channel: Cbar = mean downstream concentrations.
        h_batch = []
        for R in R_batch:
            h = self._extract_code(R)
            h_batch.append(h)
        h_batch = np.array(h_batch)  # (batch_size, code_dim)
        
        # Step 2: Estimate I(R; code)
        mi_value = self.mi_estimator.estimate(h_batch)
        
        # Step 3: Compute ∇_code I(R; code) for each sample
        # (Bernoulli: ∇_h I; log-normal: ∇_Cbar I)
        grad_h_batch = self.mi_estimator.compute_gradients(h_batch)  # (batch_size, code_dim)
        
        # Step 4: Chain rule through CRN to get ∇_θ I
        # For each sample: ∇_θ I = ∇_h I · ∂h/∂C · ∂C/∂θ
        grad_params = self._accumulate_parameter_gradients(R_batch, grad_h_batch)
        
        # Step 5: Apply optimizer update (gradient ASCENT to maximize MI)
        grad_norms = self._update_parameters(grad_params, maximize=True)
        
        # Compute IPR (sparsity measure) for the batch
        h_batch_iprs = [compute_fractional_ipr(h) for h in h_batch]
        mean_ipr = np.mean(h_batch_iprs)
        
        # Diagnostics
        diagnostics = {
            'grad_norms': grad_norms,
            'h_mean': h_batch.mean(axis=0),
            'h_std': h_batch.std(axis=0),
            'mi_value': mi_value,
            'h_ipr': mean_ipr
        }
        
        if compute_diagnostics:
            diagnostics.update({
                'h_min': h_batch.min(axis=0),
                'h_max': h_batch.max(axis=0),
                'h_samples': h_batch.copy(),
                'h_iprs': h_batch_iprs
            })
        
        return mi_value, diagnostics
    
    def _accumulate_parameter_gradients(self, 
                                       R_batch: List,
                                       grad_h_batch: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Chain rule: ∇_θ I = Σ_n ∇_{h_n} I · ∂h_n/∂θ
        
        For CRNModel: ∇_θ I = ∇_h I · ∂h/∂C · ∂C/∂θ (sensitivity matrices)
        For MLPModel: ∇_θ I = ∇_h I · ∂h/∂z · backprop(z) (standard backprop)
        """
        # Initialize gradient accumulation
        grad_accumulator = {}
        for name in self.model.get_param_shapes().keys():
            if name not in self.frozen_params:
                grad_accumulator[name] = None
        
        # Check model type
        is_mlp = isinstance(self.model, MLPModel)
        
        # Accumulate gradients over batch
        for R, grad_h in zip(R_batch, grad_h_batch):
            # Forward pass to set model state
            _ = self.model.forward(R)
            
            if is_mlp:
                # MLP: backprop through sigmoid and network layers
                grad_params_flat = self._backprop_mlp(grad_h)
                
                # Accumulate
                if grad_accumulator['params'] is None:
                    grad_accumulator['params'] = grad_params_flat
                else:
                    grad_accumulator['params'] += grad_params_flat
            
            else:
                # CRN: use sensitivity matrices
                # Get sensitivity matrices ∂C/∂θ
                dC_dk_full, dC_dl_full = self.model._compute_sensitivity_matrices()
                
                # Get ∂code/∂C (depends on channel + readout type)
                dh_dC = self._compute_code_jacobian()
                
                # Chain rule: ∇_C I = ∇_code I · ∂code/∂C
                grad_C = grad_h @ dh_dC  # (n_species,)
                
                # Apply log-input-space scaling if requested
                if self.log_input_space:
                    grad_C = self._apply_log_input_scaling(grad_C, R)
                
                # ∇_θ I = ∇_C I · ∂C/∂θ
                if 'log_rates' not in self.frozen_params:
                    grad_k = grad_C @ dC_dk_full  # (n_rates,)
                    if grad_accumulator['log_rates'] is None:
                        grad_accumulator['log_rates'] = grad_k
                    else:
                        grad_accumulator['log_rates'] += grad_k
                
                if 'log_l0' not in self.frozen_params:
                    grad_l = grad_C @ dC_dl_full  # (n_l0,)
                    if grad_accumulator['log_l0'] is None:
                        grad_accumulator['log_l0'] = grad_l
                    else:
                        grad_accumulator['log_l0'] += grad_l
        
        # Average over batch
        batch_size = len(R_batch)
        for name in grad_accumulator:
            if grad_accumulator[name] is not None:
                grad_accumulator[name] /= batch_size
        
        return grad_accumulator
    
    def _backprop_mlp(self, grad_h: np.ndarray) -> np.ndarray:
        """
        Backpropagate ∇_h I through MLP to get ∇_θ I.
        
        Chain rule: ∇_θ I = ∇_h I · ∂h/∂z · ∂z/∂θ
        where z is the pre-activation of the penultimate layer,
        h = sigmoid(z), and ∂z/∂θ is computed via standard backprop.
        
        The gradient flows through the penultimate and all earlier layers,
        but NOT through the final output layer.
        
        Args:
            grad_h: Gradient w.r.t. h, shape (code_dim,)
        
        Returns:
            grad_params: Flat gradient vector for all MLP parameters
        """
        # Penultimate layer index (0-indexed layer number)
        penultimate_idx = self.model.mlp.n_layers - 2
        
        # Get h (penultimate activations under sigmoid)
        penultimate_pre = self.model.mlp.pre_activations[penultimate_idx]
        h = 1.0 / (1.0 + np.exp(-np.clip(penultimate_pre, -500, 500)))
        
        # Sigmoid derivative: dh/dz = h * (1 - h)
        dh_dz = h * (1.0 - h)
        
        # Gradient w.r.t. pre-activation: ∇_z I = ∇_h I · ∂h/∂z
        delta = grad_h * dh_dz
        
        # Initialize gradient storage
        grad_weights = [np.zeros_like(W) for W in self.model.mlp.weights]
        grad_biases = [np.zeros_like(b) for b in self.model.mlp.biases]
        
        # Backprop from penultimate layer down to input
        for i in range(penultimate_idx, -1, -1):
            # Gradient w.r.t. weights and biases of layer i
            a_prev = self.model.mlp.activations[i]
            grad_weights[i] = np.outer(a_prev, delta)
            grad_biases[i] = delta.copy()
            
            # Backprop to previous layer (if not at input)
            if i > 0:
                # Gradient w.r.t. input to layer i
                delta = self.model.mlp.weights[i] @ delta
                # Backprop through activation function of layer i-1
                delta = delta * self.model.mlp._activation_derivative(self.model.mlp.pre_activations[i - 1])
        
        # Output layer (final layer) gets zero gradient (not used for MI)
        # (already zero from initialization)
        
        # Flatten to match parameter structure
        grad_list = []
        for dW, db in zip(grad_weights, grad_biases):
            grad_list.append(dW.flatten())
            grad_list.append(db.flatten())
        
        return np.concatenate(grad_list)
    
    def _compute_code_jacobian(self) -> np.ndarray:
        """
        Compute ∂code/∂C for the configured layer.

        Bernoulli channel: ∂h/∂C (activation-fraction / hidden jacobian).
        Log-normal channel: ∂Cbar/∂C (selection of active-form concentrations).

        Returns:
            jacobian: shape (code_dim, n_species)
        """
        return self._compute_code_jacobian_from_indices(self.layer_indices)

    def _compute_code_jacobian_from_indices(self, indices: List[int]) -> np.ndarray:
        """
        Compute ∂code/∂C for arbitrary layer specified by indices.
        
        Dispatches on channel type:
        - Bernoulli channel: ∂(activation_fraction)/∂C
        - Log-normal channel: ∂(active-form concentration)/∂C (selection matrix)
        
        Args:
            indices: List of species indices defining the layer
        
        Returns:
            jacobian: shape (code_dim, n_species)
        """
        state = self.model.get_regularization_state()
        n_species = len(state['C_full'])
        
        if self.model.readout_type == 'biochemical':
            from CRNs.regularizers import _group_substrate_forms, _active_form_index
            
            substrate_groups = _group_substrate_forms(state['species_names'], indices)
            code_dim = len(substrate_groups)
            jacobian = np.zeros((code_dim, n_species))
            
            if self.channel == 'lognormal':
                # Log-normal: selection matrix (1 at active-form index)
                for group_idx, (_, forms) in enumerate(sorted(substrate_groups.items())):
                    active_idx = _active_form_index(list(forms))
                    jacobian[group_idx, active_idx] = 1.0
            else:
                # Bernoulli: activation fraction derivatives
                jacobian = self._jacobian_activation_fractions_from_indices(
                    state, n_species, indices
                )
            
            return jacobian
        
        elif self.model.readout_type == 'linear':
            # Linear readout: h = extract_hidden_vector(C, groups)
            return self._jacobian_linear_readout(state, n_species)
        
        else:
            raise ValueError(f"Unknown readout_type: {self.model.readout_type}")
    
    def _jacobian_activation_fractions_from_indices(
        self, state: Dict, n_species: int, indices: List[int]
    ) -> np.ndarray:
        """
        ∂(activation_fraction)/∂C for biochemical readout at specified indices.
        
        For substrate family {S_i, S_i^*, ...}:
        activation = C_active / C_total
        
        C_total includes free phosphoforms plus bound complexes (names with
        ``_``). Bound species are inactive, so
        ∂activation/∂C_complex = -C_active/C_total^2 (times token count).
        
        ∂activation/∂C_active = 1/C_total - C_active/C_total^2
        ∂activation/∂C_inactive = -C_active/C_total^2
        
        Args:
            state: Model regularization state
            n_species: Total number of species
            indices: Species indices to compute activations for
        
        Returns:
            jacobian: shape (code_dim, n_species)
        """
        from CRNs.regularizers import (
            _group_substrate_forms,
            _activation_fraction_for_group,
            _complex_indices_by_base_id,
            _bound_mass_by_base_id,
        )
        
        species_names = state['species_names']
        C_full = state['C_full']
        contribs = _complex_indices_by_base_id(species_names)
        bound = _bound_mass_by_base_id(C_full, species_names, contribs)
        
        # Group substrates by base ID
        substrate_groups = _group_substrate_forms(species_names, indices)
        
        # Initialize Jacobian
        code_dim = len(substrate_groups)
        jacobian = np.zeros((code_dim, n_species))
        
        # Compute Jacobian for each substrate group
        for group_idx, (base_id, forms) in enumerate(sorted(substrate_groups.items())):
            result = _activation_fraction_for_group(
                C_full, forms, bound_mass=bound.get(base_id, 0.0)
            )
            if result is None:
                continue
            
            activation, active_idx, C_total, sorted_forms = result
            
            if C_total < 1e-10:
                continue
            
            # ∂(activation)/∂C
            for _, idx, _ in sorted_forms:
                if idx == active_idx:
                    # ∂a/∂C_active = 1/C_total - C_active/C_total^2
                    jacobian[group_idx, idx] = 1.0 / C_total - C_full[active_idx] / (C_total ** 2)
                else:
                    # ∂a/∂C_inactive = -C_active/C_total^2
                    jacobian[group_idx, idx] = -C_full[active_idx] / (C_total ** 2)
            da_dC_bound = -C_full[active_idx] / (C_total ** 2)
            for idx in contribs.get(base_id, []):
                jacobian[group_idx, idx] += da_dC_bound
        
        return jacobian
    
    def _apply_log_input_scaling(self, grad_C: np.ndarray, R: np.ndarray) -> np.ndarray:
        """
        Scale gradients to optimize I(ln(R); z) instead of I(R; z).
        
        Applies chain rule: ∂/∂θ [f(ln(R))] = (1/R) · ∂/∂θ [f(R)]
        
        Only scales gradients for input species (receptors), leaves others unchanged.
        This effectively optimizes in log-input space, which is more stable for
        exp-scale data with large dynamic range.
        
        Args:
            grad_C: Gradient w.r.t. all species concentrations (n_species,)
            R: Input concentrations (n_inputs,)
        
        Returns:
            grad_C_scaled: Scaled gradient (n_species,)
        """
        grad_C_scaled = grad_C.copy()
        
        # Get number of input species
        n_inputs = len(R)
        R = np.asarray(R)
        
        # Scale gradients for input species by 1/R
        # Add small epsilon to avoid division by zero
        eps = 1e-10
        scaling = 1.0 / (R + eps)
        
        # Input species are the first n_inputs entries in C
        grad_C_scaled[:n_inputs] = grad_C[:n_inputs] * scaling
        
        return grad_C_scaled
    
    def _jacobian_linear_readout(self, state: Dict, n_species: int) -> np.ndarray:
        """
        ∂h/∂C for linear readout.
        
        h = extract_hidden_vector(C, groups) just indexes C,
        so Jacobian is identity at those positions.
        """
        from CRNs.training import extract_hidden_vector
        
        hidden_groups = self.model._hidden_groups
        code_dim = len(hidden_groups)
        jacobian = np.zeros((code_dim, n_species))
        
        # Each h_i corresponds to one C component
        for i, group in enumerate(hidden_groups):
            # group is dict: {'base_id': int, 'active_idx': int, 'forms': [...]}
            active_idx = group['active_idx']
            jacobian[i, active_idx] = 1.0
        
        return jacobian
    
    def _update_parameters(self, grads: Dict[str, np.ndarray], maximize: bool = True) -> Dict[str, float]:
        """
        Update parameters using optimizer (Adam, SGD, or Fletcher-Reeves).
        
        Args:
            grads: Gradient dict
            maximize: If True, perform gradient ascent (for MI maximization)
        
        Returns:
            grad_norms: Dict of gradient norms per parameter
        """
        self.t += 1
        grad_norms = {}
        
        # Get current parameters
        params = self.model.get_params()
        
        for name in grads:
            if grads[name] is None or name in self.frozen_params:
                continue
            
            grad = grads[name]
            
            # Flip sign if maximizing
            if maximize:
                grad = -grad  # Gradient ascent = negative gradient descent
            
            # Gradient clipping
            grad_norm = np.linalg.norm(grad)
            if grad_norm > self.max_grad_norm:
                grad = grad * (self.max_grad_norm / grad_norm)
                grad_norms[name] = self.max_grad_norm
            else:
                grad_norms[name] = grad_norm
            
            # Get learning rate
            param_lr = self.lr_dict.get(name, self.lr)
            
            if self.optimizer_type == 'adam':
                # Adam update
                self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
                self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad ** 2)
                
                # Bias correction
                m_hat = self.m[name] / (1 - self.beta1 ** self.t)
                v_hat = self.v[name] / (1 - self.beta2 ** self.t)
                
                # Update
                update = param_lr * m_hat / (np.sqrt(v_hat) + self.eps)
            
            elif self.optimizer_type == 'sgd':
                # Simple SGD
                update = param_lr * grad
            
            elif self.optimizer_type == 'fletcher_reeves':
                # Fletcher-Reeves conjugate gradient
                # β_k = ||∇f_k||² / ||∇f_{k-1}||²
                
                if self.t == 1:
                    # First iteration: use steepest descent
                    beta = 0.0
                else:
                    # Compute Fletcher-Reeves coefficient
                    grad_norm_sq = np.sum(grad ** 2)
                    prev_grad_norm_sq = np.sum(self.m[name] ** 2)
                    
                    if prev_grad_norm_sq > 1e-10:
                        beta = grad_norm_sq / prev_grad_norm_sq
                    else:
                        beta = 0.0
                    
                    # Restart if beta is too large (loss of conjugacy)
                    if beta > 10.0:
                        beta = 0.0
                
                # Update search direction: d_k = -∇f_k + β_k * d_{k-1}
                # (negative because we're doing descent after sign flip)
                self.d[name] = -grad + beta * self.d[name]
                
                # Store current gradient for next iteration
                self.m[name] = grad.copy()
                
                # Update with search direction
                update = -param_lr * self.d[name]  # negative to undo the earlier sign flip
            
            else:
                raise ValueError(f"Unknown optimizer: {self.optimizer_type}")
            
            # Apply update
            params[name] = params[name] - update
        
        # Set updated parameters
        self.model.set_params(params)
        
        return grad_norms


def run_training_mi(
    trainer: MITrainer,
    input_data: InputData,
    num_batches: int,
    batch_size: int,
    print_every: int = 50,
    history: Optional[MITrainingHistory] = None,
    verbose: bool = True
) -> MITrainingHistory:
    """
    Training loop for mutual information maximization.
    
    Similar structure to run_training_crn but optimizes I(R; z) instead
    of classification accuracy.
    
    Supports three optimizers:
    - 'adam': Adaptive learning rates with momentum (default, most stable)
    - 'sgd': Vanilla stochastic gradient descent
    - 'fletcher_reeves': Nonlinear conjugate gradient (can be faster, 
                        but more sensitive to learning rate)
    
    Args:
        trainer: MITrainer instance
        input_data: InputData with samples from p(R)
        num_batches: Total number of training batches
        batch_size: Number of samples per batch
        print_every: Print frequency
        history: Optional existing history to continue
        verbose: Whether to print progress
    
    Returns:
        history: MITrainingHistory with training metrics
    """
    if history is None:
        history = MITrainingHistory()
    
    start_time = time.time()
    
    if verbose:
        if isinstance(trainer.model, MLPModel):
            model_type = "MLP-MI"
        elif hasattr(trainer.model, 'forward_method'):
            model_type = f"CRN-MI ({trainer.model.forward_method})"
        else:
            model_type = "CRN-MI"
        print(f"Starting {model_type} training: maximize I(R; z)")
        print(f"Batches: {num_batches}, Batch size: {batch_size}")
        print("=" * 80)
    
    # Get n_classes from input_data
    n_classes = input_data.n_classes
    
    for batch in range(num_batches):
        # Sample R batch from p(R) (uniformly across classes)
        R_batch = []
        for _ in range(batch_size):
            class_idx = random.randrange(n_classes)
            R = input_data.get_next_training_sample(class_idx)
            R_batch.append(R)
        
        try:
            # Training step
            mi_value, diagnostics = trainer.train_step_batch(
                R_batch,
                compute_diagnostics=(batch % print_every == 0)
            )
            
            # Check for numerical issues
            if np.isnan(mi_value) or np.isinf(mi_value):
                if verbose:
                    print(f"  Warning: Invalid MI value at batch {batch}")
                continue
            
            # Record history
            history.record_batch_mi(
                mi_value=mi_value,
                h_mean=diagnostics['h_mean'],
                h_std=diagnostics['h_std'],
                h_ipr=diagnostics['h_ipr']
            )
            history.record_batch(
                avg_loss=0.0,  # No classification loss (use avg_loss not loss)
                accuracy=0.0,  # No classification accuracy
                grad_norms=diagnostics['grad_norms']
            )
            
            # Print progress
            if verbose and (batch % print_every == 0 or batch == num_batches - 1):
                elapsed = time.time() - start_time
                recent_mi = history.get_recent_mi(window=print_every)
                
                grad_norm_str = ", ".join([
                    f"{name}={norm:.2e}"
                    for name, norm in diagnostics['grad_norms'].items()
                ])
                
                print(f"Batch {batch:4d}/{num_batches} | "
                      f"I(R;z)={mi_value:.4f} (avg={recent_mi:.4f}) | "
                      f"h: μ={diagnostics['h_mean'].mean():.3f} σ={diagnostics['h_std'].mean():.3f} IPR={diagnostics['h_ipr']:.3f} | "
                      f"grad: {grad_norm_str} | "
                      f"time={elapsed:.1f}s")
        
        except Exception as e:
            if verbose:
                print(f"  Warning: Training step failed at batch {batch}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if verbose:
        elapsed = time.time() - start_time
        final_mi = history.get_recent_mi(window=50)
        print("=" * 80)
        print(f"Training complete! Final I(R;z) = {final_mi:.4f}")
        print(f"Total time: {elapsed:.1f}s ({elapsed/num_batches:.2f}s/batch)")
    
    return history
