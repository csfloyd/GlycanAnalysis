"""
Regularization module for CRN training.

This module provides a base class and example regularizers for adding
penalties to the training objective function.
"""

import re
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple


# Canonical phosphoforms only: S0, S0s, S3ss. Not complexes like S2_S0.
_SUBSTRATE_NAME_RE = re.compile(r'^S(\d+)(s*)$')


def _parse_substrate_species(name: str) -> Optional[Tuple[int, int]]:
    """
    Parse a substrate species name into (base_id, num_trailing_s).

    Examples: S0 -> (0, 0), S0s -> (0, 1), S3ss -> (3, 2)
    Returns None for non-substrate names (receptors, catalyzed complexes, etc.).
    """
    match = _SUBSTRATE_NAME_RE.match(name)
    if match is None:
        return None
    return int(match.group(1)), len(match.group(2))


def _complex_indices_by_base_id(species_names: List[str]) -> Dict[int, List[int]]:
    """
    Map substrate id -> species indices of bound complexes containing that substrate.

    Complexes are names with ``_`` (e.g. S0_S1s). Each parsed S-token contributes
    one entry, so S0_S0 lists the same index twice. Bound forms are inactive.
    """
    contribs: Dict[int, List[int]] = {}
    for idx, name in enumerate(species_names):
        if '_' not in name:
            continue
        if _parse_substrate_species(name) is not None:
            continue
        for token in name.split('_'):
            parsed = _parse_substrate_species(token)
            if parsed is None:
                continue
            contribs.setdefault(parsed[0], []).append(idx)
    return contribs


def _bound_mass_by_base_id(
    C_full: np.ndarray,
    species_names: List[str],
    contribs: Optional[Dict[int, List[int]]] = None,
) -> Dict[int, float]:
    """Total bound concentration per substrate id (inactive mass)."""
    if contribs is None:
        contribs = _complex_indices_by_base_id(species_names)
    return {
        base_id: float(sum(C_full[i] for i in idxs))
        for base_id, idxs in contribs.items()
    }


def _group_substrate_forms(
    species_names: List[str],
    indices: List[int],
) -> Dict[int, List[Tuple[int, int, str]]]:
    """Group free substrate phosphoforms by base id: {base_id: [(num_s, idx, name), ...]}."""
    substrate_groups: Dict[int, List[Tuple[int, int, str]]] = {}
    for idx in indices:
        parsed = _parse_substrate_species(species_names[idx])
        if parsed is None:
            continue
        base_id, num_s = parsed
        substrate_groups.setdefault(base_id, []).append((num_s, idx, species_names[idx]))
    return substrate_groups


def _active_form_index(forms: List[Tuple[int, int, str]]) -> int:
    """
    Index of the ON/active species in a substrate group.

    Matches network/readout convention: fewest trailing s's is active
    (e.g. S0 active, S0s off; for MP families S active, Ss/Sss/... off).
    Bound complexes are never active.
    """
    forms.sort(key=lambda x: x[0])
    return forms[0][1]


def _activation_fraction_for_group(
    C_full: np.ndarray,
    forms: List[Tuple[int, int, str]],
    bound_mass: float = 0.0,
) -> Optional[Tuple[float, int, float, List[Tuple[int, int, str]]]]:
    """
    Return (activation, active_idx, C_total, sorted_forms) or None if undefined.

    C_total is free phosphoforms plus ``bound_mass`` (dimers/complexes, treated
    as inactive). The numerator is only the free ON form.
    """
    if len(forms) < 2:
        return None
    forms = sorted(forms, key=lambda x: x[0])
    active_idx = forms[0][1]
    C_total = sum(C_full[idx] for _, idx, _ in forms) + float(bound_mass)
    if C_total <= 1e-10:
        return 0.0, active_idx, C_total, forms
    return C_full[active_idx] / C_total, active_idx, C_total, forms


def compute_activation_fractions(
    C_full: np.ndarray,
    species_names: List[str],
    hidden_indices: List[int] = None,
) -> Dict[int, float]:
    """
    Compute activation fractions for substrate groups.

    For each substrate family (e.g. S3 + S3s + S3ss), activation is:
        activation = C_on / C_total

    where the ON form is the free species with the fewest trailing ``s``
    characters (e.g. S3 active, S3s/S3ss off). Bound complexes (names with
    ``_``, e.g. S0_S1s) are inactive but included in C_total.

    Args:
        C_full: Full concentration vector for all species
        species_names: List of species names (e.g., ['R0', 'S0', 'S0s', ...])
        hidden_indices: Optional list of indices to consider (if None, considers all)

    Returns:
        activation_dict: {substrate_id: activation_fraction}
    """
    indices_to_check = hidden_indices if hidden_indices is not None else range(len(species_names))
    substrate_groups = _group_substrate_forms(species_names, list(indices_to_check))
    bound = _bound_mass_by_base_id(C_full, species_names)

    activation_dict = {}
    for base_id, forms in substrate_groups.items():
        result = _activation_fraction_for_group(
            C_full, forms, bound_mass=bound.get(base_id, 0.0)
        )
        if result is not None:
            activation_dict[base_id] = result[0]
    return activation_dict


def compute_code_dim_for_indices(species_names: List[str], indices: List[int]) -> int:
    """
    Compute code dimension (number of substrate groups) for a given set of indices.
    
    This is useful for sizing MI estimators and noise scale vectors.
    
    Args:
        species_names: List of species names
        indices: List of species indices to consider
    
    Returns:
        code_dim: Number of substrate groups at the specified indices
    """
    substrate_groups = _group_substrate_forms(species_names, indices)
    return len(substrate_groups)


def compute_ipr(activations) -> float:
    """
    Inverse Participation Ratio: IPR = (sum a_i)^2 / (sum a_i^2).

    Returns 1.0 for a zero vector. For N nonnegative activations, IPR is in [1, N].
    """
    a = np.asarray(activations, dtype=float)
    sum_a2 = np.sum(a ** 2)
    if sum_a2 < 1e-10:
        return 1.0
    return float((np.sum(a) ** 2) / sum_a2)


def remap_hidden_activations_to_unit_interval(h, activation: str) -> np.ndarray:
    """
    Map post-nonlinearity hidden activations to [0, 1] for IPR visualization/regularization.

    Matches training_signaling_networks.ipynb get_mlp_hidden_activation_vector(..., to_unit_interval=True).
    """
    h = np.asarray(h, dtype=float)
    if activation in ('relu', 'linear'):
        return h / (h + 1.0)
    if activation == 'tanh':
        return 0.5 * (h + 1.0)
    return h  # sigmoid already in [0, 1]


def remap_hidden_activations_jacobian_diag(h, activation: str) -> np.ndarray:
    """Element-wise derivative da/dh for remap_hidden_activations_to_unit_interval."""
    h = np.asarray(h, dtype=float)
    if activation in ('relu', 'linear'):
        return 1.0 / (h + 1.0) ** 2
    if activation == 'tanh':
        return np.full_like(h, 0.5, dtype=float)
    return np.ones_like(h)


def compute_fractional_ipr(activations) -> float:
    """IPR / N on a (typically unit-interval) activation vector; range [1/N, 1] for nonnegative a."""
    a = np.asarray(activations, dtype=float)
    return compute_ipr(a) / a.size


def compute_mlp_hidden_fractional_ipr(h, activation: str) -> float:
    """Fractional IPR on penultimate hidden activations after unit-interval remap."""
    a = remap_hidden_activations_to_unit_interval(h, activation)
    return compute_fractional_ipr(a)


class Regularizer(ABC):
    """
    Base class for regularization terms.
    
    Subclasses should implement:
    - compute_penalty(): compute scalar penalty from model state
    - compute_gradients(): compute gradients w.r.t. parameters
    """
    
    def __init__(self, weight: float = 1.0, name: str = None):
        """
        Args:
            weight: Regularization strength (lambda)
            name: Optional name for this regularizer (for logging)
        """
        self.weight = weight
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    def compute_metric(self, state_dict: Dict) -> float:
        """
        Compute the raw metric value being regularized (before penalty transformation).
        
        Args:
            state_dict: Dictionary containing model state
            
        Returns:
            metric: Raw metric value (e.g., actual IPR, L2 norm, etc.)
        """
        pass
    
    @abstractmethod
    def compute_penalty(self, state_dict: Dict) -> float:
        """
        Compute regularization penalty from model state.
        
        Args:
            state_dict: Dictionary containing:
                - 'C_full': Full concentration vector (all species)
                - 'C_reduced': Reduced concentration vector (remaining species after conservation)
                - 'l0': Conservation constants
                - 'rates': Rate constants
                - 'species_names': List of species names
                - 'hidden_indices': Indices of hidden nodes in C_full
                - 'class_ids': Indices of output nodes
                - 'n_inputs': Number of input nodes
                
        Returns:
            penalty: Scalar regularization penalty (will be multiplied by self.weight)
        """
        pass
    
    @abstractmethod
    def compute_gradients(self, state_dict: Dict, dC_dk_full: np.ndarray, dC_dl_full: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute gradients of regularization w.r.t. parameters.
        
        Uses chain rule: dL_reg/dk = dL_reg/dC * dC/dk
        
        Args:
            state_dict: Same as compute_penalty
            dC_dk_full: Sensitivity matrix dC/dk for ALL species (n_species, n_rates)
            dC_dl_full: Sensitivity matrix dC/dl for ALL species (n_species, n_l0)
            
        Returns:
            grad_dict: Dictionary with keys 'log_rates', 'log_l0' containing gradient arrays
        """
        pass


class IPRSparsityRegularizer(Regularizer):
    """
    IPR (Inverse Participation Ratio) based sparsity regularizer.
    
    Uses standard IPR definition: IPR = (∑ a_i)² / (∑ a_i²)
    
    Properties:
    - IPR = 1: All activation on one node (maximally localized)
    - IPR = N: Uniform across N nodes (maximally delocalized)
    - Interpretation: Effective number of active nodes
    
    Lower IPR = more localized (sparse)
    Higher IPR = more distributed (many nodes active)
    """
    
    def __init__(self, weight: float = 1.0, target_ipr: float = 1.0, penalty_type: str = 'above'):
        """
        Args:
            weight: Regularization strength
            target_ipr: Target IPR value (1 to N_hidden)
                - 1.0 = maximally sparse (encourage one active node)
                - N = maximally distributed (encourage all nodes active)
            penalty_type: 
                - 'above': penalize when IPR > target (encourage sparsity/localization)
                - 'below': penalize when IPR < target (encourage distribution)
                - 'deviation': penalize deviation from target in either direction
        """
        super().__init__(weight, name=f'IPR')
        self.target_ipr = target_ipr
        self.penalty_type = penalty_type
    
    def _compute_ipr(self, activations):
        """Compute standard IPR on activation vector."""
        return compute_ipr(activations)
    
    def compute_metric(self, state_dict: Dict) -> float:
        """Return the actual IPR value."""
        C_full = state_dict['C_full']
        species_names = state_dict['species_names']
        hidden_indices = state_dict['hidden_indices']
        
        # Use utility function to get activation fractions
        activation_dict = compute_activation_fractions(C_full, species_names, hidden_indices)
        
        if len(activation_dict) == 0:
            return 1.0  # No substrates -> treated as maximally localized
        
        # Extract activations in sorted order
        activations = np.array([activation_dict[base_id] for base_id in sorted(activation_dict.keys())])
        
        return self._compute_ipr(activations)
    
    def compute_penalty(self, state_dict: Dict) -> float:
        """Compute IPR-based sparsity penalty."""
        # Get the raw IPR metric
        ipr = self.compute_metric(state_dict)
        
        # Apply penalty transformation based on type
        if self.penalty_type == 'above':
            # Penalize when IPR is above target (too distributed, encourage sparsity)
            penalty = max(0, ipr - self.target_ipr) ** 2
        elif self.penalty_type == 'below':
            # Penalize when IPR is below target (too sparse, encourage distribution)
            penalty = max(0, self.target_ipr - ipr) ** 2
        elif self.penalty_type == 'deviation':
            # Penalize deviation in either direction
            penalty = (ipr - self.target_ipr) ** 2
        else:
            raise ValueError(f"Unknown penalty_type: {self.penalty_type}")
        
        return penalty
    
    def compute_gradients(self, state_dict: Dict, dC_dk_full: np.ndarray, dC_dl_full: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute gradients of IPR penalty w.r.t. parameters using analytical derivatives.
        
        Uses chain rule: dL/dk = dL/dIPR * dIPR/da * da/dC * dC/dk
        where a = activation vector for substrate groups.
        """
        C_full = state_dict['C_full']
        species_names = state_dict['species_names']
        hidden_indices = state_dict['hidden_indices']
        rates = state_dict['rates']
        l0 = state_dict['l0']
        
        # Initialize gradient vector
        dL_dC = np.zeros_like(C_full)
        
        substrate_groups = _group_substrate_forms(species_names, hidden_indices)
        contribs = _complex_indices_by_base_id(species_names)
        bound = _bound_mass_by_base_id(C_full, species_names, contribs)
        
        # Build activation list and group info
        substrate_group_list = []
        activations = []
        
        for base_id, forms in substrate_groups.items():
            result = _activation_fraction_for_group(
                C_full, forms, bound_mass=bound.get(base_id, 0.0)
            )
            if result is None:
                continue
            activation, active_idx, C_total, sorted_forms = result
            substrate_group_list.append(
                (sorted_forms, active_idx, C_total, contribs.get(base_id, []))
            )
            activations.append(activation)
        
        if len(activations) == 0:
            return {
                'log_rates': np.zeros_like(rates),
                'log_l0': np.zeros_like(l0)
            }
        
        activations = np.array(activations)
        
        # Compute current IPR
        ipr = self._compute_ipr(activations)
        
        # Step 1: Compute dL/dIPR based on penalty type
        if self.penalty_type == 'above':
            if ipr > self.target_ipr:
                dL_dipr = 2 * (ipr - self.target_ipr)
            else:
                dL_dipr = 0.0
        elif self.penalty_type == 'below':
            if ipr < self.target_ipr:
                dL_dipr = -2 * (self.target_ipr - ipr)
            else:
                dL_dipr = 0.0
        elif self.penalty_type == 'deviation':
            dL_dipr = 2 * (ipr - self.target_ipr)
        
        if abs(dL_dipr) < 1e-12:
            # No gradient if penalty is zero
            return {
                'log_rates': np.zeros_like(rates),
                'log_l0': np.zeros_like(l0)
            }
        
        # Step 2: Compute IPR intermediate terms
        # IPR = S1^2 / S2 where S1 = sum(a_i), S2 = sum(a_i^2)
        S1 = np.sum(activations)
        S2 = np.sum(activations ** 2)
        
        if S2 < 1e-10:
            # Degenerate case - no gradient
            return {
                'log_rates': np.zeros_like(rates),
                'log_l0': np.zeros_like(l0)
            }
        
        # Step 3: Compute dIPR/da for each activation
        # dIPR/da_i = (2/S2) * (S1 - IPR * a_i)
        dipr_da = (2.0 / S2) * (S1 - ipr * activations)
        
        # Step 4: Compute da/dC for each concentration and apply chain rule
        for group_idx, (forms, active_idx, C_total, complex_idxs) in enumerate(substrate_group_list):
            a_g = activations[group_idx]
            dipr_da_g = dipr_da[group_idx]
            
            # Chain rule component: dL/dIPR * dIPR/da_g
            dL_da_g = dL_dipr * dipr_da_g
            if C_total <= 1e-10:
                continue
            
            # For each concentration in this substrate group
            for num_s, idx, name in forms:
                if idx == active_idx:
                    # Active form: da/dC = (1 - a) / C_total
                    da_dC = (1.0 - a_g) / C_total
                else:
                    # Inactive form: da/dC = -a / C_total
                    da_dC = -a_g / C_total
                
                # Chain rule: dL/dC = dL/da * da/dC
                dL_dC[idx] += dL_da_g * da_dC
            for idx in complex_idxs:
                dL_dC[idx] += dL_da_g * (-a_g / C_total)
        
        # Chain rule: dL/dk = dL/dC * dC/dk
        grad_rates = np.dot(dL_dC, dC_dk_full)
        grad_l0 = np.dot(dL_dC, dC_dl_full)
        
        # Convert to log-space
        grad_log_rates = self.weight * grad_rates * rates
        grad_log_l0 = self.weight * grad_l0 * l0
        
        return {
            'log_rates': grad_log_rates,
            'log_l0': grad_log_l0
        }
        
        # ========== FINITE DIFFERENCE METHOD (LEGACY - KEPT FOR REFERENCE) ==========
        # The analytical method above is preferred for accuracy and efficiency.
        # Uncomment below to use finite differences instead.
        # 
        # # For each substrate group, compute d(IPR)/d(C_i) using finite differences
        # for group_idx, (forms, active_idx, C_total) in enumerate(substrate_group_list):
        #     # Use finite differences to compute dipr/dC for each form in the group
        #     epsilon = 1e-6
        #     
        #     for num_s, idx, name in forms:
        #         # Perturb this concentration
        #         C_full_perturbed = C_full.copy()
        #         C_full_perturbed[idx] += epsilon
        #         
        #         # Recompute activations for all groups
        #         activations_perturbed = activations.copy()
        #         for g_idx, (g_forms, g_active_idx, g_C_total) in enumerate(substrate_group_list):
        #             C_active_new = C_full_perturbed[g_active_idx]
        #             C_total_new = sum(C_full_perturbed[g_idx] for _, g_idx, _ in g_forms)
        #             if C_total_new > 1e-10:
        #                 activations_perturbed[g_idx] = C_active_new / C_total_new
        #         
        #         ipr_perturbed = self._compute_ipr(activations_perturbed)
        #         
        #         # Finite difference
        #         dipr_dC = (ipr_perturbed - ipr) / epsilon
        #         
        #         # Chain rule: dL/dC = dL/dipr * dipr/dC
        #         dL_dC[idx] += dL_dipr * dipr_dC
        # 
        # # Chain rule: dL/dk = dL/dC * dC/dk
        # grad_rates = np.dot(dL_dC, dC_dk_full)
        # grad_l0 = np.dot(dL_dC, dC_dl_full)
        # 
        # # Convert to log-space
        # grad_log_rates = self.weight * grad_rates * rates
        # grad_log_l0 = self.weight * grad_l0 * l0
        # 
        # return {
        #     'log_rates': grad_log_rates,
        #     'log_l0': grad_log_l0
        # }


class BinarizationRegularizer(Regularizer):
    """
    Encourages activations to be near 0 or 1 (binary), avoiding intermediate values.
    
    This is often more effective than IPR for encouraging sparse patterns.
    """
    
    def __init__(self, weight: float = 1.0, sharpness: float = 2.0):
        """
        Args:
            weight: Regularization strength
            sharpness: How strongly to penalize intermediate values (higher = sharper)
        """
        super().__init__(weight, name='Binarization')
        self.sharpness = sharpness
    
    def compute_metric(self, state_dict: Dict) -> float:
        """
        Return average activation value (for monitoring).
        Could also return average distance from binary {0,1}.
        """
        C_full = state_dict['C_full']
        species_names = state_dict['species_names']
        hidden_indices = state_dict['hidden_indices']
        
        # Use utility function to get activation fractions
        activation_dict = compute_activation_fractions(C_full, species_names, hidden_indices)
        
        if len(activation_dict) == 0:
            return 0.0
        
        activations = np.array(list(activation_dict.values()))
        # Return average "intermediate-ness" (0=perfectly binary, 0.5=maximally non-binary)
        return np.mean(4 * activations * (1 - activations))
    
    def compute_penalty(self, state_dict: Dict) -> float:
        """
        Penalize activations near 0.5 (intermediate).
        
        Uses penalty = sum_i (4 * a_i * (1 - a_i))^sharpness
        This is 0 when a_i is 0 or 1, and maximized at a_i = 0.5
        """
        C_full = state_dict['C_full']
        species_names = state_dict['species_names']
        hidden_indices = state_dict['hidden_indices']
        
        penalty = 0.0
        
        substrate_groups = _group_substrate_forms(species_names, hidden_indices)
        bound = _bound_mass_by_base_id(C_full, species_names)
        
        for base_id, forms in substrate_groups.items():
            result = _activation_fraction_for_group(
                C_full, forms, bound_mass=bound.get(base_id, 0.0)
            )
            if result is None:
                continue
            a, _, _, _ = result
            
            # Penalty is maximum at a=0.5, zero at a=0 or a=1
            intermediate_penalty = (4 * a * (1 - a)) ** self.sharpness
            penalty += intermediate_penalty
        
        return penalty
    
    def compute_gradients(self, state_dict: Dict, dC_dk_full: np.ndarray, dC_dl_full: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute gradients for binarization penalty."""
        C_full = state_dict['C_full']
        species_names = state_dict['species_names']
        hidden_indices = state_dict['hidden_indices']
        rates = state_dict['rates']
        l0 = state_dict['l0']
        
        dL_dC = np.zeros_like(C_full)
        
        substrate_groups = _group_substrate_forms(species_names, hidden_indices)
        contribs = _complex_indices_by_base_id(species_names)
        bound = _bound_mass_by_base_id(C_full, species_names, contribs)
        
        for base_id, forms in substrate_groups.items():
            result = _activation_fraction_for_group(
                C_full, forms, bound_mass=bound.get(base_id, 0.0)
            )
            if result is None:
                continue
            a, active_idx, C_total, sorted_forms = result
            C_active = C_full[active_idx]
            if C_total <= 1e-10:
                continue
            
            # d(penalty)/d(a)
            base = 4 * a * (1 - a)
            if base > 1e-10:
                dP_da = self.sharpness * (base ** (self.sharpness - 1)) * 4 * (1 - 2*a)
            else:
                dP_da = 0.0
            
            for num_s, idx, name in sorted_forms:
                if idx == active_idx:
                    da_dC = (C_total - C_active) / (C_total ** 2)
                else:
                    da_dC = -C_active / (C_total ** 2)
                
                dL_dC[idx] += dP_da * da_dC
            for idx in contribs.get(base_id, []):
                dL_dC[idx] += dP_da * (-C_active / (C_total ** 2))
        
        # Chain rule
        grad_rates = np.dot(dL_dC, dC_dk_full)
        grad_l0 = np.dot(dL_dC, dC_dl_full)
        
        grad_log_rates = self.weight * grad_rates * rates
        grad_log_l0 = self.weight * grad_l0 * l0
        
        return {
            'log_rates': grad_log_rates,
            'log_l0': grad_log_l0
        }


class SparsityRegularizer(Regularizer):
    """
    Simple L1/L2 regularizer on activation deviations.
    
    Penalizes deviation from target activation level (default 0.5 for balance).
    Simpler than IPR but less targeted for sparsity.
    """
    
    def __init__(self, weight: float = 1.0, target_sparsity: float = 0.5, norm: str = 'l2'):
        """
        Args:
            weight: Regularization strength
            target_sparsity: Target average activation (0=all inactive, 1=all active, 0.5=balanced)
            norm: Type of penalty - 'l1' or 'l2'
        """
        super().__init__(weight, name=f'Sparsity_{norm}')
        self.target_sparsity = target_sparsity
        self.norm = norm
        
        if norm not in ['l1', 'l2']:
            raise ValueError(f"norm must be 'l1' or 'l2', got '{norm}'")
    
    def compute_metric(self, state_dict: Dict) -> float:
        """Return the average activation value."""
        C_full = state_dict['C_full']
        species_names = state_dict['species_names']
        hidden_indices = state_dict['hidden_indices']
        
        # Use utility function to get activation fractions
        activation_dict = compute_activation_fractions(C_full, species_names, hidden_indices)
        
        if len(activation_dict) == 0:
            return 0.0
        
        return np.mean(list(activation_dict.values()))
    
    def compute_penalty(self, state_dict: Dict) -> float:
        """Compute simple sparsity penalty."""
        C_full = state_dict['C_full']
        species_names = state_dict['species_names']
        hidden_indices = state_dict['hidden_indices']
        
        penalty = 0.0
        
        substrate_groups = _group_substrate_forms(species_names, hidden_indices)
        bound = _bound_mass_by_base_id(C_full, species_names)
        
        for base_id, forms in substrate_groups.items():
            result = _activation_fraction_for_group(
                C_full, forms, bound_mass=bound.get(base_id, 0.0)
            )
            if result is None:
                continue
            activation, _, _, _ = result
            deviation = activation - self.target_sparsity
            
            if self.norm == 'l2':
                penalty += deviation ** 2
            elif self.norm == 'l1':
                penalty += abs(deviation)
        
        return penalty
    
    def compute_gradients(self, state_dict: Dict, dC_dk_full: np.ndarray, dC_dl_full: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute simple sparsity gradients."""
        C_full = state_dict['C_full']
        species_names = state_dict['species_names']
        hidden_indices = state_dict['hidden_indices']
        rates = state_dict['rates']
        l0 = state_dict['l0']
        
        dL_dC = np.zeros_like(C_full)
        
        substrate_groups = _group_substrate_forms(species_names, hidden_indices)
        contribs = _complex_indices_by_base_id(species_names)
        bound = _bound_mass_by_base_id(C_full, species_names, contribs)
        
        for base_id, forms in substrate_groups.items():
            result = _activation_fraction_for_group(
                C_full, forms, bound_mass=bound.get(base_id, 0.0)
            )
            if result is None:
                continue
            activation, active_idx, C_total, sorted_forms = result
            C_active = C_full[active_idx]
            if C_total <= 1e-10:
                continue
            deviation = activation - self.target_sparsity
            
            if self.norm == 'l2':
                dP_da = 2 * deviation
            elif self.norm == 'l1':
                dP_da = np.sign(deviation)
            
            for num_s, idx, name in sorted_forms:
                if idx == active_idx:
                    da_dC = (C_total - C_active) / (C_total ** 2)
                else:
                    da_dC = -C_active / (C_total ** 2)
                
                dL_dC[idx] += dP_da * da_dC
            for idx in contribs.get(base_id, []):
                dL_dC[idx] += dP_da * (-C_active / (C_total ** 2))
        
        grad_rates = np.dot(dL_dC, dC_dk_full)
        grad_l0 = np.dot(dL_dC, dC_dl_full)
        
        grad_log_rates = self.weight * grad_rates * rates
        grad_log_l0 = self.weight * grad_l0 * l0
        
        return {
            'log_rates': grad_log_rates,
            'log_l0': grad_log_l0
        }


class L2RateRegularizer(Regularizer):
    """
    L2 penalty on rate constants (weight decay).
    
    Penalizes large rate constants to prevent overfitting.
    """
    
    def __init__(self, weight: float = 0.01):
        super().__init__(weight, name='L2_Rates')
    
    def compute_metric(self, state_dict: Dict) -> float:
        """Return the sum of squared rates (same as penalty for L2)."""
        rates = state_dict['rates']
        return np.sum(rates ** 2)
    
    def compute_penalty(self, state_dict: Dict) -> float:
        """Compute L2 penalty on rates."""
        return self.compute_metric(state_dict)
    
    def compute_gradients(self, state_dict: Dict, dC_dk_full: np.ndarray, dC_dl_full: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute L2 rate gradients."""
        rates = state_dict['rates']
        l0 = state_dict['l0']
        
        # d(sum(k^2))/dk = 2*k
        # In log-space: d/d(log_k) = k * d/dk = k * 2*k = 2*k^2
        grad_log_rates = self.weight * 2 * (rates ** 2)
        grad_log_l0 = np.zeros_like(l0)
        
        return {
            'log_rates': grad_log_rates,
            'log_l0': grad_log_l0
        }


class L1RateRegularizer(Regularizer):
    """
    L1 penalty on rate constants (promotes sparsity in rates).
    """
    
    def __init__(self, weight: float = 0.01):
        super().__init__(weight, name='L1_Rates')
    
    def compute_metric(self, state_dict: Dict) -> float:
        """Return the L1 norm of rates."""
        rates = state_dict['rates']
        return np.sum(np.abs(rates))
    
    def compute_penalty(self, state_dict: Dict) -> float:
        """Compute L1 penalty on rates."""
        return self.compute_metric(state_dict)
    
    def compute_gradients(self, state_dict: Dict, dC_dk_full: np.ndarray, dC_dl_full: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute L1 rate gradients."""
        rates = state_dict['rates']
        l0 = state_dict['l0']
        
        # d(sum(|k|))/dk = sign(k)
        # In log-space: d/d(log_k) = k * sign(k) = k * sign(k)
        grad_log_rates = self.weight * rates * np.sign(rates)
        grad_log_l0 = np.zeros_like(l0)
        
        return {
            'log_rates': grad_log_rates,
            'log_l0': grad_log_l0
        }


class MLPHiddenIPRRegularizer(Regularizer):
    """
    Fractional IPR regularizer on the penultimate (last hidden) layer of an MLP.

    Metric (matches training_signaling_networks activation plots):
        1. Remap post-nonlinearity hidden activations to [0, 1] (tanh/relu/linear/sigmoid rules)
        2. fractional_ipr = compute_ipr(a) / N

    For N hidden units, fractional_ipr is in [1/N, 1]. target_ipr should use that scale
    (e.g. raw IPR 3 with N=100 → target_ipr=0.03).

    Expects state_dict from MLPModel.get_regularization_state() with:
        - 'hidden_activations': post-nonlinearity vector from the last hidden layer
        - 'penultimate_index': index into mlp.activations for that layer
        - 'mlp': SimpleMLP instance (must have completed forward())
    """

    def __init__(self, weight: float = 1.0, target_ipr: float = 1.0, penalty_type: str = 'above'):
        super().__init__(weight, name='MLP_IPR')
        self.target_ipr = target_ipr
        self.penalty_type = penalty_type

    def _hidden_activations(self, state_dict: Dict) -> np.ndarray:
        return np.asarray(state_dict['hidden_activations'], dtype=float)

    def _unit_interval_activations(self, state_dict: Dict) -> np.ndarray:
        mlp = state_dict['mlp']
        return remap_hidden_activations_to_unit_interval(
            self._hidden_activations(state_dict),
            mlp.activation,
        )

    def compute_metric(self, state_dict: Dict) -> float:
        return compute_fractional_ipr(self._unit_interval_activations(state_dict))

    def compute_penalty(self, state_dict: Dict) -> float:
        ipr = self.compute_metric(state_dict)
        if self.penalty_type == 'above':
            return max(0.0, ipr - self.target_ipr) ** 2
        if self.penalty_type == 'below':
            return max(0.0, self.target_ipr - ipr) ** 2
        if self.penalty_type == 'deviation':
            return (ipr - self.target_ipr) ** 2
        raise ValueError(f"Unknown penalty_type: {self.penalty_type}")

    def _dL_dipr(self, ipr: float) -> float:
        if self.penalty_type == 'above':
            return 2.0 * (ipr - self.target_ipr) if ipr > self.target_ipr else 0.0
        if self.penalty_type == 'below':
            return -2.0 * (self.target_ipr - ipr) if ipr < self.target_ipr else 0.0
        if self.penalty_type == 'deviation':
            return 2.0 * (ipr - self.target_ipr)
        raise ValueError(f"Unknown penalty_type: {self.penalty_type}")

    def compute_gradients(self, state_dict: Dict, dC_dk_full: np.ndarray, dC_dl_full: np.ndarray) -> Dict[str, np.ndarray]:
        raise NotImplementedError("MLPHiddenIPRRegularizer applies to MLPModel only")

    def compute_gradients_mlp(self, state_dict: Dict) -> Dict[str, np.ndarray]:
        """Analytical fractional-IPR gradient backpropagated through hidden layers (not readout)."""
        mlp = state_dict['mlp']
        activation_index = state_dict['penultimate_index']
        h = self._hidden_activations(state_dict)
        a = remap_hidden_activations_to_unit_interval(h, mlp.activation)
        n = a.size
        ipr_full = compute_ipr(a)
        frac_ipr = ipr_full / n

        dL_dfrac = self._dL_dipr(frac_ipr)
        if abs(dL_dfrac) < 1e-12:
            return {'params': np.zeros(mlp.get_param_count(), dtype=float)}

        S1 = np.sum(a)
        S2 = np.sum(a ** 2)
        if S2 < 1e-10:
            return {'params': np.zeros(mlp.get_param_count(), dtype=float)}

        diprfull_da = (2.0 / S2) * (S1 - ipr_full * a)
        dfrac_da = diprfull_da / n
        dL_da = self.weight * dL_dfrac * dfrac_da
        dL_dh = dL_da * remap_hidden_activations_jacobian_diag(h, mlp.activation)

        grad_weights, grad_biases = mlp.backprop_from_post_activation_grad(activation_index, dL_dh)
        grads = []
        for dW, db in zip(grad_weights, grad_biases):
            grads.append(dW.flatten())
            grads.append(db.flatten())
        return {'params': np.concatenate(grads)}

