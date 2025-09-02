"""
Simulator module for reaction networks.

This module contains the ReactionNetworkSimulator class and all simulation-related methods.
"""

import numpy as np
import sympy
from scipy.integrate import solve_ivp
from typing import List, Tuple, Dict



class ReactionNetworkSimulator:
    """
    Simulator for reaction networks.
    
    This class provides methods for simulating reaction networks, including
    ODE integration, conservation law handling, and sensitivity analysis.
    """
    
    def __init__(self, reaction_network):
        """
        Args:
            reaction_network: An instance of ReactionNetwork
        """
        self.r_n = reaction_network
        self.rhs = self._make_ode_rhs()
        self.reduced_ode_rhs = None
        self.remaining_species = None
        self.const_syms = None
        self.elimination_solutions = None
    
    def _make_ode_rhs(self):
        def rhs(C):
            v_monomials = self.r_n.Psi(C)
            dCdt = np.zeros(self.r_n.Y.shape[0])
            for idx, (i, j, k) in enumerate(self.r_n.reactions):
                rate = k * v_monomials[idx]
                stoichiometric_change = self.r_n.Y[:, j] - self.r_n.Y[:, i]
                dCdt += rate * stoichiometric_change
            return dCdt
        return rhs
    

    
    def integrate(
        self,
        ode_rhs,
        initial_conditions: np.ndarray,
        t_span: Tuple[float, float] = (0, 5),
        num_points: int = 20000,
        **kwargs
        ) -> Tuple[object, np.ndarray]:
        """
        Integrate an ODE system using the provided right-hand side function.

        Args:
            ode_rhs: Callable (t, y) -> dy/dt (the ODE right-hand side)
            initial_conditions: Initial concentrations (array)
            t_span: Tuple (start, end) for integration time
            t_eval: Array of time points to evaluate at (optional)
            kwargs: Additional arguments to pass to solve_ivp

        Returns:
            sol: The solution object from solve_ivp
            final_concentrations: Concentrations at the final time point
        """
        t_eval = np.linspace(t_span[0], t_span[1], num_points)
        sol = solve_ivp(lambda t, C: ode_rhs(C), t_span, initial_conditions, t_eval=t_eval, **kwargs)
        return sol, sol.y[:,-1]
    
    def solve_conservation_laws(self):
        """
        Symbolically solves the conservation laws for the eliminated variables.
        """
        _, all_species_syms, _ = self.get_symbolic_rhs()  # Use sympy symbols, not string names
        species_names = [str(s) for s in all_species_syms]
        species_map = {str(s): s for s in all_species_syms}
        const_syms = [sympy.symbols(f'const{i}') for i in range(len(self.r_n.L))]
        conservation_eqns = []
        eliminated_species = []
        for i, conservation_vector in enumerate(self.r_n.L):
            nonzero_indices = np.nonzero(conservation_vector)[0]
            species_indices = [(idx, str(all_species_syms[idx])) for idx in nonzero_indices]
            species_indices.sort(key=lambda x: x[1])
            if not species_indices:
                continue
            elim_idx, elim_species = next((idx, species) for idx, species in species_indices if species not in eliminated_species)
            eliminated_species.append(elim_species)
            # Use sympy sum for symbolic computation
            eqn = sum(conservation_vector[j] * all_species_syms[j] for j in range(len(all_species_syms))) - const_syms[i]
            conservation_eqns.append(eqn)
        elim_syms = [species_map[s] for s in eliminated_species]
        solutions_list = sympy.solve(conservation_eqns, elim_syms, dict=True)
        if not solutions_list:
            raise RuntimeError("Could not solve conservation laws for eliminated species.")
        solutions = solutions_list[0]  # dict: sympy.Symbol -> expression
        elimination_solutions = {str(k): v for k, v in solutions.items()}
        remaining_species = [s for s in species_names if s not in eliminated_species]
        self.eliminated_species = eliminated_species
        self.remaining_species = remaining_species
        self.const_syms = const_syms
        self.elimination_solutions = elimination_solutions
        
        # Pre-compile elimination functions for fast recovery
        self._compile_elimination_functions()
        self._precompute_species_indices()
    
    def _compile_elimination_functions(self):
        """Pre-compile elimination functions to avoid symbolic evaluation during recovery."""
        self._elimination_funcs = {}
        
        # Resolve all dependencies once
        resolved_solutions = {}
        for elim_name, expr in self.elimination_solutions.items():
            resolved_expr = expr
            # Substitute other eliminated species if present
            for other_name, other_expr in self.elimination_solutions.items():
                if other_name != elim_name:
                    resolved_expr = resolved_expr.subs(other_name, other_expr)
            resolved_solutions[elim_name] = resolved_expr
        
        # Create numerical functions for each eliminated species
        for elim_name, resolved_expr in resolved_solutions.items():
            # Create function that takes remaining species and conservation constants
            func_args = self.remaining_species + self.const_syms
            self._elimination_funcs[elim_name] = sympy.lambdify(
                func_args, 
                resolved_expr, 
                modules=['numpy']
            )
    
    def _precompute_species_indices(self):
        """Pre-compute species indices to avoid repeated lookups."""
        self._species_to_idx = {species: idx for idx, species in enumerate(self.r_n.species_names)}
        self._remaining_indices = [self._species_to_idx[species] for species in self.remaining_species]
        self._eliminated_indices = [self._species_to_idx[species] for species in self.elimination_solutions.keys()]
    
    def make_reduced_rhs_with_conservation(self, const_vals=None, rate_vals=None):
        """
        Constructs a reduced ODE rhs function with conservation laws substituted in.
        Args:
            const_vals: values for the conservation constants (list or array) - optional for symbolic setup
            rate_vals: values for the rate constants (optional, defaults to r_n.rate_constants or [r[2] for r in r_n.reactions])
        Returns:
            ode_rhs(C): function of concentrations for the remaining species
        """
        
        # Only do symbolic substitution once if not already done
        if not hasattr(self, '_substituted_rhs_func'):
            dCdt_sym, all_species_syms, rate_syms = self.get_symbolic_rhs()
            species_names = [str(s) for s in all_species_syms]
            species_map = {str(s): s for s in all_species_syms}
            remaining_syms = [species_map[s] for s in self.remaining_species]
            
            # Substitute solutions into the ODEs
            solutions = {species_map[k]: v for k, v in self.elimination_solutions.items()}
            substituted_rhs = [expr.subs(solutions, simultaneous=True) for expr in dCdt_sym]
            rhs_kept = [substituted_rhs[species_names.index(s)] for s in self.remaining_species]
            
            # Create symbolic function with all parameters
            func_args = remaining_syms + self.const_syms + list(rate_syms)
            self._substituted_rhs_func = sympy.lambdify(func_args, rhs_kept, modules=['numpy'])
            self._rate_syms = rate_syms
        
        # Get rate values (default if not provided)
        if rate_vals is None:
            rate_vals = [r[2] for r in self.r_n.reactions]
        
        # Create the numerical function with current const_vals
        def reduced_ode_rhs(C, const_vals=const_vals):
            args = list(C) + list(const_vals) + list(rate_vals)
            return np.array(self._substituted_rhs_func(*args)).flatten()
        
        # Store for later use
        self.reduced_ode_rhs = reduced_ode_rhs
        return reduced_ode_rhs
    
    def make_reduced_rhs_with_conservation_flexible(self, rate_vals=None):
        """
        Creates a flexible reduced ODE rhs function that accepts const_vals as a parameter.
        This allows changing const_vals without recreating the function.
        
        Args:
            rate_vals: values for the rate constants (optional)
        Returns:
            ode_rhs(C, const_vals): function that takes both concentrations and const_vals
        """
        # Ensure symbolic substitution is done
        if not hasattr(self, '_substituted_rhs_func'):
            self.make_reduced_rhs_with_conservation()
        
        # Get rate values (default if not provided)
        if rate_vals is None:
            rate_vals = [r[2] for r in self.r_n.reactions]
        
        # Create flexible function that accepts const_vals as parameter
        def flexible_reduced_ode_rhs(C, const_vals):
            args = list(C) + list(const_vals) + list(rate_vals)
            return np.array(self._substituted_rhs_func(*args)).flatten()
        
        return flexible_reduced_ode_rhs
    
    def get_const_and_reduced_init(self, C):

        # Compute the values of the conservation constants from the initial conditions
        const_vals = np.dot(self.r_n.L, C)

        # Get the indices of the remaining species
        idx_remain = [self.r_n.species_names.index(s) for s in self.remaining_species]
        reduced_init = C[idx_remain]

        return const_vals, reduced_init
    
    def recover_eliminated_species(self, const_vals, C_reduced):
        """
        Fast recovery of eliminated species using pre-compiled functions.
        """
        # Use pre-compiled functions for fast evaluation
        recovered = {}
        args = list(C_reduced) + list(const_vals)
        
        for elim_name, func in self._elimination_funcs.items():
            try:
                recovered[elim_name] = float(func(*args))
            except (TypeError, ValueError):
                # Fallback to original method if pre-compiled function fails
                recovered[elim_name] = self._recover_species_fallback(elim_name, const_vals, C_reduced)
        
        # Build complete concentration vector efficiently
        C_complete = np.zeros(len(self.r_n.species_names))
        
        # Fill in remaining species using pre-computed indices
        C_complete[self._remaining_indices] = C_reduced
        
        # Fill in eliminated species using pre-computed indices
        for species, value in recovered.items():
            idx = self._species_to_idx[species]
            C_complete[idx] = value
        
        return C_complete
    
    def _recover_species_fallback(self, elim_name, const_vals, C_reduced):
        """Fallback method using original symbolic evaluation if pre-compiled function fails."""
        # Build the substitution dictionary for sympy
        subs = dict(zip(self.remaining_species, C_reduced))
        subs.update(zip(self.const_syms, const_vals))
        
        expr = self.elimination_solutions[elim_name]
        # Substitute other eliminated species if present
        for other_name, other_expr in self.elimination_solutions.items():
            if other_name != elim_name:
                expr = expr.subs(other_name, other_expr)
        
        # Now substitute numeric values
        val = expr.evalf(subs=subs)
        try:
            return float(val)
        except TypeError:
            return val  # fallback: return symbolic if not numeric
    
    def get_symbolic_rhs(self) -> Tuple[List[sympy.Expr], List[sympy.Symbol], List[sympy.Symbol]]:
        """
        Returns the symbolic right-hand side of the ODEs for the reaction network.
        Returns:
            dCdt_sym: List of sympy expressions for dC/dt for each species
            species_syms: List of sympy symbols for each species
            rate_syms: List of sympy symbols for each rate constant
        """

        # Create symbolic variables for each species
        species_syms = sympy.symbols(self.r_n.species_names)
        
        # Create descriptive names for the symbolic rate constants
        rate_names = []
        for r_idx, (i, j, _) in enumerate(self.r_n.reactions):
            # Creates names like: k_0_C+C_B+C
            # reactant_str = self.r_n.all_complexes[i]
            # product_str = self.r_n.all_complexes[j]
            # rate_names.append(f'k_{r_idx};{reactant_str}->{product_str}')
            rate_names.append(f'k_{r_idx}')

        rate_syms = sympy.symbols(rate_names)

        # Initialize a list to hold the symbolic ODE for each species
        dCdt_sym = [sympy.Integer(0)] * len(species_syms)

        # Iterate through each reaction to build the symbolic ODEs
        for idx, (i, j, k_val) in enumerate(self.r_n.reactions):
            rate_k = rate_syms[idx]
            
            # Build the monomial for the reactant complex
            reactant_complex_stoich = self.r_n.Y[:, i]
            monomial = sympy.Integer(1)
            for species_idx, exponent in enumerate(reactant_complex_stoich):
                if exponent > 0:
                    monomial *= species_syms[species_idx]**exponent
            
            # Symbolic representation of the reaction rate
            reaction_rate_sym = rate_k * monomial
            
            # Stoichiometric change vector for the reaction
            stoich_change = self.r_n.Y[:, j] - self.r_n.Y[:, i]
            
            # Add this reaction's contribution to each species' ODE
            for species_idx, change in enumerate(stoich_change):
                if change != 0:
                    dCdt_sym[species_idx] += change * reaction_rate_sym
                
        return dCdt_sym, species_syms, rate_syms
    
    def get_symbolic_reduced_rhs(self):
        """
        Returns the symbolic right-hand side of the reduced ODEs (after eliminating variables using conservation laws).
        Returns:
            reduced_rhs: List of sympy expressions for dC/dt for each remaining species
            remaining_syms: List of sympy symbols for the remaining species
            const_syms: List of sympy symbols for the conservation constants
            rate_syms: List of sympy symbols for each rate constant
        """
        # Ensure conservation laws have been solved
        if self.elimination_solutions is None or self.remaining_species is None:
            self.solve_conservation_laws()

        # Get the full symbolic ODEs
        dCdt_sym, all_species_syms, rate_syms = self.get_symbolic_rhs()
        species_names = [str(s) for s in all_species_syms]
        species_map = {str(s): s for s in all_species_syms}
        remaining_syms = [species_map[s] for s in self.remaining_species]

        # Substitute eliminated species in the ODEs
        solutions = {species_map[k]: v for k, v in self.elimination_solutions.items()}
        substituted_rhs = [expr.subs(solutions, simultaneous=True) for expr in dCdt_sym]

        # Only keep the ODEs for the remaining species
        reduced_rhs = [substituted_rhs[species_names.index(s)] for s in self.remaining_species]

        return reduced_rhs, remaining_syms, self.const_syms, rate_syms
    
    def get_first_order_derivatives(self):
        """
        Returns first-order derivatives (Jacobians) of the reduced ODE system.
        These are independent of each other and can be computed separately.
        
        Returns:
            dR_dC, dR_dC_func: Jacobian w.r.t. remaining species
            dR_dl, dR_dl_func: Jacobian w.r.t. conservation constants  
            dR_dk, dR_dk_func: Jacobian w.r.t. rate constants
            remaining_syms: list of sympy symbols for the remaining species
        """
        reduced_rhs, remaining_syms, const_syms, rate_syms = self.get_symbolic_reduced_rhs()
        arg_syms = list(remaining_syms) + list(const_syms) + list(rate_syms)
        
        # First-order derivatives (independent of each other)
        dR_dC = sympy.Matrix(reduced_rhs).jacobian(remaining_syms)
        dR_dC_func = sympy.lambdify(arg_syms, dR_dC, modules='numpy')

        dR_dl = sympy.Matrix(reduced_rhs).jacobian(const_syms)
        dR_dl_func = sympy.lambdify(arg_syms, dR_dl, modules='numpy')

        dR_dk = sympy.Matrix(reduced_rhs).jacobian(rate_syms)
        dR_dk_func = sympy.lambdify(arg_syms, dR_dk, modules='numpy')
        
        return dR_dC, dR_dC_func, dR_dl, dR_dl_func, dR_dk, dR_dk_func, remaining_syms

    def get_second_order_derivatives(self, dR_dC=None, dR_dl=None):
        """
        Returns second-order derivatives that depend on first-order derivatives.
        
        Args:
            dR_dC: First-order derivative w.r.t. species (optional, computed if not provided)
            dR_dl: First-order derivative w.r.t. conservation constants (optional, computed if not provided)
            
        Returns:
            dR_dC_dk, dR_dC_dk_func: 2nd derivative w.r.t. species and rate constants
            dR_dl_dk, dR_dl_dk_func: 2nd derivative w.r.t. conservation constants and rate constants
        """
        if dR_dC is None or dR_dl is None:
            # Get first-order derivatives if not provided
            _, _, _, _, _, _, remaining_syms = self.get_first_order_derivatives()
            reduced_rhs, remaining_syms, const_syms, rate_syms = self.get_symbolic_reduced_rhs()
            if dR_dC is None:
                dR_dC = sympy.Matrix(reduced_rhs).jacobian(remaining_syms)
            if dR_dl is None:
                dR_dl = sympy.Matrix(reduced_rhs).jacobian(const_syms)
        else:
            _, remaining_syms, const_syms, rate_syms = self.get_symbolic_reduced_rhs()
        
        arg_syms = list(remaining_syms) + list(const_syms) + list(rate_syms)
        
        # Helper function for tensor evaluation
        def evaluate_tensor_func(func_array, *args):
            out = np.empty(func_array.shape, dtype=float)
            it = np.nditer(func_array, flags=['multi_index', 'refs_ok'])
            for x in it:
                idx = it.multi_index
                out[idx] = func_array[idx](*args)
            return out

        # dR_dC_dk: (m, n, p)
        m, n = dR_dC.shape
        p = len(rate_syms)
        dR_dC_dk = sympy.MutableDenseNDimArray(
            [[[dR_dC[i, j].diff(rate_syms[k]) for k in range(p)] for j in range(n)] for i in range(m)]
        )
        dR_dC_dk_func_array = np.empty((m, n, p), dtype=object)
        for i in range(m):
            for j in range(n):
                for k in range(p):
                    dR_dC_dk_func_array[i, j, k] = sympy.lambdify(arg_syms, dR_dC_dk[i, j, k], modules='numpy')
        def dR_dC_dk_func(*args):
            return evaluate_tensor_func(dR_dC_dk_func_array, *args)

        # dR_dl_dk: (m_dl, n_dl, p)
        m_dl, n_dl = dR_dl.shape
        dR_dl_dk = sympy.MutableDenseNDimArray(
            [[[dR_dl[i, j].diff(rate_syms[k]) for k in range(p)] for j in range(n_dl)] for i in range(m_dl)]
        )
        dR_dl_dk_func_array = np.empty((m_dl, n_dl, p), dtype=object)
        for i in range(m_dl):
            for j in range(n_dl):
                for k in range(p):
                    dR_dl_dk_func_array[i, j, k] = sympy.lambdify(arg_syms, dR_dl_dk[i, j, k], modules='numpy')
        def dR_dl_dk_func(*args):
            return evaluate_tensor_func(dR_dl_dk_func_array, *args)
        
        return dR_dC_dk, dR_dC_dk_func, dR_dl_dk, dR_dl_dk_func

    def get_hessian_derivatives(self, dR_dC=None):
        """
        Returns second-order derivatives (Hessian) with respect to concentrations.
        
        Args:
            dR_dC: First-order derivative w.r.t. species (optional, computed if not provided)
            
        Returns:
            dR_dC_dC, dR_dC_dC_func: Hessian w.r.t. species (2nd derivative)
        """
        if dR_dC is None:
            # Get first-order derivatives if not provided
            _, _, _, _, _, _, remaining_syms = self.get_first_order_derivatives()
            reduced_rhs, remaining_syms, const_syms, rate_syms = self.get_symbolic_reduced_rhs()
            dR_dC = sympy.Matrix(reduced_rhs).jacobian(remaining_syms)
        else:
            _, remaining_syms, const_syms, rate_syms = self.get_symbolic_reduced_rhs()
        
        arg_syms = list(remaining_syms) + list(const_syms) + list(rate_syms)
        
        # Helper function for tensor evaluation
        def evaluate_tensor_func(func_array, *args):
            out = np.empty(func_array.shape, dtype=float)
            it = np.nditer(func_array, flags=['multi_index', 'refs_ok'])
            for x in it:
                idx = it.multi_index
                out[idx] = func_array[idx](*args)
            return out

        # dR_dC_dC: (m, n, n) - Hessian with respect to concentrations
        m, n = dR_dC.shape
        dR_dC_dC = sympy.MutableDenseNDimArray(
            [[[dR_dC[i, j].diff(remaining_syms[k]) for k in range(n)] for j in range(n)] for i in range(m)]
        )
        dR_dC_dC_func_array = np.empty((m, n, n), dtype=object)
        for i in range(m):
            for j in range(n):
                for k in range(n):
                    dR_dC_dC_func_array[i, j, k] = sympy.lambdify(arg_syms, dR_dC_dC[i, j, k], modules='numpy')
        def dR_dC_dC_func(*args):
            return evaluate_tensor_func(dR_dC_dC_func_array, *args)
        
        return dR_dC_dC, dR_dC_dC_func

    def get_derivatives(self):
        """
        Returns symbolic and lambdified derivatives of the reduced ODE system with respect to remaining species, conservation constants, and rate constants.
        
        This is a convenience function that calls the modular derivative functions.
        For better performance, consider calling get_first_order_derivatives() and get_second_order_derivatives() separately.
        
        Returns:
            dR_dC, dR_dC_func: Jacobian w.r.t. remaining species
            dR_dl, dR_dl_func: Jacobian w.r.t. conservation constants
            dR_dk, dR_dk_func: Jacobian w.r.t. rate constants
            dR_dC_dk, dR_dC_dk_func: 2nd derivative w.r.t. species and rate constants
            dR_dl_dk, dR_dl_dk_func: 2nd derivative w.r.t. conservation constants and rate constants
            remaining_syms: list of sympy symbols for the remaining species
        """
        # Get first-order derivatives
        dR_dC, dR_dC_func, dR_dl, dR_dl_func, dR_dk, dR_dk_func, remaining_syms = self.get_first_order_derivatives()
        
        # Get second-order derivatives (passing the first-order ones to avoid recomputation)
        dR_dC_dk, dR_dC_dk_func, dR_dl_dk, dR_dl_dk_func = self.get_second_order_derivatives(dR_dC, dR_dl)
        
        return (
            dR_dC, dR_dC_func,
            dR_dl, dR_dl_func,
            dR_dk, dR_dk_func,
            dR_dC_dk, dR_dC_dk_func,
            dR_dl_dk, dR_dl_dk_func,
            remaining_syms
        )

    def dC_dl_func(self, C, const_vals, rates, dR_dC_func, dR_dl_func):
        args = list(C) + list(const_vals) + list(rates)
        dR_dC_eval = dR_dC_func(*args) 
        dR_dl_eval = dR_dl_func(*args)
        dC_dl = -np.tensordot(np.linalg.inv(dR_dC_eval), dR_dl_eval, axes = ([1], [0]))
        return dC_dl
    
    def dC_dk_func(self, C, const_vals, rates, dR_dC_func, dR_dk_func):
        args = list(C) + list(const_vals) + list(rates)
        dR_dC_eval = dR_dC_func(*args) 
        dR_dk_eval = dR_dk_func(*args)
        dC_dk = -np.tensordot(np.linalg.inv(dR_dC_eval), dR_dk_eval, axes = ([1], [0]))
        return dC_dk

    def dC_dl_laplacian_func(self, C, const_vals, rates, dR_dC_func, dR_dl_func, dR_dl_dl_func, dR_dC_dl_func):
        args = list(C) + list(const_vals) + list(rates)
        dR_dC_eval = dR_dC_func(*args) 
        dR_dl_eval = dR_dl_func(*args)
        dR_dl_dl_eval = dR_dl_dl_func(*args)
        dR_dC_dl_eval = dR_dC_dl_func(*args)

        M = np.linalg.inv(dR_dC_eval)
        Q = - np.tensordot(np.tensordot(M, dR_dC_dl_eval, axes = ([1], [0])), M, axes = ([1], [0])).swapaxes(-1,-2) # dM_{ij} / dl_k

        term_1 = - np.einsum('ijj->i', np.tensordot(Q, dR_dl_eval, axes = ([1], [0])))
        dR_dl_dl_contr = np.einsum('ijj->i', dR_dl_dl_eval)
        term_2 = - np.tensordot(M, dR_dl_dl_contr, axes = ([1], [0]))

        return term_1 + term_2

    def dC_func(self, t, C, rates, l_unit_vec, l_base, dR_dC_func, dR_dl_func):
        const_vals = l_unit_vec * t + l_base
        dC_dl = self.dC_dl_func(C, const_vals, rates, dR_dC_func, dR_dl_func)
        dC = np.tensordot(dC_dl, l_unit_vec, axes = ([1], [0]))
        return dC
    
    def dC_dl_dk_func(self, t, C, rates, l_unit_vec, l_base, dR_dC_func, dR_dl_func, dR_dC_dk_func, dR_dl_dk_func):
        const_vals = l_unit_vec * t + l_base
        args = list(C) + list(const_vals) + list(rates)

        dR_dC_eval = dR_dC_func(*args)
        M = np.linalg.inv(dR_dC_eval)
        dR_dl_eval = dR_dl_func(*args)
        dR_dC_dk_eval = dR_dC_dk_func(*args)
        dR_dl_dk_eval = dR_dl_dk_func(*args)

        term_1 = np.tensordot(
            np.tensordot(
                np.tensordot(
                    M, dR_dC_dk_eval, axes = ([1], [0])), 
                M, axes = ([1], [0])),
            dR_dl_eval, axes = ([2], [0])).swapaxes(-2,-1)

        term_2 = -np.tensordot(M, dR_dl_dk_eval, axes = ([1], [0]))

        dC_dl_dk = term_1 + term_2
        return dC_dl_dk
    
    def dC_and_dC_dk_combined_func(self, t, C_combined, rates, l_unit_vec, l_base, dR_dC_func, dR_dl_func, dR_dC_dk_func, dR_dl_dk_func):
        dC = self.dC_func(t, C_combined[:len(self.remaining_species)], rates, l_unit_vec, l_base, dR_dC_func, dR_dl_func)
        dC_dk = np.tensordot(self.dC_dl_dk_func(t, C_combined[:len(self.remaining_species)], rates, l_unit_vec, l_base, dR_dC_func, dR_dl_func, dR_dC_dk_func, dR_dl_dk_func), l_unit_vec, axes = ([1], [0]))
        dC_combined = np.concatenate([dC, dC_dk.flatten()], axis = 0)
        return dC_combined
    
    def convert_dC_combined(self, dC_combined):
        dC = dC_combined[:len(self.remaining_species)]
        dC_dk = dC_combined[len(self.remaining_species):].reshape(len(self.remaining_species), -1)
        return dC, dC_dk
    
    def compute_dC_dk_full(self, dC_dk, l_bool = False):
        """
        Computes the full dC_dk matrix including both eliminated and remaining species.
        
        Args:
            sim: Simulation object containing elimination solutions and remaining species
            dC_dk: Matrix of derivatives for remaining species
        
        Returns:
            numpy array: Full dC_dk matrix for all species
        """
        dsp_dk_list = []
        for sp, expr in self.elimination_solutions.items():
            dsp_dk = np.zeros_like(dC_dk[0])
            for (i, rem_sp) in enumerate(self.remaining_species):
                coef = float(expr.coeff(sympy.Symbol(rem_sp)))
                dsp_dk = dsp_dk + dC_dk[i] * coef
            dsp_dk_list.append(dsp_dk)

        # Create new dC_dk list with interleaved values
        new_dC_dk = []
        eliminated_species = list(self.elimination_solutions.keys())
        all_species = eliminated_species + self.remaining_species

        for sp in self.r_n.species_names:
            if sp in eliminated_species:
                # Get index in eliminated species list to find corresponding dsp_dk
                idx = eliminated_species.index(sp)
                if l_bool:
                    dsp_dk_list[idx][idx] += 1.0
                new_dC_dk.append(dsp_dk_list[idx])
            else:
                # Get index in remaining species list to find corresponding dC_dk
                idx = self.remaining_species.index(sp)
                new_dC_dk.append(dC_dk[idx])

        return np.array(new_dC_dk)
    
