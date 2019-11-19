#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from timeit import default_timer as timer
import numpy as np
import scipy.linalg as sl
from alm.utility import initialize_autoregressive_components
from alm.solver import negative_log_likelihood, coef_update
from alm.utility import package_observations


class Alm:

    def __init__(self, coef_penalty_type='l1', step_size=1e-1, tol=1e-4, max_iter=int(2.5e3), solver='bcd',
                 verbose=False):
        """
        Instantiate ALM model
        :param coef_penalty_type: None, l0, or l1
        :param step_size: float; (0, 1)
        :param tol: positive float
        :param max_iter: positive integer
        :param solver: bcd, altmin, or palm
        :param verbose: boolean
        """

        if coef_penalty_type is None:
            self.coef_penalty_type = coef_penalty_type
        elif coef_penalty_type == 'l0':
            self.coef_penalty_type = coef_penalty_type
        elif coef_penalty_type == 'l1':
            self.coef_penalty_type = coef_penalty_type
        else:
            raise ValueError(coef_penalty_type + ' is not a valid penalty type: None, l0, or l1.')
        if isinstance(float(step_size), float) and step_size < 1:
            self.step_size = float(step_size)
        else:
            raise ValueError('Step size must be a float less than 1.')
        if isinstance(tol, float) and tol > 0:
            self.tol = tol
        else:
            raise ValueError('Tolerance must be a positive float.')
        if np.issubdtype(type(max_iter), np.int) and max_iter > 0:
            self.max_iter = max_iter
        else:
            raise TypeError('Max iteration must be a positive integer.')
        if solver == 'bcd':
            from alm.solver import component_update_bcd
            self.solver = solver
            self.update_coef = coef_update
            self.update_component = component_update_bcd
        elif solver == 'altmin':
            self.solver = solver
            from alm.solver import component_update_altmin
            self.update_coef = coef_update
            self.update_component = component_update_altmin
        elif solver == 'palm':
            self.solver = solver
            from alm.solver import component_update_palm, coef_update_palm
            self.update_coef = coef_update_palm
            self.update_component = component_update_palm
        else:
            raise TypeError('Solver must be bcd, altmin, or palm.')
        if isinstance(verbose, bool):
            self.verbose = verbose
        else:
            raise TypeError('Verbose must be a boolean.')
        self.component, self.mixing_coef, self.solver_time, self.nll, self.residual, self.stop_condition \
            = [], [], [], [], [], []

    def fit(self, obs, model_ord, num_comps, penalty_param, num_starts=5, initial_comps=None, return_path=False,
            return_all=False):
        """
        Fit the ALMM model to obs
        :param obs: list of obs_len x signal_dim numpy array
        :param model_ord: positive integer
        :param num_comps: positive integer
        :param penalty_param: positive float
        :param num_starts: positive integer
        :param initial_comps: num_comps x model_ord*signal_dim x signal_dim numpy array
        :param return_path: boolean
        :param return_all: boolean
        :return ar_comps: [list of] num_comps x model_ord*signal_dim x signal_dim numpy array
        :return mixing_coef: [list of] num_observations x num_comps numpy array
        :return nll: [list of] float
        :return solver_time: list of float
        """

        if not np.issubdtype(type(num_comps), np.int) or num_comps < 1:
            raise TypeError('Number of components must be a positive integer.')
        if not np.issubdtype(type(model_ord), np.int) or model_ord < 1:
            raise TypeError('Model order must be a positive integer.')
        if not isinstance(penalty_param, float) and penalty_param < 0:
            raise ValueError('Penalty parameter must be a positive float.')
        if not np.issubdtype(type(num_starts), np.int) or num_starts < 1:
            raise ValueError('Number of starts must be a positive integer.')
        _, signal_dimension = obs[0].shape
        if initial_comps is None:
            initial_comps = [initialize_autoregressive_components(num_comps, model_ord, signal_dimension)
                             for _ in range(num_starts)]
        elif np.shape(initial_comps) != (num_starts, num_comps, model_ord * signal_dimension,
                                         signal_dimension):
            raise ValueError('Initial dictionary estimate must be list of num_comps x model_ord*signal_dim'
                             + ' x signal_dim numpy arrays.')
        else:
            initial_comps = [np.array([component_kj / sl.norm(component_kj[:]) for component_kj in component_k])
                             for component_k in initial_comps]
        if not isinstance(return_path, bool):
            raise TypeError('Return path must be a boolean.')
        if not isinstance(return_all, bool):
            raise TypeError('Return all must be a boolean.')

        self.component, self.mixing_coef, self.solver_time, self.nll, self.residual, self.stop_condition \
            = [], [], [], [], [], []
        if self.verbose:
            print('-Formatting data...', end=" ", flush=True)
        YtY, XtY, XtX = package_observations(obs, model_ord)
        if self.verbose:
            print('Complete.')
        if self.verbose:
            print('-Fitting model to data...')
        for start_k in range(num_starts):
            if self.verbose and num_starts > 1:
                print('--Start: ' + str(start_k))
            component_k, mixing_coef_k, component_residual_k, coef_residual_k, stop_condition_k, solver_time_k \
                = self._fit(XtX, XtY, model_ord, num_comps, penalty_param, initial_comps[start_k],
                            return_path=return_path)
            self.component.append(component_k)
            self.mixing_coef.append(mixing_coef_k)
            self.residual.append((component_residual_k, coef_residual_k))
            self.stop_condition.append(stop_condition_k)
            self.solver_time.append(solver_time_k)
        if self.verbose:
            print('-Complete.')
        if self.verbose:
            print('-Computing likelihood...', end=" ", flush=True)
        for component_k, mixing_coef_k in zip(self.component, self.mixing_coef):
            if return_path:
                nll_k = [negative_log_likelihood(YtY, XtX, XtY, component_ki, mixing_coef_ki, penalty_param,
                                                 self.coef_penalty_type) 
                         for component_ki, mixing_coef_ki in zip(component_k, mixing_coef_k)]
            else:
                nll_k = negative_log_likelihood(YtY, XtX, XtY, component_k, mixing_coef_k, penalty_param,
                                                self.coef_penalty_type)
            self.nll.append(nll_k)
        if self.verbose:
            print('Complete.')

        if num_starts == 1:
            return self.component.pop(), self.mixing_coef.pop(), self.nll.pop(), self.solver_time.pop()
        elif return_all:
            return self.component, self.mixing_coef, self.nll, self.solver_time
        else:
            opt = 0
            if return_path:
                nll_min = self.nll[0][-1]
                for i, nll_k in enumerate(self.nll):
                    if nll_k[-1] < nll_min:
                        opt = i
                        nll_min = nll_k[-1]
            else:
                nll_min = self.nll[0]
                for i, nll_k in enumerate(self.nll):
                    if nll_k < nll_min:
                        opt = i
                        nll_min = nll_k
            return self.component[opt], self.mixing_coef[opt], self.nll[opt], self.solver_time[opt]

    def _fit(self, XtX, XtY, model_ord, num_comps, penalty_param, initial_comps, return_path=False):
        """
        Fit ALMM according to algorithm specified by solver
        :param XtX: num_obs x model_ord*signal_dim x model_ord*signal_dim numpy array
        :param XtY: num_obs x model_ord*signal_dim x signal_dim numpy array
        :param model_ord: positive integer
        :param num_comps: positive integer
        :param penalty_param: float
        :param initial_comps: num_comps x model_ord*signal_dim x signal_dim numpy array
        :param return_path: boolean
        :return ar_comps: [list of] num_comps x model_ord*signal_dim x signal_dim numpy array
        :return mixing_coef: [list of] num_obs x num_comps numpy array
        :return coef_residual: list of float
        :return comp_residual: list of float
        :return stop_condition: string
        :return elapsed_time: list of float
        """

        def compute_component_residual(component_diff):
            return sl.norm(component_diff[:]) / (num_comps ** (1 / 2) * model_ord ** (1 / 2) * signal_dim)

        def compute_coef_residual(coef_diff):
            return sl.norm(coef_diff[:]) / (num_obs ** (1 / 2) * num_comps ** (1 / 2))

        def stopping_condition(current_step, residual_1, residual_2):
            if current_step == 0:
                return False
            elif residual_1[-1] >= self.tol * residual_1[0]:
                return False
            elif residual_2[-1] >= self.tol * residual_2[0]:
                return False
            else:
                return True

        start_time = timer()
        num_obs = len(XtY)
        _, signal_dim = XtY[0].shape
        ar_comps = np.copy(initial_comps)
        mixing_coef = np.zeros([num_obs, num_comps])
        mixing_coef, _ = coef_update(XtX, XtY, initial_comps, mixing_coef, penalty_param,
                                     self.coef_penalty_type, self.step_size)
        elapsed_time = [timer() - start_time]
        stop_condition = 'maximum iteration'
        if return_path:
            comp_path, coef_path = [np.copy(ar_comps)], [np.copy(mixing_coef)]
        comp_residual, coef_residual = [], []
        for step in range(self.max_iter):
            ar_comps, comp_change = self.update_component(XtX, XtY, ar_comps, mixing_coef, self.step_size)
            mixing_coef, coef_change = self.update_coef(XtX, XtY, ar_comps, mixing_coef, penalty_param,
                                                        self.coef_penalty_type, self.step_size)
            if return_path:
                comp_path.append(np.copy(ar_comps))
                coef_path.append(np.copy(mixing_coef))
            comp_residual.append(compute_component_residual(comp_change))
            coef_residual.append(compute_coef_residual(coef_change))
            elapsed_time.append(timer() - start_time)
            if stopping_condition(step, comp_residual, coef_residual):
                stop_condition = 'relative tolerance'
                break
        if self.verbose:
            print('*Solver: ' + self.solver)
            print('*Stopping condition: ' + stop_condition)
            print('*Iterations: ' + str(step + 1))
            print('*Duration: ' + str(elapsed_time[-1]) + 's')

        if return_path:
            return comp_path, coef_path, comp_residual, coef_residual, stop_condition, elapsed_time
        else:
            return ar_comps, mixing_coef, coef_residual, comp_residual, stop_condition, elapsed_time
