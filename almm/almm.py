#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from itertools import product
from timeit import default_timer as timer
import numpy as np
import scipy.linalg as sl
from almm.utility import train_val_split, initialize_components
from almm.solver import negative_log_likelihood, coef_update
from almm.timeseries import Timeseries


class Almm:

    def __init__(self, coef_penalty_type='l1', step_size=1e-1, tol=1e-4, max_iter=int(2.5e3), solver='bcd',
                 verbose=False):
        """
        Instantiate ALMM model
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
        if isinstance(max_iter, int) and max_iter > 0:
            self.max_iter = max_iter
        else:
            raise TypeError('Max iteration must be a positive integer.')
        if solver == 'bcd':
            from almm.solver import component_update_bcd
            self.solver = solver
            self.update_coef = coef_update
            self.update_component = component_update_bcd
        elif solver == 'altmin':
            self.solver = solver
            from almm.solver import component_update_altmin
            self.update_coef = coef_update
            self.update_component = component_update_altmin
        elif solver == 'palm':
            self.solver = solver
            from almm.solver import component_update_palm, coef_update_palm
            self.update_coef = coef_update_palm
            self.update_component = component_update_palm
        else:
            raise TypeError('Solver must be bcd, altmin, or palm.')
        if isinstance(verbose, bool):
            self.verbose = verbose
        else:
            raise TypeError('Verbose must be a boolean.')
        self.component, self.mixing_coef, self.solver_time, self.nll, self.params, self.residual, self.stop_condition \
            = [], [], [], [], [], [], []

    def fit(self, observation, model_order, num_components, penalty_parameter, num_starts=5, initial_component=None,
            return_path=False, return_all=False):
        """
        Fit the ALMM model to observations
        :param observation: list of observation_length x signal_dimension numpy array
        :param model_order: positive integer
        :param num_components: positive integer
        :param penalty_parameter: positive float
        :param num_starts: positive integer
        :param initial_component: num_components x model_order*signal_dimension x signal_dimension numpy array
        :param return_path: boolean
        :param return_all: boolean
        :return component: [list of] num_components x model_order*signal_dimension x signal_dimension numpy array
        :return mixing_coef: [list of] num_observations x num_components numpy array
        :return nll: [list of] float
        :return solver_time: list of float
        """

        if not isinstance(num_components, int) or num_components < 1:
            raise TypeError('Number of components must be a positive integer.')
        if not isinstance(model_order, int) or model_order < 1:
            raise TypeError('Model order must be a positive integer.')
        if not isinstance(penalty_parameter, float) and penalty_parameter < 0:
            raise ValueError('Penalty parameter must be a positive float.')
        if not isinstance(num_starts, int) or num_starts < 1:
            raise ValueError('Number of starts must be a positive integer.')
        _, signal_dimension = observation[0].shape
        if initial_component is None:
            initial_component = [initialize_components(num_components, model_order, signal_dimension)
                                 for _ in range(num_starts)]
        elif np.shape(initial_component) != (num_starts, num_components, model_order * signal_dimension,
                                             signal_dimension):
            raise ValueError('Initial dictionary estimate must be list of num_components x model_order*signal_dimension'
                             + ' x signal_dimension numpy arrays.')
        else:
            initial_component = [np.array([component_kj / sl.norm(component_kj[:]) for component_kj in component_k])
                                 for component_k in initial_component]
        if not isinstance(return_path, bool):
            raise TypeError('Return path must be a boolean.')
        if not isinstance(return_all, bool):
            raise TypeError('Return all must be a boolean.')

        self.component, self.mixing_coef, self.solver_time, self.nll, self.params, self.residual, self.stop_condition \
            = [], [], [], [], [], [], []
        if self.verbose:
            print('-Formatting data...', end=" ", flush=True)
        YtY, XtX, XtY = [], [], []
        for observation_i in observation:
            observation_i = Timeseries(observation_i)
            YtY.append(observation_i.YtY(model_order))
            XtX.append(observation_i.XtX(model_order))
            XtY.append(observation_i.XtY(model_order))
        if self.verbose:
            print('Complete.')
        if self.verbose:
            print('-Fitting model...')
        for start_k in range(num_starts):
            if self.verbose and num_starts > 1:
                print('--Start: ' + str(start_k))
            component_k, mixing_coef_k, component_residual_k, coef_residual_k, stop_condition_k, solver_time_k \
                = self._fit(XtX, XtY, model_order, num_components, penalty_parameter, initial_component[start_k],
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
                nll_k = [negative_log_likelihood(YtY, XtX, XtY, Dis, Cis, penalty_parameter, self.coef_penalty_type)
                         for Dis, Cis in zip(component_k, mixing_coef_k)]
            else:
                nll_k = negative_log_likelihood(YtY, XtX, XtY, component_k, mixing_coef_k, penalty_parameter,
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

    def fit_cv(self, observation, model_order=None, num_components=None, penalty_parameter=None, num_starts=5,
               val_pct=0.25, return_path=False, return_all=False):
        """
        Fit the ALMM model to observations using cross-validation
        :param observation: list of observation_length x signal_dimension numpy array
        :param model_order: list of positive integer
        :param num_components: list of positive integer
        :param penalty_parameter: list of positive float
        :param num_starts: positive integer
        :param val_pct: float; (0, 1)
        :param return_path: boolean
        :param return_all: boolean
        :return component: nested list of numpy array
        :return mixing_coef: nested list of numpy array
        :return nll: nested list of float
        :return params: list of tuples
        """

        num_observations = len(observation)
        _, signal_dimension = observation[0].shape
        if isinstance(model_order, int):
            if model_order > 0:
                model_order = [model_order]
            else:
                raise ValueError('Model order must be a positive integer or list of positive integers.')
        elif isinstance(model_order, list):
            if not all([isinstance(i, int) and i > 0 for i in model_order]):
                raise ValueError('Model order must be a positive integer or list of positive integers.')
        elif model_order is None:
            model_order = [1]
        else:
            raise TypeError('Model order must be a positive integer or list of positive integers.')
        if isinstance(num_components, int):
            if num_components > 0:
                num_components = [num_components]
            else:
                raise ValueError('Number of components must be a positive integer or list of positive integers.')
        elif isinstance(num_components, list):
            if not all([isinstance(i, int) and i > 0 for i in num_components]):
                raise ValueError('Number of components must be a positive integer or list of positive integers.')
        elif num_components is None:
            num_components = [1]
        else:
            raise TypeError('Number of components must be a positive integer or list of positive integers.')
        if isinstance(penalty_parameter, float):
            if penalty_parameter >= 0:
                penalty_parameter = [penalty_parameter]
            else:
                raise ValueError('Penalty parameter must be a non-negative float or list of non-negative floats.')
        elif isinstance(penalty_parameter, list):
            if not all([isinstance(i, float) and i >= 0 for i in penalty_parameter]):
                raise ValueError('Penalty parameter must be a non-negative float or list of non-negative floats.')
        elif penalty_parameter is None:
            penalty_parameter = [0]
        else:
            raise TypeError('Penalty parameter must be a non-negative float or list of non-negative floats.')
        if not isinstance(num_starts, int) or num_starts < 1:
            raise TypeError('Number of starts must be a positive integer.')

        def zip_coef(coef_train, coef_val):
            if return_path:
                zipped_coef = []
                for coef_to_zip_train, coef_to_zip_val in zip(coef_train, coef_val):
                    coef_to_zip = [i for i in zip(train_idx, list(coef_to_zip_train))]
                    coef_to_zip.extend([i for i in zip(val_idx, list(coef_to_zip_val))])
                    coef_to_zip.sort()
                    zipped_coef.append(np.array([c for _, c in coef_to_zip]))
            else:
                zipped_coef = [i for i in zip(train_idx, list(coef_train))]
                zipped_coef.extend([i for i in zip(val_idx, list(coef_val))])
                zipped_coef.sort()
                zipped_coef = np.array([c for _, c in zipped_coef])

            return zipped_coef

        def fit_coef(autocorrelation, correlation, components, penalty_param, coef_penalty_type):
            coef, _ = coef_update(autocorrelation, correlation, components,
                                  np.zeros([num_observations, num_components]), penalty_param, coef_penalty_type)
            return coef

        if self.verbose:
            print('-Formatting and splitting data...', end=" ", flush=True)
        train_idx, val_idx = train_val_split(len(observation), val_pct)
        observation = [Timeseries(ts_i) for ts_i in observation]
        observation_train = [observation[i] for i in train_idx]
        observation_val = [observation[i] for i in val_idx]
        if self.verbose:
            print('Complete.')
        self.component, self.mixing_coef, self.nll, self.params = [], [], [], []
        for (model_order_i, num_components_i, penalty_parameter_i) in product(model_order, num_components,
                                                                              penalty_parameter):
            if self.verbose:
                print('-Parameters: p=' + str(model_order_i) + ', r=' + str(num_components_i) + ', mu='
                      + str(penalty_parameter_i))
            XtX_train, XtY_train = [ob.XtX(model_order_i) for ob in observation_train], [ob.XtY(model_order_i)
                                                                                         for ob in observation_train]
            YtY_val, XtX_val, XtY_val = [ob.YtY(model_order_i) for ob in observation_val], \
                                        [ob.XtX(model_order_i) for ob in observation_val], \
                                        [ob.XtY(model_order_i) for ob in observation_val]
            component_i, mixing_coef_i, nll_i = [], [], []
            for start_k in range(num_starts):
                if self.verbose and num_starts > 1:
                    print('--Start: ' + str(start_k))
                initial_component = initialize_components(num_components, model_order, signal_dimension)
                component_i_k, mixing_coef_i_k_t, _, _, _, _ = self._fit(XtX_train, XtY_train, model_order_i,
                                                                         num_components_i, penalty_parameter_i,
                                                                         self.coef_penalty_type, initial_component,
                                                                         solver=self.solver, max_iter=self.max_iter,
                                                                         step_size=self.step_size, tol=self.tol,
                                                                         return_path=return_path, verbose=self.verbose)
                component_i.append(component_i_k)
                if self.verbose:
                    print('---Fitting coefficients and computing likelihood... ', end=" ", flush=True)
                if return_path:
                    mixing_coef_i_k_v = [fit_coef(XtX_val, XtY_val, D_iii, penalty_parameter_i, self.coef_penalty_type)
                                         for D_iii in component_i_k]
                    mixing_coef_i.append(zip_coef(mixing_coef_i_k_t, mixing_coef_i_k_v))
                    nll_i.append([negative_log_likelihood(YtY_val, XtX_val, XtY_val, D_iii, C_iiiv, penalty_parameter_i,
                                                          self.coef_penalty_type)
                                  for D_iii, C_iiiv in zip(component_i_k, mixing_coef_i_k_v)])
                else:
                    mixing_coef_i_k_v = fit_coef(XtX_val, XtY_val, component_i_k[-1], penalty_parameter_i,
                                                 self.coef_penalty_type)
                    mixing_coef_i.append(zip_coef(mixing_coef_i_k_t, mixing_coef_i_k_v))
                    nll_i.append(negative_log_likelihood(YtY_val, XtX_val, XtY_val, component_i_k[-1],
                                                         mixing_coef_i_k_v, penalty_parameter_i,
                                                         self.coef_penalty_type))
                if self.verbose:
                    print('Complete.')
            if self.verbose:
                print('-Complete.')
            self.component.append(component_i)
            self.mixing_coef.append(mixing_coef_i)
            self.nll.append(nll_i)
            self.params.append((model_order_i, num_components_i, penalty_parameter_i))

        if len(self.params) == 1:
            return self.component.pop(), self.mixing_coef.pop(), self.nll.pop(), self.params.pop()
        elif return_all:
            return self.component, self.mixing_coef, self.nll, self.params
        else:
            opt = 0
            if return_path:
                nll_min = self.nll[0][-1]
                for i, nll_i in enumerate(self.nll):
                    if nll_i[-1] < nll_min:
                        opt = i
                        nll_min = nll_i[-1]
            else:
                nll_min = self.nll[0]
                for i, nll_i in enumerate(self.nll):
                    if nll_i < nll_min:
                        opt = i
                        nll_min = nll_i
            return self.component[opt], self.mixing_coef[opt], self.nll[opt], self.params[opt]

    def _fit(self, XtX, XtY, model_order, num_components, penalty_parameter, initial_component, return_path=False):
        """
        Fit ALMM according to algorithm specified by solver
        :param XtX: num_observations x model_order*signal_dimension x model_order*signal_dimension numpy array
        :param XtY: num_observations x model_order*signal_dimension x signal_dimension numpy array
        :param model_order: positive integer
        :param num_components: positive integer
        :param penalty_parameter: float
        :param initial_component: num_components x model_order*signal_dimension x signal_dimension numpy array
        :param return_path: boolean
        :return component: [list of] num_components x model_order*signal_dimension x signal_dimension numpy array
        :return mixing_coef: [list of] num_observations x num_components numpy array
        :return coef_residual: list of float
        :return component_residual: list of float
        :return stop_condition: string
        :return elapsed_time: list of float
        """

        def compute_component_residual(component_diff):
            return sl.norm(component_diff[:]) / (num_components ** (1 / 2) * model_order ** (1 / 2) * signal_dimension)

        def compute_coef_residual(coef_diff):
            return sl.norm(coef_diff[:]) / (num_observations ** (1 / 2) * num_components ** (1 / 2))

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
        num_observations = len(XtY)
        _, signal_dimension = XtY[0].shape
        component = np.copy(initial_component)
        mixing_coef = np.zeros([num_observations, num_components])
        mixing_coef, _ = coef_update(XtX, XtY, initial_component, mixing_coef, penalty_parameter,
                                     self.coef_penalty_type, self.step_size)
        elapsed_time = [timer() - start_time]
        stop_condition = 'maximum iteration'
        if return_path:
            component_path, coef_path = [np.copy(component)], [np.copy(mixing_coef)]
        component_residual, coef_residual = [], []
        for step in range(self.max_iter):
            component, component_change = self.update_component(XtX, XtY, component, mixing_coef, self.step_size)
            mixing_coef, coef_change = self.update_coef(XtX, XtY, component, mixing_coef, penalty_parameter,
                                                        self.coef_penalty_type, self.step_size)
            if return_path:
                component_path.append(np.copy(component))
                coef_path.append(np.copy(mixing_coef))
            component_residual.append(compute_component_residual(component_change))
            coef_residual.append(compute_coef_residual(coef_change))
            elapsed_time.append(timer() - start_time)
            if stopping_condition(step, component_residual, coef_residual):
                stop_condition = 'relative tolerance'
                break
        if self.verbose:
            print('*Solver: ' + self.solver)
            print('*Stopping condition: ' + stop_condition)
            print('*Iterations: ' + str(step))
            print('*Duration: ' + str(elapsed_time[-1]) + 's')

        if return_path:
            return component_path, coef_path, component_residual, coef_residual, stop_condition, elapsed_time
        else:
            return component, mixing_coef, coef_residual, component_residual, stop_condition, elapsed_time
