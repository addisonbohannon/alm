#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from alm.utility import inner_product, circulant_matrix


class Timeseries:
    """
    Defines a convenient class for computations on the observations
    """

    def __init__(self, observation):
        """
        :param observation: observation_length x dimension numpy array
        """

        if len(np.shape(observation)) != 2:
            raise TypeError('Observation dimension invalid; expected m x d numpy array.')

        self.observation = observation
        self.observation_length, self.dimension = observation.shape

    def Y(self, model_order):
        """
        Returns observations without the first (model_order) observations
        :param model_order: positive integer
        :return stacked observation: (observation_length-model_order) x dimension numpy array
        """

        if not isinstance(model_order, int) or model_order < 1:
            raise TypeError('Model order must be a positive integer.')

        return self.observation[model_order:, :]

    def X(self, model_order):
        """
        Return the circulant observation matrix
        :param model_order: positive integer
        :return circulant observation: (observation_length-model_order) x (model_order*dimension) numpy array
        """

        if not isinstance(model_order, int) or model_order < 1:
            raise TypeError('Model order must be a positive integer.')
        
        _X, _ = circulant_matrix(self.observation, model_order)
        
        return _X

    def YtY(self, model_order):
        """
        Return the sample covariance of the observation
        :param model_order: positive integer
        :return sample covariance: (observation_length-model_order) x dimension x dimension numpy array
        """

        if not isinstance(model_order, int) or model_order < 1:
            raise TypeError('Model order must be a positive integer.')
        
        _Y = self.Y(model_order)
        
        return inner_product(_Y, _Y) / (self.observation_length - model_order)

    def XtX(self, model_order):
        """
        Returns the sample autocovariance of the observation
        :param model_order: positive integer
        :return sample autocovariance: (observation_length-model_order) x (model_order*dimension) x (model_order*dimension)
        """

        if not isinstance(model_order, int) or model_order < 1:
            raise TypeError('Model order must be a positive integer.')
        
        _X = self.X(model_order)
        
        return np.dot(_X.T, _X) / (self.observation_length - model_order)

    def XtY(self, model_order):
        """
        Returns the sample autocovariance of the observation
        :param model_order: positive integer
        :return sample autocovariance: (observation_length-model_order) x (model_order*dimension) x dimension
        """

        if not isinstance(model_order, int) or model_order < 1:
            raise TypeError('Model order must be a positive integer.')
        
        _X = self.X(model_order)
        _Y = self.Y(model_order)

        return np.dot(_X.T, _Y) / (self.observation_length - model_order)
