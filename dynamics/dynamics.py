import abc

import numpy as np


class Dynamics(object):
    """ Dynamics superclass. """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def update_prior(self, sample):
        """ Update dynamics prior. """
        raise NotImplementedError("Must be implemented in subclass.")

    @abc.abstractmethod
    def get_prior(self):
        """ Returns prior object. """
        raise NotImplementedError("Must be implemented in subclass.")

    @abc.abstractmethod
    def fit(self, sample_list):
        """ Fit dynamics. """
        raise NotImplementedError("Must be implemented in subclass.")
