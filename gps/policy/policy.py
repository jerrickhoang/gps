""" This file defines the base class for the policy. """
import abc


class Policy(object):
    """ Computes actions from states/observations. """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def act(self, x, t, noise):
        """
        :param x: State vector.
        :param t: Time step.
        :returns: A dU-dimensional action vector
        """
        raise NotImplementedError("Must be implemented in subclass.")
