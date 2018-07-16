"""Base class for survival models and related mixin classes."""

import abc
import inspect

import numpy as np

from ..utils import check_random_state


class Model(metaclass=abc.ABCMeta):
    """Abstract base class for survival models.

    Subclasses of Model should have an __init__() method of the form
        def __init__(self, a, b, ...):
            self.a = a
            self.b = b
            ...

    Each attribute 'a', 'b', ... should be a property instance with a setter
    method that performs validation for that __init__() parameter. Validation
    should not be done inside __init__() itself.

    Properties
    ----------
    model_type : str
        The name of this model type.
    random_state : numpy.random.RandomState
        This model's random number generator.
    summary : Summary
        A summary of this model.
    """
    model_type: str

    # Internal random number generator
    _random_state: np.random.RandomState = np.random.RandomState(None)

    @property
    def random_state(self):
        """This model's random number generator."""
        return self._random_state

    @random_state.setter
    def random_state(self, random_state):
        """Set the random number generator."""
        self._random_state = check_random_state(random_state)

    def __repr__(self):
        """Return the call used to initialize this model.

        Returns
        -------
        repr(self) : str
            A string which should be able to be used to instantiate a new
            identical model.
        """
        name = self.__class__.__name__
        param_names = inspect.signature(self.__init__).parameters.keys()
        kwargs = [f"{k}={repr(getattr(self, k))}" for k in param_names]
        return name + "(" + ", ".join(kwargs) + ")"

    @property
    def summary(self):
        """Return a summary of this model."""
        return Summary(self)


class Summary(object):
    """Summary of a survival model.

    Properties
    ----------
    model : Model
        The model being summarized.
    """
    model: Model

    def __init__(self, model):
        """Initialize a Summary object.

        Parameters
        ----------
        model : Model
            The model to be summarized.
        """
        self.model = model

    def __str__(self):
        """Return a basic string representation of the model."""
        return f"{self.model.model_type}\n{repr(self.model)}"


class Fittable(metaclass=abc.ABCMeta):
    """Abstract mixin class for models implementing a fit() method.

    Properties
    ----------
    fitted : bool
        Indicates whether this model has been fitted.
    """
    fitted: bool = False

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        """Fit this model to data.

        This function should set the `fitted` property to True and return self.
        """
        pass

    def check_fitted(self):
        """Check whether this model is fitted. If not, raise an exception."""
        if not self.fitted:
            raise RuntimeError(
                f"This {self.__class__.__name__} object is not fitted.")


class Predictor(metaclass=abc.ABCMeta):
    """Abstract mixin class for models implementing a predict() method."""

    @abc.abstractmethod
    def predict(self, *args, **kwargs):
        pass
