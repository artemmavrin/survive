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
    random_state : any
        Seed for this model's random number generator.
        NB: this may not be a random number generator (i.e., a
        numpy.random.RandomState object). The actual RNG is not a public
        attribute and should not be used directly.
    """
    model_type: str

    # Internal random number generator (RNG)
    _random_state: np.random.RandomState = np.random.RandomState(None)

    # Internal seed for this model's RNG
    _random_state_seed = None

    @property
    def random_state(self):
        """Seed for this model's random number generator."""
        return self._random_state_seed

    @random_state.setter
    def random_state(self, random_state):
        """Set the random number generator."""
        self._random_state_seed = random_state
        self._random_state = check_random_state(random_state)

    def __repr__(self):
        """Return a pretty-printed call used to initialize this model.

        Algorithm modified from sklearn.base._pprint().

        Returns
        -------
        repr(self) : str
            A string which should be able to be used to instantiate a new
            identical model.
        """
        class_name = self.__class__.__name__
        offset = len(class_name) + 1
        max_line_length = 75

        keys = sorted(inspect.signature(self.__init__).parameters.keys())
        params = {k: getattr(self, k) for k in keys}

        params_list = list()
        this_line_length = offset
        line_sep = ",\n" + offset * " "
        for i, (k, v) in enumerate(params.items()):
            if isinstance(v, float):
                this_repr = f"{k}={str(v)}"
            else:
                this_repr = f"{k}={repr(v)}"
            if i > 0:
                if (this_line_length + len(this_repr) >= max_line_length
                        or "\n" in this_repr):
                    # Start a new line
                    params_list.append(line_sep)
                    this_line_length = len(line_sep)
                else:
                    # Continue current line
                    params_list.append(", ")
                    this_line_length += 2
            params_list.append(this_repr)
            this_line_length += len(this_repr)

        lines = "".join(params_list)
        lines = "\n".join(l.rstrip() for l in lines.splitlines())

        return f"{class_name}({lines})"

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
        return self.model.model_type


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
