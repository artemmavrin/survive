"""Base class for survival models and related mixin classes."""

import abc
import inspect

import numpy as np

from ..utils import check_random_state


class Model(metaclass=abc.ABCMeta):
    """Abstract base class for survival models.

    Notes
    -----
    Subclasses of :class:`Model` should have an :func:`__init__` method of the
    form::

        def __init__(self, a, b, ...):
            self.a = a
            self.b = b
            ...

    Each attribute `a`, `b`, ... should be a :class:`property` with a setter
    method that performs validation for that :func:`__init__` parameter.
    Validation should not be done inside :func:`__init__` itself.
    """
    model_type: str

    # Internal random number generator (RNG)
    _random_state: np.random.RandomState = np.random.RandomState(None)

    # Internal seed for this model's RNG
    _random_state_seed = None

    @property
    def random_state(self):
        """Seed for this model's random number generator. This may not be an
        :class:`numpy.random.RandomState` instance. The internal RNG is not a
        public attribute and should not be used directly.

        Returns
        -------
        random_state : object
            The seed for this model's RNG.
        """
        return self._random_state_seed

    @random_state.setter
    def random_state(self, random_state):
        """Set the random number generator."""
        self._random_state_seed = random_state
        self._random_state = check_random_state(random_state)

    @property
    def as_string(self):
        """String representation of this model.

        Notes
        -----
        The formatting algorithm is modified from :func:`sklearn.base._pprint`.

        Returns
        -------
        model_string : str
            A pretty-printed string representation of this model which should be
            able to be used to instantiate a new identical model.
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

    def __repr__(self):
        return self.as_string

    @property
    def summary(self):
        """Structure summarizing this model.

        Returns
        -------
        summary : survive.base.Summary
            This model's summary.

        See Also
        --------
        survive.base.Summary
        """
        return Summary(self)


class Summary(object):
    """Base class for summaries of survival models (intended for subclassing).

    Parameters
    ----------
    model : survive.base.Model
        The :class:`Model` being summarized.

    Attributes
    ----------
    model : survive.base.Model
        The :class:`Model` being summarized.
    """
    model: Model

    def __init__(self, model):
        self.model = model

    def __repr__(self):
        """Return a basic string representation of the model."""
        return self.model.model_type


class Fittable(metaclass=abc.ABCMeta):
    """Abstract mixin class for models implementing a :func:`fit` method."""
    fitted: bool = False

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        """Fit this model to data.

        Returns
        -------
        self : Model
            The :class:`Model` being fitted.
        """
        # Intended implementation:
        # This function should set self.fitted = True and return self.
        pass

    def check_fitted(self):
        """Check whether this model is fitted. If not, raise an exception."""
        if not self.fitted:
            raise RuntimeError(
                f"This {self.__class__.__name__} object is not fitted.")


class Predictor(metaclass=abc.ABCMeta):
    """Abstract mixin class for models implementing a :func:`predict` method."""

    @abc.abstractmethod
    def predict(self, *args, **kwargs):
        """Make predictions.

        Returns
        -------
        predictions : object
            The predictions.
        """
        pass
