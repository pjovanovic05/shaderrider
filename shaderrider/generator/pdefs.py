from abc import ABCMeta, abstractmethod


class PlatformFactory(object):
    """Abstract factory that defines a platform."""
    __metaclass__ = ABCMeta

    @abstractmethod
    def init_platform(self):
        """
        Performs platform initialization, like context and queue creation.
        """
        raise NotImplementedError

    @abstractmethod
    def finalize_platform(self):
        """
        Finalizes the platform, closing the queues and contexts.
        """
        raise NotImplementedError

    @abstractmethod
    def create_valuation(self):
        raise NotImplementedError

    @abstractmethod
    def create_function(self, expressions=None, updates=None, name=None, skip_platform_opts=False):
        raise NotImplementedError

    @abstractmethod
    def create_op(self, type_name, operands, params):
        raise NotImplementedError

    # ARRAY CREATION
    @abstractmethod
    def empty(self, shape, dtype=None, order='C', name=None):
        raise NotImplementedError

    @abstractmethod
    def empty_like(self, a, dtype=None, order='C', name=None):
        raise NotImplementedError

    @abstractmethod
    def eye(self, N, M=0, k=0, dtype=None, const=False, name=None):
        raise NotImplementedError

    @abstractmethod
    def identity(self, N, dtype=None, const=False, name=None):                      # TODO do we need this?
        raise NotImplementedError

    @abstractmethod
    def ones(self, shape, dtype=None, order='C', const=False, name=None):
        raise NotImplementedError

    @abstractmethod
    def ones_like(self, a, dtype=None, order='C', const=False, name=None):
        raise NotImplementedError

    @abstractmethod
    def from_data(self):                    # TODO parameters?
        raise NotImplementedError

    @abstractmethod
    def arange(self, start, stop, step=None, dtype=None, const=False, name=None):
        raise NotImplementedError

    @abstractmethod
    def linspace(self, start, stop, num=None, endpoint=None, const=False, name=None):
        raise NotImplementedError

    @abstractmethod
    def logspace(self, start, stop, num, endpoint, base, const=False, name=None):
        raise NotImplementedError


class Valuation(object):
    """
    Holds the computation state from inputs to outputs and intermediate results in between.
    """
    def __init__(self):
        self._shared = {}
        self._vars = {}
        self._events = {}

    @property
    def events(self):
        return self._events

    def add(self, name, value):
        if name in self._shared:
            raise KeyError('Shared variable "' + name + '" already present in valuation.')
        if name in self._vars:
            raise KeyError('Variable "' + name + '" already present in valuation. Use set to overwrite.')
        self._vars[name] = value

    def add_shared(self, name, value):
        if name in self._shared:
            raise KeyError('Shared variable "' + name + '" already present in valuation.')
        if name in self._vars:
            raise KeyError('Variable "' + name + '" already present in valuation. Use set to overwrite.')
        self._shared[name] = value

    def __contains__(self, item):
        return (item in self._vars) or (item in self._shared)

    def get(self, name):
        if name in self._shared:
            return self._shared[name]
        elif name in self._vars:
            return self._vars[name]
        else:
            raise KeyError('Variable "' + name + '" not found in this valuation.')

    def set(self, name, value):
        if name in self._shared:
            self._shared[name] = value
        elif name in self._vars:
            self._shared[name] = value
        else:
            raise KeyError('Variable "' + name + '" not found in this valuation.')

    def clear(self):
        self._vars.clear()

    def remove(self, name):
        del self._vars[name]
        del self._shared[name]


class Function(object):
    __metaclass__ = ABCMeta
    _ctr = 0

    def __init__(self, expressions=None, updates=None, name=None):
        self._name = name if name is not None else 'f' + str(Function._ctr)
        Function._ctr += 1

    def __call__(self, *args, **kwargs):
        """
        :param valuation:
        :type valuation: Valuation
        """
        assert 'valuation' in kwargs
        # TODO umotavanje parametara u valuation bi moglo ovde da se desi?

        return self.evaluate(kwargs['valuation'])

    @abstractmethod
    def evaluate(self, valuation):
        raise NotImplementedError
