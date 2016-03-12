import abc


class GraphValidation(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def validate(self, graph):
        """
        Checks the expression graph for any kind of problems (syntactic or semantic)

        If the check fails, the GraphValidationError is raised.

        :param graph:
        :rtype: boolean
        """
        raise NotImplementedError
