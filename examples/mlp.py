"""Demonstrates MLP running on MNIST."""
import numpy as np
import shaderrider


class FullyConnectedLayer(object):
    def __init__(self, nin, nout):
        self.W = None   # TODO

class MLP(object):
    def __init__(self, nin, nhid, nout):
        self.layers = []

    def train(self, X, Y):
        pass

    def classify(self):
        pass
