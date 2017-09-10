"""Theano test for comparison with shaderrider."""

import numpy as np
import pandas as pd
import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d, sigmoid, relu

import cPickle
import argparse
import sys


class ConvLayer(object):
    def __init__(self, label, rng, img, filter_shape,
                 fstrides=(1, 1), zero_pad=(0, 0),
                 poolsize=(0, 0), W=None, b=None,
                 activation_fn=None):
        if W is None:
            fan_in = np.prod(filter_shape[1:])
            fan_out = filter_shape[0]*np.prod(filter_shape[2:])
            W_bound = np.sqrt(6./(fan_in+fan_out))
            nW = np.asarray(rng.uniform(low=-W_bound, high=W_bound,
                                        size=filter_shape),
                            dtype=theano.config.floatX)
            self.W = theano.shared(nW, borrow=True)
        else:
            self.W = W

        if b is None:
            self.b = theano.shared(np.zeros((filter_shape[0],),
                                            dtype=theano.config.floatX),
                                   borrow=True)
        else:
            self.b = b

        conv_out = conv2d(input=img, filters=self.W, filter_shape=filter_shape,
                          border_mode=zero_pad, subsample=fstrides)

        pooled_out = pool.pool_2d(input=conv_out, ds=poolsize)  # ignore_border
        if activation_fn:
            self.output = activation_fn(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        else:
            self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.W, self.b]
        self.input = img


class FullyConnectedLayer(object):
    def __init__(self, label, rng, input, nin, nout, W=None, b=None,
                 activation_fn=sigmoid):
        self.input = input
        if W is None:
            nw = np.asarray(rng.uniform(low=-np.sqrt(6./(nin+nout)),
                                        high=np.sqrt(6./(nin+nout)),
                                        size=(nin, nout)),
                            dtype=theano.config.floatX)
            if activation_fn == theano.tensor.nnet.sigmoid:
                nw *= 4
            W = theano.shared(value=nw, name='W', borrow=True)
        if b is None:
            b = theano.shared(np.zeros((nout,), dtype=theano.config.floatX),
                              name='b', borrow=True)
        self.W = W
        self.b = b

        lin_out = T.dot(input, self.W) + self.b
        self.output = lin_out if activation_fn is None else activation_fn(lin_out)
        self.params = [self.W, self.b]


class SoftmaxLayer(object):
    def __init__(self, label, rng, input, n_in, n_out, W=None, b=None):
        if W is None:
            self.W = theano.shared(
                value=np.zeros((n_in, n_out), dtype=theano.config.floatX),
                name='W', borrow=True)
        else:
            self.W = W
        if b is None:
            self.b = theano.shared(
                value=np.zeros((n_out,), dtype=theano.config.floatX),
                name='b', borrow=True)
        else:
            self.b = b
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]
        self.input = input

    def negative_log_likelihood(self, y):
        # TODO da li treba nll ili nesto drugo?
        # mislim da sam ovamo koristio MeanSquaredErr
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def mean_squared_error(self, y):
        # raise NotImplemented()
        return T.mean((self.p_y_given_x[T.arange(y.shape[0]), y] - y)**2)


class Alexnet(object):
    def __init__(self, rng, n_out, params=None):
        self.X = T.tensor4('X')
        self.Y = T.ivector('Y')

        self.conv1 = ConvLayer('_C1', rng, self.X, (32, 3, 5, 5),
                               fstrides=(1, 1), zero_pad=(2, 2),
                               poolsize=(2, 2), activation_fn=relu,
                               W=None if params is None else params['W_C1'],
                               b=None if params is None else params['b_C1'])
        self.conv2 = ConvLayer('_C2', rng, self.conv1.output, (32, 32, 5, 5),
                               fstrides=(1, 1), zero_pad=(2, 2),
                               poolsize=(2, 2), activation_fn=relu,
                               W=None if params is None else params['W_C2'],
                               b=None if params is None else params['b_C2'])
        self.conv3 = ConvLayer('_C3', rng, self.conv2.output, (64, 32, 5, 5),
                               fstrides=(1, 1), zero_pad=(2, 2),
                               poolsize=(2, 2), activation_fn=relu,
                               W=None if params is None else params['W_C3'],
                               b=None if params is None else params['b_C3'])
        nin = 64*4*4
        # NOTE provereno, reshape se ponasa kako treba.
        reshaped_conv_out = T.reshape(self.conv3.output, (-1, nin))
        self.fc64 = FullyConnectedLayer('_F1', rng, reshaped_conv_out, nin, 64,
                                        activation_fn=relu,
                                        W=None if params is None else params['W_F1'],
                                        b=None if params is None else params['b_F1'])
        self.fc10 = FullyConnectedLayer('_F2', rng, self.fc64.output, 64, 10,
                                        activation_fn=relu,
                                        W=None if params is None else params['W_F2'],
                                        b=None if params is None else params['b_F2'])
        self.layer3 = SoftmaxLayer('_S1', rng, self.fc10.output, 10, n_out,
                                   W=None if params is None else params['W_S1'],
                                   b=None if params is None else params['b_S1'])
        # NOTE koristim mean squared error jer neg.log likelihood ne mogu sad
        # da implementiram u shaderrideru zbog ogranicenog indeksiranja?
        self.cost = self.layer3.mean_squared_error(self.Y)
        self.error = T.mean(T.neq(self.layer3.y_pred, self.Y))
        self.params = self.conv1.params + self.conv2.params + \
            self.conv3.params + self.fc64.params + self.fc10.params + \
            self.layer3.params
        self.prev_grad = None
        self.grads = T.grad(self.cost, self.params)
        # TODO ovde kreirati theano funkcije za trening, validaciju i test...
        self.testf = theano.function([self.X, self.Y], self.error)

    def get_trainf(self, learning_rate=0.01):
        # NOTE bez momentuma jer ni u shaderrider verziji realno nije koriscen
        # X = T.tensor4('X')
        # Y = T.ivector('Y')
        updates = [(param_i, param_i - learning_rate*grad_i)
                   for param_i, grad_i in zip(self.params, self.grads)]
        trainf = theano.function([self.X, self.Y], self.cost, updates=updates)
        return trainf

    def test(self, X, Y):
        return self.testf(X, Y)


def load_net(q, netf):
    with open(netf, 'rb') as f:
        params = cPickle.load(f)
        return params
    raise ValueError('unable to load the specified net')


def save_net(nnet, netf):
    ps = []
    for param in nnet.params:
        ps.append((param.name, param.get_value()))  # TODO borrow maybe?
    with open(netf, 'wb') as f:
        cPickle.dump(dict(ps), f)


def load_cifar(file):
    fo = open(file, 'rb')
    dictionary = cPickle.load(fo)
    fo.close()
    return dictionary


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--params', help='saved network parameters')
    argparser.add_argument('-v', '--validate', action='store_true',
                           help='use the last batch as validation set')
    args = argparser.parse_args()

    net_params = None
    if args.params:
        net_params = load_net(args.params)
    rng = np.random.RandomState(1234)
    anet = Alexnet(rng, 10, params=net_params)

    db1 = load_cifar('/home/petar/datasets/cifar-10-batches-py/data_batch_1')
    db2 = load_cifar('/home/petar/datasets/cifar-10-batches-py/data_batch_2')
    db3 = load_cifar('/home/petar/datasets/cifar-10-batches-py/data_batch_3')
    db4 = load_cifar('/home/petar/datasets/cifar-10-batches-py/data_batch_4')
    db5 = load_cifar('/home/petar/datasets/cifar-10-batches-py/data_batch_5')
    tdb = load_cifar('/home/petar/datasets/cifar-10-batches-py/test_batch')
    X1 = db1['data'].reshape(10000, 3, 32, 32).astype(np.float32)/255.0
    Y1 = db1['labels']
    trainY1 = np.asarray(Y1, dtype=np.int32)    # pd.get_dummies(Y1).values.astype(np.int32)
    X2 = db2['data'].reshape(10000, 3, 32, 32).astype(np.float32)/255.0
    Y2 = db2['labels']
    trainY2 = np.asarray(Y2, dtype=np.int32)    # pd.get_dummies(Y2).values.astype(np.int32)
    X3 = db3['data'].reshape(10000, 3, 32, 32).astype(np.float32)/255.0
    Y3 = db3['labels']
    trainY3 = np.asarray(Y3, dtype=np.int32)    # pd.get_dummies(Y3).values.astype(np.int32)
    X4 = db4['data'].reshape(10000, 3, 32, 32).astype(np.float32)/255.0
    Y4 = db4['labels']
    trainY4 = np.asarray(Y4, dtype=np.int32)    # pd.get_dummies(Y4).values.astype(np.int32)
    X5 = db5['data'].reshape(10000, 3, 32, 32).astype(np.float32)/255.0
    Y5 = db5['labels']
    trainY5 = np.asarray(Y5, dtype=np.int32)    # pd.get_dummies(Y5).values.astype(np.int32)
    validY = np.asarray(Y5, dtype=np.int32)

    tX = tdb['data'].reshape(-1, 3, 32, 32).astype(np.float32)/255.0
    tY = np.asarray(tdb['labels'], dtype=np.int32)

    n_epochs = 10
    batch_size = 128
    n_train_batches = trainY1.shape[0]/batch_size
    n_valid_batches = trainY5.shape[0]/batch_size
    n_test_batches = tY.shape[0]/batch_size

    epoch = 0
    lrn_rate = 0.01
    momentum = 0.0
    trainfn = anet.get_trainf(lrn_rate)
    while epoch < n_epochs:
        epoch += 1
        print 'epoch:', epoch, 'of', n_epochs, 'lr:', lrn_rate, 'm:', momentum
        for mbi in xrange(n_train_batches):
            print '\r>training batch', mbi, 'of', n_train_batches,
            sys.stdout.flush()
            trainfn(X1[mbi*batch_size:(mbi+1)*batch_size, :],
                    trainY1[mbi*batch_size:(mbi+1)*batch_size])
            trainfn(X2[mbi*batch_size:(mbi+1)*batch_size, :],
                    trainY2[mbi*batch_size:(mbi+1)*batch_size])
            trainfn(X3[mbi*batch_size:(mbi+1)*batch_size, :],
                    trainY3[mbi*batch_size:(mbi+1)*batch_size])
            trainfn(X4[mbi*batch_size:(mbi+1)*batch_size, :],
                    trainY4[mbi*batch_size:(mbi+1)*batch_size])
            if not args.validate:
                trainfn(X5[mbi*batch_size:(mbi+1)*batch_size, :],
                        trainY5[mbi*batch_size:(mbi+1)*batch_size])
            if args.validate and mbi % 13 == 0:
                verr = np.mean([anet.test(X5[vbi*batch_size:(vbi+1)*batch_size],
                                          validY[vbi*batch_size:(vbi+1)*batch_size])
                                for vbi in range(n_valid_batches)])
                print '\rvalidation error:', verr
        if epoch % 8 == 0 and epoch > 0:
            lrn_rate /= 10.0
            # TODO rekreiraj trainf sa novim learning rateom
        print
        print '='*70
    print 'test error:',
    es = []
    for mbi in xrange(n_test_batches):
        er = anet.test(tX[mbi*batch_size:(mbi+1)*batch_size],
                       tY[mbi*batch_size:(mbi+1)*batch_size])
        es.append(er)
    print np.mean([float(e) for e in es])
    save_net(anet, 'alexnet_cifar.pkl')


if __name__ == '__main__':
    main()
