"""Alexnet on cifar-10 example."""
import numpy as np
import pandas as pd
import pyopencl as cl
from pyopencl import array as clarray

import cPickle
import argparse
import timeit
import sys
sys.path.append('..')
from shaderrider import clplatf as pl
from shaderrider import expr
from shaderrider import operators as op
from shaderrider.utils import misc


class ConvLayer(object):
    def __init__(self, label, rng, img, filter_shape,
                 fstrides=(0, 0), zero_pad=(0, 0),
                 poolsize=(0, 0), W=None, b=None,
                 activation_fn=None):
        q = pl.qs[0]
        self.img = img
        # init W and b
        if W is None:
            fan_in = np.prod(filter_shape[1:])
            fan_out = filter_shape[0]*np.prod(filter_shape[2:])
            W_bound = np.sqrt(6./(fan_in+fan_out))
            nW = np.asarray(rng.uniform(low=-W_bound, high=W_bound,
                                        size=filter_shape), dtype=np.float32)

            self.W = clarray.to_device(q, nW)
        else:
            self.W = W
        if b is None:
            self.b = clarray.zeros(q, (filter_shape[0],), dtype=np.float32)
        else:
            self.b = b

        pf, ps = poolsize

        vW = expr.Variable('W'+label)
        vb = expr.Variable('b'+label)
        _out = op.Conv2d(self.img, vW, vb, strides=fstrides,
                         zero_padding=zero_pad)
        if pf > 0 and ps > 0:
            _out = op.MaxPool(_out, pf, ps)
        self.output = activation_fn(_out)
        self.params = [(vW.name, self.W), (vb.name, self.b)]


class FullyConnectedLayer(object):
    def __init__(self, label, rng, input, nin, nout, W=None, b=None,
                 activation_fn=op.Sigmoid):
        q = pl.qs[0]
        self.input = input
        if W is None:
            nw = np.asarray(rng.uniform(low=-np.sqrt(6./(nin+nout)),
                                        high=np.sqrt(6./(nin+nout)),
                                        size=(nin, nout)),
                            dtype=np.float32)
            if activation_fn == op.Sigmoid:
                nw *= 4
            W = clarray.to_device(q, nw)
        if b is None:
            b = clarray.zeros(q, (nout,), np.float32)

        self.W = W
        self.b = b

        vW = expr.Variable('W'+label)
        vb = expr.Variable('b'+label)
        lin_out = op.Add(op.Dot(self.input, vW), vb)
        self.output = lin_out if activation_fn is None else activation_fn(lin_out)
        self.params = [(vW.name, self.W), (vb.name, self.b)]


class SoftmaxLayer(object):
    def __init__(self, label, rng, input, n_in, n_out, W=None, b=None):
        q = pl.qs[0]
        self.input = input
        if W is None:
            self.W = clarray.zeros(q, (n_in, n_out), dtype=np.float32)
        else:
            self.W = W
        if b is None:
            self.b = clarray.zeros(q, (n_out,), dtype=np.float32)
        else:
            self.b = b

        vW = expr.Variable('W'+label)
        vb = expr.Variable('b'+label)

        self.p_y_given_x = op.Softmax(op.Add(op.Dot(self.input, vW), vb))
        self.y_pred = op.Argmax(self.p_y_given_x, axis=-1)

        self.params = [(vW.name, self.W), (vb.name, self.b)]


class Alexnet(object):
    def __init__(self, rng, n_out, params=None):
        X = expr.Variable('X')
        Y = expr.Variable('Y')

        self.conv1 = ConvLayer('_C1', rng, X, (32, 3, 5, 5),
                               fstrides=(1, 1), zero_pad=(2, 2),
                               poolsize=(2, 2), activation_fn=op.ReLU,
                               W=None if params is None else params['W_C1'],
                               b=None if params is None else params['b_C1'])
        self.conv2 = ConvLayer('_C2', rng, self.conv1.output, (32, 32, 5, 5),
                               fstrides=(1, 1), zero_pad=(2, 2),
                               poolsize=(2, 2), activation_fn=op.ReLU,
                               W=None if params is None else params['W_C2'],
                               b=None if params is None else params['b_C2'])
        self.conv3 = ConvLayer('_C3', rng, self.conv2.output, (64, 32, 5, 5),
                               fstrides=(1, 1), zero_pad=(2, 2),
                               poolsize=(2, 2), activation_fn=op.ReLU,
                               W=None if params is None else params['W_C3'],
                               b=None if params is None else params['b_C3'])
        nin = 64*4*4
        reshaped_conv_out = op.Reshape(self.conv3.output, (-1, nin))
        self.fc64 = FullyConnectedLayer('_F1', rng, reshaped_conv_out, nin, 64,
                                        activation_fn=op.ReLU,
                                        W=None if params is None else params['W_F1'],
                                        b=None if params is None else params['b_F1'])
        self.fc10 = FullyConnectedLayer('_F2', rng, self.fc64.output, 64, 10,
                                        activation_fn=op.ReLU,
                                        W=None if params is None else params['W_F2'],
                                        b=None if params is None else params['b_F2'])
        # self.do1 = op.Dropout(self.fc10.output)
        self.layer3 = SoftmaxLayer('_S1', rng, self.fc10.output, 10, n_out,
                                   W=None if params is None else params['W_S1'],
                                   b=None if params is None else params['b_S1'])
        self.cost = op.MeanSquaredErr(self.layer3.p_y_given_x, Y)
        self.error = op.Mean(op.NotEq(self.layer3.y_pred, Y))
        self.params = self.conv1.params + self.conv2.params + \
            self.conv3.params + self.fc64.params + self.fc10.params + \
            self.layer3.params
        self.prev_grad = None

    def train(self, X, Y, learning_rate=0.01, momentum=0.0):
        # self.do1.test = False
        val = pl.valuation()    # TODO Ovo uzrokuje trasfere
        val['X'] = X
        val['Y'] = Y
        for name, value in self.params:
            val[name] = value

        grad = self.cost.rev_grad(val)
        # print grad
        for name, value in self.params:
            # print 'updating', name
            # print 'shape:', value.shape, 'grad shape:', grad[name].shape
            if name.startswith('b_F') or name.startswith('b_S'):
                bgsum = misc.sum(pl.qs[0], grad[name], axis=0)
                value -= learning_rate*bgsum
            else:
                dv = learning_rate*grad[name]
                if self.prev_grad is not None and momentum > 0:
                    dv += momentum*self.prev_grad[name]
                value -= dv
        if momentum > 0:
            self.prev_grad = grad

    def test(self, X, Y):
        val = pl.valuation()
        val['X'] = X
        val['Y'] = Y
        for param, value in self.params:
            val[param] = value
        # self.do1.test = True
        err = self.error.evaluate(val)
        return err.get()


def load_net(q, netf):
    with open(netf, 'rb') as f:
        params = cPickle.load(f)
        for p in params:
            params[p] = clarray.to_device(q, params[p])
        return params
    raise ValueError('unable to load the specified net')


def save_net(nnet, netf):
    ps = []
    for param, value in nnet.params:
        ps.append((param, value.get()))
    if len(ps) == 0:
        raise ValueError('No params to save')
    with open(netf, 'wb') as f:
        cPickle.dump(dict(ps), f)


def load_cifar(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict


def main():
    pl.init_cl(1)
    q = pl.qs[0]

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--params', help='saved network parameters')
    argparser.add_argument('-v', '--validate', action='store_true',
                           help='use last batch as validation set')
    args = argparser.parse_args()

    net_params = None
    if args.params:
        net_params = load_net(q, args.params)

    rng = np.random.RandomState(1234)
    anet = Alexnet(rng, 10, params=net_params)

    # db1 = load_cifar('/home/petar/datasets/cifar-10-batches-py/data_batch_1')
    # db2 = load_cifar('/home/petar/datasets/cifar-10-batches-py/data_batch_2')
    # db3 = load_cifar('/home/petar/datasets/cifar-10-batches-py/data_batch_3')
    # db4 = load_cifar('/home/petar/datasets/cifar-10-batches-py/data_batch_4')
    # db5 = load_cifar('/home/petar/datasets/cifar-10-batches-py/data_batch_5')
    # tdb = load_cifar('/home/petar/datasets/cifar-10-batches-py/test_batch')
    # X1 = db1['data'].reshape(10000, 3, 32, 32).astype(np.float32)/255.0
    # Y1 = db1['labels']
    # trainY1 = pd.get_dummies(Y1).values.astype(np.float32)
    # X2 = db2['data'].reshape(10000, 3, 32, 32).astype(np.float32)/255.0
    # Y2 = db2['labels']
    # trainY2 = pd.get_dummies(Y2).values.astype(np.float32)
    # X3 = db3['data'].reshape(10000, 3, 32, 32).astype(np.float32)/255.0
    # Y3 = db3['labels']
    # trainY3 = pd.get_dummies(Y3).values.astype(np.float32)
    # X4 = db4['data'].reshape(10000, 3, 32, 32).astype(np.float32)/255.0
    # Y4 = db4['labels']
    # trainY4 = pd.get_dummies(Y4).values.astype(np.float32)
    # X5 = db5['data'].reshape(10000, 3, 32, 32).astype(np.float32)/255.0
    # Y5 = db5['labels']
    # trainY5 = pd.get_dummies(Y5).values.astype(np.float32)
    # validY = np.asarray(Y5, dtype=np.float32)
    #
    # tX = tdb['data'].reshape(-1, 3, 32, 32).astype(np.float32)/255.0
    # tY = np.asarray(tdb['labels'], dtype=np.float32)

    with open('/home/petar/datasets/cifar-10.pkl', 'rb') as f:
        [(X, Y), (testX, testY)] = cPickle.load(f)

    n_epochs = 10
    batch_size = 128
    n_train_batches = Y.shape[0]/batch_size
    n_valid_batches = 0
    if args.validate:
        n_train_batches = int(0.8*n_train_batches)
        n_valid_batches = int(0.2*Y.shape[0]/batch_size)
    n_test_batches = testY.shape[0]/batch_size

    print 'starting training...'
    start_time = timeit.default_timer()

    # TODO preload batches into GPU memory
    X_batches = []
    Y_batches = []
    X_validbs = []
    Y_validbs = []
    X_test = []
    Y_test = []
    for minibatch_index in xrange(n_train_batches):
        # TODO ovo treba da se kopira u clarray.Array (trebace i command queue za to)
        X_batches.append(clarray.to_device(pl.qs[0],
                         X[minibatch_index*batch_size:(minibatch_index+1)*batch_size]))
        Y_batches.append(clarray.to_device(pl.qs[0],
                         pd.get_dummies(Y[minibatch_index*batch_size:(minibatch_index+1)*batch_size]).values.astype(np.float32)))
    for minibatch_index in xrange(n_train_batches, n_train_batches+n_valid_batches):
        X_validbs.append(clarray.to_device(pl.qs[0],
                         X[minibatch_index*batch_size:(minibatch_index+1)*batch_size]))
        Y_validbs.append(clarray.to_device(pl.qs[0],
                         Y[minibatch_index*batch_size:(minibatch_index+1)*batch_size]))
    for minibatch_index in xrange(n_test_batches):
        X_test.append(clarray.to_device(pl.qs[0],
                      testX[minibatch_index*batch_size:(minibatch_index+1)*batch_size]))
        Y_test.append(clarray.to_device(pl.qs[0],
                      testY[minibatch_index*batch_size:(minibatch_index+1)*batch_size]))

    epoch = 0
    lrn_rate = 0.01
    momentum = 0.0
    while epoch < n_epochs:
        epoch += 1
        print 'epoch:', epoch, 'of', n_epochs, 'lr:', lrn_rate, 'm:', momentum
        for mbi in xrange(n_train_batches):
            print '\r>training batch', mbi, 'of', n_train_batches,
            sys.stdout.flush()
            # anet.train(X1[mbi*batch_size:(mbi+1)*batch_size, :],
            #            trainY1[mbi*batch_size:(mbi+1)*batch_size, :],
            #            lrn_rate, momentum)
            # anet.train(X2[mbi*batch_size:(mbi+1)*batch_size, :],
            #            trainY2[mbi*batch_size:(mbi+1)*batch_size, :],
            #            lrn_rate, momentum)
            # anet.train(X3[mbi*batch_size:(mbi+1)*batch_size, :],
            #            trainY3[mbi*batch_size:(mbi+1)*batch_size, :],
            #            lrn_rate, momentum)
            # anet.train(X4[mbi*batch_size:(mbi+1)*batch_size, :],
            #            trainY4[mbi*batch_size:(mbi+1)*batch_size, :],
            #            lrn_rate, momentum)
            # if not args.validate:
            #     anet.train(X5[mbi*batch_size:(mbi+1)*batch_size, :],
            #                trainY5[mbi*batch_size:(mbi+1)*batch_size, :],
            #                lrn_rate, momentum)
            anet.train(X_batches[mbi], Y_batches[mbi], lrn_rate, momentum)
            if args.validate and mbi % 65 == 0:
                # verr = np.mean([float(anet.test(X5[vbi*batch_size:(vbi+1)*batch_size],
                #                                 validY[vbi*batch_size:(vbi+1)*batch_size]))
                #                 for vbi in range(n_valid_batches)])
                verr = np.mean([float(anet.test(X_validbs[vbi],
                                                Y_validbs[vbi]))
                                for vbi in range(n_valid_batches)])
                print '\rvalidation error:', verr
        if epoch % 8 == 0 and epoch > 0:
            lrn_rate /= 10.0
            # momentum += 0.1
        print
        print '='*70
        # print '>>final wc:\n', anet.conv1.params[0][1]
    print 'test error:',
    es = []
    for mbi in xrange(n_test_batches):
        er = anet.test(X_test[mbi], Y_test[mbi])

        # print 'test batch', mbi, 'error:', er
        es.append(er)
    print np.mean([float(e) for e in es])
    end_time = timeit.default_timer()
    print 'ran for %.2fm' % ((end_time - start_time)/60.)
    save_net(anet, 'alexnet_cifar.pkl')


if __name__ == '__main__':
    main()
