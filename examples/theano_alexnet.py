"""reimplementacija alexneta preko lenet5 tutorijala za theano."""

import os
import sys
import timeit
import cPickle

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d, relu

from logistic_sgd import LogisticRegression
from mlp import HiddenLayer


def load_data(dataset=''):
    df = open(dataset, 'rb')
    datasets = cPickle.load(df)
    df.close()
    train_set_x, train_set_y = shared_dataset(datasets[0])
    valid_set_x, valid_set_y = shared_dataset(datasets[1])
    test_set_x, test_set_y = shared_dataset(datasets[2])
    return [(train_set_x, train_set_y),
            (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]


def shared_dataset(data_xy, borrow=True):
    """
    Function that loads the dataset into shared variables.

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2),
                 zero_pad=None):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        if zero_pad is None:
            conv_out = conv2d(
                input=input,
                filters=self.W,
                filter_shape=filter_shape,
                input_shape=image_shape
            )
        else:
            conv_out = conv2d(
                input=input,
                filters=self.W,
                filter_shape=filter_shape,
                input_shape=image_shape,
                border_mode=zero_pad
            )

        # pool each feature map individually, using maxpooling
        pooled_out = pool.pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=False
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        #self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output = relu(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        # TODO ReLU umesto tanh?

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input


def evaluate_lenet5(learning_rate=0.1, n_epochs=200,
                    dataset='mnist.pkl.gz', batch_size=500):
    nkerns = [32, 32, 64]
    #rng = numpy.random.RandomState(1234)
    rng = numpy.random.RandomState()

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    layer0_input = x.reshape((batch_size, 3, 32, 32))

    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(nkerns[0], 3, 5, 5),
        poolsize=(2, 2),
        zero_pad=2
    )

    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 16, 16),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2),
        zero_pad=2
    )

    # TODO raspisati dimenzije slika i filtera kroz layere
    layer2 = LeNetConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size, nkerns[1], 8, 8),
        filter_shape=(nkerns[2], nkerns[1], 5, 5),
        poolsize=(2, 2),
        zero_pad=2
    )

    layer3_input = layer2.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer3 = HiddenLayer(
        rng,
        input=layer3_input,
        n_in=nkerns[2] * 4 * 4,
        n_out=64,
        activation=relu
    )

    layer4 = HiddenLayer(
        rng,
        input=layer3.output,
        n_in=64,
        n_out=10,
        activation=relu
    )

    # classify the values of the fully-connected sigmoidal layer
    layer5 = LogisticRegression(input=layer4.output, n_in=10, n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer5.negative_log_likelihood(y)
    # cost = layer5.mean_squared_error(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer5.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer5.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer5.params + layer4.params + layer3.params + layer2.params + \
        layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    updates_slow = [
        (param_i, param_i - learning_rate/10 * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    updates_slower = [
        (param_i, param_i - learning_rate/100 * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    train_model_slow = theano.function(
        [index],
        cost,
        updates=updates_slow,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }

    )

    train_model_slower = theano.function(
        [index],
        cost,
        updates=updates_slower,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }

    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
            if epoch < n_epochs*0.3:
                cost_ij = train_model(minibatch_index)
            elif n_epochs*0.3 < epoch < n_epochs*0.6:
                cost_ij = train_model_slow(minibatch_index)
            else:
                cost_ij = train_model_slower(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print 'epoch %i, minibatch %i/%i, validation error %f %%' % \
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.)

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print ('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') % \
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.)

            # if patience <= iter:
            #     done_looping = True
            #     break

    end_time = timeit.default_timer()
    print 'Optimization complete.'
    print 'Best validation score of %f %% obtained at iteration %i, ' \
          'with test performance %f %%' % \
          (best_validation_loss * 100., best_iter + 1, test_score * 100.)
    print >>sys.stderr, ('The code for file ' +
                         os.path.split(__file__)[1] +
                         ' ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
    lr = 0.01
    nep = 10
    batchs = 128
    print 'rand_init relu relu relu relu relu softmax'
    print 'Learning rate: %f, n epochs: %d, batch size: %d' % (lr, nep, batchs)
    evaluate_lenet5(learning_rate=lr, n_epochs=nep, batch_size=batchs,
                    dataset='/home/petarj/cifar_compact2.pkl')


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)
