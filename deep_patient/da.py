from theano.tensor.shared_randomstreams import RandomStreams
from sklearn import preprocessing as preproc
from scipy import sparse
import random
import numpy as np
import theano.tensor as T
import theano
import timeit

max_sample = 200000


class DA(object):

    """
    General purpose Denoising Autoencoder
    """

    def __init__(self, nvisible, nhidden=50, param={}, layer=1):
        rng = np.random.RandomState()
        self.theano_rng = RandomStreams(rng.randint(2 ** 30))

        try:
            self.layer = int(layer)
        except Exception:
            self.layer = 1

        try:
            self.nv = int(nvisible)
        except Exception:
            print 'ERROR: visible unit no. not valid'
            return

        try:
            self.nh = int(nhidden)
        except Exception:
            print 'ERROR: hidden unit no. not valid'
            return

        # hidden units
        self.w = self._init_w(rng)
        self.b = self._init_b()

        # visible units (prime)
        self.wp = self.w.T
        self.bp = self._init_bp()

        self.param = [self.w, self.b, self.bp]

        self._init_train_param(param)

        self.normalizer = preproc.MinMaxScaler()

        self._log()

        return

    def train(self, data):
        if data.shape[1] != self.nv:
            print 'ERROR: no. of data features not agreeing with the' \
                'no. of visible units'
            return

        print 'training: DA [layer: %d]' % self.layer

        try:
            data = data.toarray()
        except Exception:
            pass

        if data.shape[0] > max_sample:
            idx = [i for i in xrange(data.shape[0])]
            random.shuffle(idx)
            idx = idx[:max_sample]
            data = data[idx, :]
            print 'resized data: using %d samples' % (data.shape[0])

        print '(*) preprocessing: normalize features'
        data = self.normalizer.fit_transform(data)

        dt = theano.shared(value=data.astype(
            theano.config.floatX), borrow=True)

        nbatch = dt.get_value().shape[0] / self.batch_size

        idx = T.lscalar()
        x = T.matrix(name='dt')

        cost, update = self._compute_cost_update(x)

        train_da = theano.function(
            [idx],
            cost,
            updates=update,
            givens={x: dt[idx * self.batch_size: (idx + 1) * self.batch_size]},
        )

        start_time = timeit.default_timer()

        pcost = 0.0
        for epc in xrange(self.epochs):
            c = []

            for bidx in xrange(nbatch):
                c.append(train_da(bidx))

            ccost = np.mean(c)
            print '(*) epoch %d, cost %.3f' % (epc + 1, ccost)

            if abs(ccost - pcost) < 10e-1:
                break
            else:
                pcost = ccost

        end_time = timeit.default_timer()

        print '(*) training time: %.2f sec.' % (end_time - start_time)

        return

    def apply(self, data):
        try:
            data = data.toarray()
        except Exception:
            pass
        data = self.normalizer.transform(data)

        dt = theano.shared(data, borrow=True)
        hrepr = self._hidden_representation(dt)

        return hrepr.eval()

    # private functions

    def _hidden_representation(self, x):
        """
        Compute the values of the hidden layer given the data
        """
        return T.nnet.sigmoid(T.dot(x, self.w) + self.b)

    def _reconstructed_input(self, h):
        """
        Compute the reconstrcuted data given the hidden representation
        """
        return T.nnet.sigmoid(T.dot(h, self.wp) + self.bp)

    def _corrupted_input(self, x):
        """
        Compute the corrupted input (masking noise)
        """
        return self.theano_rng.binomial(size=x.shape,
                                        n=1,
                                        p=1 - self.corrupt_lvl,
                                        dtype=theano.config.floatX) * x

    def _compute_cost_update(self, x):
        """
                Compute the cost update for one training step
        """
        tilde_x = self._corrupted_input(x)
        y = self._hidden_representation(tilde_x)
        z = self._reconstructed_input(y)

        # compute the cost of the minibatch
        gcst = - T.sum(x * T.log(z) + (1 - x) * T.log(1 - z), axis=1)

        cost = T.mean(gcst)

        # compute the gradien of the cost
        gparam = T.grad(cost, self.param)

        # generate the list of updates
        update = [(p, p - self.learn_rate * g)
                  for p, g in zip(self.param, gparam)]

        return (cost, update)

    def _init_w(self, rng):
        """
        Initialize model parameters
        """
        init_w = np.asarray(rng.uniform(
            low=-4 * np.sqrt(6. / (self.nh + self.nv)),
            high=4 * np.sqrt(6. / (self.nh + self.nv)),
            size=(self.nv, self.nh)
        ),
            dtype=theano.config.floatX
        )
        return theano.shared(value=init_w, name='w', borrow=True)

    def _init_b(self):
        return theano.shared(value=np.zeros(self.nh,
                                            dtype=theano.config.floatX),
                             name='b', borrow=True)

    def _init_bp(self):
        return theano.shared(value=np.zeros(self.nv,
                                            dtype=theano.config.floatX),
                             name='bp', borrow=True)

    def _init_train_param(self, param):
        try:
            self.corrupt_lvl = float(param['corrupt_lvl'])
        except Exception:
            self.corrupt_lvl = 0.1

        try:
            self.learn_rate = float(param['learn_rate'])
        except Exception:
            self.learn_rate = 0.1

        try:
            self.batch_size = int(param['batch_size'])
        except Exception:
            self.batch_size = 5

        try:
            self.epochs = int(param['epochs'])
        except Exception:
            self.epochs = 3

        return

    def _shuffle(self, x):
        """
        Shuffle data matrix
        """
        if not sparse.isspmatrix_csr(x):
            x = sparse.csr_matrix(x)
        idx = np.arange(x.shape[0])
        np.random.shuffle(idx)
        return x[idx, :]

    def _log(self):
        """
        Print model details
        """
        print 'initialized: DA [layer: %d]' % self.layer
        print '(*) no. of visible units: %d' % self.nv
        print '(*) no. of hidden units: %d' % self.nh
        print '(*) data corruption level: %.2f' % self.corrupt_lvl
        print '(*) learning rate: %.2f' % self.learn_rate
        print '(*) batch size: %d' % self.batch_size
        print '(*) no. of epochs: %d' % self.epochs
        print ''
        return
