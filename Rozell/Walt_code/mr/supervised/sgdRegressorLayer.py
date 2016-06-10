
from mr.modelBase import SklearnModelBase

import numpy as np
from sklearn.linear_model import SGDRegressor


class SGDRegressorLayer(SklearnModelBase):
    UNSUPERVISED = False
    PARAMS = SklearnModelBase.PARAMS + [ 'eta0', 'power_t' ]

    def __init__(self, eta0 = 0.01, power_t = 0.1, **kwargs):
        kwargs['eta0'] = eta0
        kwargs['power_t'] = power_t
        super(SGDRegressorLayer, self).__init__(**kwargs)

    def _init(self, nInputs, nOutputs):
        self._layer = [ SGDRegressor(power_t = 0.1, n_iter = 1)
                for _ in range(nOutputs) ]
        [ self._layer[i]._allocate_parameter_mem(1, nInputs)
                for i in range(nOutputs) ]

    def _partial_fit(self, x, y):
        for i in range(self.nOutputs):
            self._layer[i].partial_fit([ x ], [ y[i] ])

    def _predict(self, x, y):
        for i in range(self.nOutputs):
            y[i] = self._layer[i].predict(x)[0]

    def _reconstruct(self, y, r):
        # y[i] = sum_j(layer[i].coef_[j] * r[j]) + layer[i].intercept_
        # sum_j(layer[i].coef_[j] * r[j]) = (y[i] - layer[i].intercept_)
        # Ma = b, M = (layer.coef_), a = r[j], b = y[i] - layer[i].intercept_
        rn = np.asmatrix(r).transpose()
        yn = np.asmatrix(y).transpose()
        for i in range(self.nOutputs):
            yn[i,0] -= self._layer[i].intercept_
        ma = np.zeros((self.nOutputs, self.nInputs), dtype = float)
        for i, layer in enumerate(self._layer):
            maxCoef = layer.coef_[layer.coef_ > 0].sum() / max(1, (layer.coef_ > 0).sum())
            for j in range(self.nInputs):
                ma[i, j] = layer.coef_[j] if layer.coef_[j] >= maxCoef else 0
        rn[:], _residuals, _rank, _s = np.linalg.lstsq(ma, yn)
        #print("{} -> {}".format(y, rn))
        #print("{}...".format(mar * rn))
