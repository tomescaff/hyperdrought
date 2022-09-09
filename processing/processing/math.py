import numpy as np
from scipy.optimize import fmin
from scipy.stats import gamma

# maximum likelihood estimation -- gamma
def mle_gamma_2d(xarr, Tarr, init_params):

    def gamma_with_trend_shift(x, T, params):
        sigma0 = params[0]
        eta = params[1]
        alpha = params[2]
        sigma = sigma0*np.exp(alpha*T)
        y = gamma.pdf(x, eta, 0, sigma)
        return y

    def maxlkh(params, *args):
        xs = args[0]
        Ts = args[1]
        f = gamma_with_trend_shift
        logp = -sum([np.log(f(x,T, params)) for (x,T) in zip(xs, Ts)])
        return logp

    xopt = fmin(func=maxlkh, x0 = init_params, args=(xarr, Tarr))
    return xopt

def mle_gamma_2d_fast(xarr, Tarr, init_params):

    def gamma_with_trend_shift(x, T, sigma0, eta, alpha):
        return gamma.pdf(x, eta, 0, sigma0*np.exp(alpha*T))

    vfun = np.vectorize(gamma_with_trend_shift) 

    def maxlkh(params, *args):
        return -np.sum(np.log(vfun(*args, *params)))

    return fmin(func=maxlkh, x0 = init_params, args=(xarr, Tarr))


