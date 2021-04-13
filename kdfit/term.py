import numpy as np
try:
    import cupy as cp
except:
    cp = np # Use numpy to emulate cupy on CPU
from .calculate import Calculation

class Parameter(Calculation):
    '''
    Represents an input to a System
    
    value is the initial or fixed value of the Parameter
    constant sets whether the parameter is fixed (True) or floated (False)
    '''
    def __init__(self, name, value=None, constant=True):
        super().__init__(name, [], constant=constant)
        self.name = name
        self.value = value
    def calculate(self, inputs, verbose=False):
        if self.constant:
            return self.value
        
class Sum(Calculation):
    '''
    A Calculation that adds a series of input Calculations
    '''
    def __init__(self, name, *inputs):
        super().__init__(name, inputs)
        self.name = name
    def calculate(self, inputs, verbose=False):
        return np.sum(inputs)
    
class UnbinnedNegativeLogLikelihoodFunction(Calculation):
    '''
    Calculates the negative log likelihood of data x_kj with a PDF given by
    the sum of the PDFs from some scaled signals. The signal scales, or number 
    of events for that signal, are the inputs to this calculation. Also 
    calculates the Poisson likelihood of observing k events given a mean event 
    rate that is the sum of the signal scales. The final result is the negative
    logarithm of the product of the data likelihood and Poisson likelihood that
    omits any terms that are constant as a function of input scales.
    '''
    def __init__(self, name, x_kj, signals, observables):
        self.x_kj = cp.ascontiguousarray(cp.asarray(x_kj))
        self.signals = signals
        self.observables = observables
        n_evs = [s.nev_param for s in signals]
        super().__init__(name,n_evs+signals)

    def calculate(self, inputs, verbose=False):
        n_evs = inputs[:len(self.signals)]
        signal_systs = inputs[len(self.signals):]
        if verbose:
            print('Evaluate:',', '.join(['%0.3f'%s for s in n_evs]))
        if cp == np: # No GPU acceleration
            pdf_sk = np.asarray([n*s.eval_pdf_multi(self.x_kj,systs=systs) for n,s,systs in zip(n_evs,self.signals,signal_systs)])
            res = np.sum(n_evs) - np.sum(np.log(np.sum(pdf_sk,axis=0)))
        else:
            pdf_sk = cp.asarray([n*s.eval_pdf_multi(self.x_kj,systs=systs,get=False) for n,s,systs in zip(n_evs,self.signals,signal_systs)])
            res = np.sum(n_evs) - cp.sum(cp.log(cp.sum(pdf_sk,axis=0))).get()
        if verbose:
            print('NLL:',res)
        return res
