import numpy as np
try:
    import cupy as cp
except:
    cp = np # Use numpy to emulate cupy on CPU
from .calculate import Calculation

class Parameter(Calculation):
    '''
    Represents an input to a System
    
    guess is the initial or fixed value of the Parameter
    constant sets whether the parameter is fixed (True) or floated (False)
    '''
    def __init__(self,name,guess=None,constant=True):
        super().__init__(name,[],constant=constant)
        self.name = name
        self.guess = guess
    def calculate(self,inputs):
        if self.constant:
            return self.guess
        
class Sum(Calculation):
    '''
    A Calculation that adds a series of input Calculations
    '''
    def __init__(self,name,*inputs):
        super().__init__(name,inputs)
        self.name = name
    def calculate(self,inputs):
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
        n_evs = [s.nev_param for s in signals]
        signal_systs = [s.systematics for s in signals]
        systs = [syst for systs in signal_systs for syst in systs]
        super().__init__(name,n_evs+systs)
        i = len(n_evs)
        self.signal_syst_bounds = [(i,i:=(i+len(systs))) for systs in signal_systs]
        self.x_kj = cp.ascontiguousarray(cp.asarray(x_kj))
        self.signals = signals
        self.observables = observables

    def calculate(self, inputs):
        n_evs = inputs[:len(self.signals)]
        signal_systs = [inputs[a:b] for a,b in self.signal_syst_bounds]
        print('Evaluate:',', '.join(['%0.3f'%s for s in n_evs]))
        if cp == np: # No GPU acceleration
            pdf_sk = np.asarray([n*s.eval_pdf_multi(self.x_kj,systs=systs) for n,s,systs in zip(n_evs,self.signals,signal_systs)])
            res = np.sum(n_evs) - np.sum(np.log(np.sum(pdf_sk,axis=0)))
        else:
            pdf_sk = cp.asarray([n*s.eval_pdf_multi(self.x_kj,systs=systs,get=False) for n,s,systs in zip(n_evs,self.signals,signal_systs)])
            res = np.sum(n_evs) - cp.sum(cp.log(cp.sum(pdf_sk,axis=0))).get()
        print('NLL:',res)
        return res
