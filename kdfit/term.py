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
    def __init__(self, name, x_kj, signals):
        super().__init__(name,[s.nev_param for s in signals])
        self.x_kj = cp.ascontiguousarray(cp.asarray(x_kj))
        self.signals = signals

    def calculate(self, inputs):
        scales = inputs[:len(self.signals)]
        print('Evaluate:',['%0.3f'%s for s in scales])
        if cp == np: # No GPU acceleration
            pdf_sk = np.asarray([n*s.eval_pdf_multi(self.x_kj) for n,s in zip(inputs,self.signals)])
            res = np.sum(scales) - np.sum(np.log(np.sum(pdf_sk,axis=0)))
        else:
            pdf_sk = cp.asarray([n*s.eval_pdf_multi(self.x_kj,get=False) for n,s in zip(inputs,self.signals)])
            res = np.sum(scales) - cp.sum(cp.log(cp.sum(pdf_sk,axis=0))).get()
        print('NLL:',res)
        return res
