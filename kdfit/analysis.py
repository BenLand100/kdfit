#  Copyright 2021 by Benjamin J. Land (a.k.a. BenLand100)
#
#  This file is part of kdfit.
#
#  kdfit is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  kdfit is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with kdfit.  If not, see <https://www.gnu.org/licenses/>.

from .observables import Observables
from .calculate import Parameter,System
from .term import Sum

import scipy.optimize as opt
from functools import partial
    
class Analysis:
    def __init__(self):
        self.parameters = {}
        self.signals = {}
        self.observables = {}
        
    def add_parameter(self,name,*args,**kwargs):
        if name in self.parameters:
            raise Exception('Duplicate name: '+name)
        param = Parameter(name,*args,**kwargs)
        self.parameters[name] = param
        return param
        
    def get_parameter(self,name):
        return self.parameters[name]
        
    def add_observables(self,name,*args,**kwargs):
        if name in self.observables:
            raise Exception('Duplicate name: '+name)
        obs = Observables(name,self,*args,**kwargs)
        self.observables[obs.name] = obs
        return obs
        
    def load_mc(self,mc_loaders):
        for obs,sig_map in mc_loaders.items():
            if type(obs) is str:
                obs = self.observables[obs]
            for sig,loader in sig_map.items():
                if type(sig) is str:
                    sig = obs.signals[sig]
                sig.mc_param.link(loader)
                
    def load_data(self,data_loaders):
        for obs,loader in data_loaders.items():
            if type(obs) is str:
                obs = self.observables[obs]
            obs.data_param.link(loader)
            
    def create_likelihood(self,verbose=False):
        '''
        Builds a calculation system that computes the sum of the log likelihood
        from all Observables in this Analysis. This *must* be called again if
        the topology of the calculation changes (new objects), but should 
        neither be called to adjust whether Parameters are floated or fixed, nor 
        if their values changed (call update_likelihood instead).
        '''
        self._terms = []
        for name,obs in self.observables.items():
            nllfn = obs.get_likelihood()
            self._terms.append(nllfn)
        self._outputs = [Sum('Total_Likelihood',*self._terms)]
        if verbose:
            print('Ouput Values:',self._outputs)
        self._system = System(self._outputs,verbose=verbose)
        self.update_likelihood(verbose=verbose)
        
    def update_likelihood(self,verbose=False):
        '''
        If Parameters are changed from fixed to floated, call this method before
        evaluating the likelihood.
        '''
        self._floated,self._fixed = self._system.classify_inputs()
        if verbose:
            print('Floated Parameters:',self._floated)
            print('Fixed Parameters:',self._fixed)
        
    def __call__(self,floated_params=None,verbose=False,show_steps=False):
        '''
        Performs the current log likelihood calculation and returns the result.
        
        floated_params should be None to use parameter values, or values for all
        floated parameters can be given in the order of self._floated_params to,
        for example, evaluate the likelihood during minimization.
        '''
        if floated_params is None:
            floated_params = [p.value for p in self._floated]
        if show_steps:
            print('Evaluating at: ',', '.join(['%0.2f'%x for x in floated_params]))
        outputs = self._system.calculate(floated_params,verbose=verbose)
        if show_steps:
            print('Result: ',outputs[0])
        return outputs[0]
        
    def minimize(self,verbose=False,show_steps=False,**kwargs):
        '''
        Will optimize the floated parameters in the current log likelihood 
        calculation to minimize the log likelihood.
        
        Keyword arguments are passed to scipy.optimize.minimize
        '''
        initial = [g if (g:=p.value) is not None else 1.0 for p in self._floated]
        minimum = opt.minimize(partial(self,show_steps=show_steps,verbose=verbose),x0=initial,**kwargs)
        minimum.params = {p:v for p,v in zip(self._floated,minimum.x)}
        return minimum
        
    def _delta_nll_profile(self,m,p,x,ci_delta=0.5,margs={}):
        p.value = x
        return self.minimize(**margs).fun - m.fun - ci_delta

    def _delta_nll_scan(self,m,p,x,ci_delta=0.5):
        p.value = x
        return self() - m.fun - ci_delta

    def confidence_intervals(self,minimum,method='scan',ci_delta=0.5,params=None,margs={}):
        '''
        Will either scan or profile the current likelihood to find the positive 
        and negative confidence intervals for minimized parameters about the 
        minimum.
        
        If profiling, the margs dictionary is used for keyword arguments to the
        minimize function.
        
        The desired confidence interval is given by ci_delta, which is a delta
        in the negative log likelihood from the minimum (0.5 for one sigma, etc)
        '''
        params = minimum.params if params is None else {p:minimum.params[p] for p in params}
        initial_state = [(p.value,p.fixed) for p in params]
        upper,lower = {},{}
        try:
            # set all parameters to minimum values
            for p,v in params.items():
                p.value = v
                p.fixed = False
            # compute confidence intervals for each parameter
            for p,v in params.items():
                p.fixed = True
                self.update_likelihood()
                if method == 'profile':
                    dnll = partial(self._delta_nll_profile,minimum,p,margs=margs,ci_delta=ci_delta)
                elif method == 'scan':
                    dnll = partial(self._delta_nll_scan,minimum,p,ci_delta=ci_delta)
                else:
                    raise Exception('Unknown confidence interval method')
                # brentq needs a bracketed interval. First try stepping half the
                # central value left and right of the minimum. Then increase 
                # step by a factor of two until the step is large enough to
                # contain the desired root.
                for step_factor in [0.5,1.0,2.0,5.0,10.0]:
                    try:
                        step = v*step_factor
                        lo = opt.brentq(dnll,v-step,v,xtol=0.01,rtol=0.00001)
                        hi = opt.brentq(dnll,v,v+step,xtol=0.01,rtol=0.00001)
                        break
                    except ValueError:
                        step_factor *= 2
                upper[p] = hi-v
                lower[p] = v-lo
                p.fixed = False
                p.value = v
        finally:
            # Put parameter settings back to how they were before
            for p,(v,c) in zip(params,initial_state):
                p.value = v
                p.fixed = c
            self.update_likelihood()
        minimum.upper = upper
        minimum.lower = lower
        return minimum
        

