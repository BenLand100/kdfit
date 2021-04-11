from .observables import Observables
from .calculate import System
from .term import Parameter,Sum

import scipy.optimize as opt
    
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
        
    def load_mc(self,mc_files):
        for obs,sig_map in mc_files.items():
            if type(obs) is str:
                obs = self.observables[obs]
            for sig,fnames in sig_map.items():
                if type(sig) is str:
                    sig = obs.signals[sig]
                sig.load_mc(fnames)
                
    def load_data(self,data_files):
        for obs,fnames in data_files.items():
            if type(obs) is str:
                obs = self.observables[obs]
            obs.load_data(fnames)
            
    def create_likelihood(self):
        self._params = [p for p in self.parameters.values()]
        self._floated = [p for p in self._params if not p.constant]
        print('Floated Parameters:',self._floated)
        self._terms = []
        for name,obs in self.observables.items():
            nllfn = obs.get_likelihood()
            self._terms.append(nllfn)
        self._outputs = [Sum('Total_Likelihood',*self._terms)]
        print('Ouput Values:',self._outputs)
        self._system = System(self._floated,self._outputs,verbose=True)
        
    def __call__(self,min_params):
        outputs = self._system.calculate(min_params)
        return outputs[0]
        
    def minimize(self,**kwargs):
        initial = [g if (g:=p.guess) else 1.0 for p in self._floated]
        minimum = opt.minimize(self,x0=initial,**kwargs)
        return minimum
