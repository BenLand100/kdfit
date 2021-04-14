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

import numpy as np
from .term import UnbinnedNegativeLogLikelihoodFunction, BinnedNegativeLogLikelihoodFunction
from .signal import Signal

class Observables:
    '''
    Represents a multi-dimensional (multi-observable) dataset and the PDFs used
    to calculate its likelihood.
    
    Contains the following:
        - A list of quantities that define the axes (dimesions) of some 
          multidimensional PDF along with the lower (lows) and upper (highs) ROI
          for each axis 
        - A set of component event classes (signals) which are used to build the
          full PDF by summing the contribution from each class
    '''
    
    def __init__(self,name,analysis,binning=None):
        self.name = name
        self.analysis = analysis
        self.dimensions = []
        self.indexes = []
        self.lows = []
        self.highs = []
        self.scales = []
        self.shifts = []
        self.resolutions = []
        
        self.x_ij = None
        self.binning = binning
        self.signals = {}
    
    def add_dimension(self,name,index,low,high):
        self.dimensions.append(name)
        self.indexes.append(index)
        self.lows.append(low)
        self.highs.append(high)
        scale = self.analysis.add_parameter(name+'_scale',value=1.0)
        self.scales.append(scale)
        shift = self.analysis.add_parameter(name+'_shift',value=0.0)
        self.shifts.append(shift)
        resolution = self.analysis.add_parameter(name+'_resolution',value=0.0)
        self.resolutions.append(resolution)
        return scale,shift,resolution
        
    def add_signal(self,name,*args,**kwargs):
        if name in self.signals:
            raise Exception('Duplicate name: '+name)
        sig = Signal(name,self,*args,**kwargs)
        self.signals[name] = sig
        return sig
        
    def load_data(self,data_files):
        x_nij = []
        for fname in data_files:
            x_nij.append(self.read_file(fname))
        self.x_ij = np.concatenate(x_nij)
        
    def eval_pdf(obs, x_j):
        return self.eval_pdf_multi([x_j])[0]
    
    def eval_pdf_multi(self, x_kj):
        n_evs = [s.nev_param.value for s in self.signals.values()]
        return np.sum([ n*s.eval_pdf_multi(x_kj) for n,s in zip(n_evs,self.signals.values()) ],axis=0)/np.sum(n_evs)
    
    def get_likelihood(self):
        if self.binning is None:
            return UnbinnedNegativeLogLikelihoodFunction(
                self.name+'_UnbinnedLikelihood',
                self.x_ij,
                list(self.signals.values()),
                self)
        else:
            return BinnedNegativeLogLikelihoodFunction(
                self.name+'_BinnedLikelihood',
                self.x_ij,
                list(self.signals.values()),
                self,
                binning=self.binning)
            
        
    def read_file(self,fname):
        events = np.load(fname)
        t_ji = [events[:,idx] for idx in self.indexes]
        return np.asarray(t_ji).T
