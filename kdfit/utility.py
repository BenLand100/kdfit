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
import itertools as it
try:
    import cupy as cp
except:
    print('kdfit.utility could not import CuPy - falling back to NumPy')
    cp = np # Use numpy to emulate cupy on CPU
from .calculate import Calculation

def binning_to_edges(binning):
    if type(binning) == int:
        return [cp.linspace(pdf.observables.lows[j],pdf.observables.highs[j],binning) for j in range(len(pdf.observables.dimensions))]
    else:
        if type(binning[0]) == int:
            return [cp.linspace(observables.lows[j],observables.highs[j],bins) for j,bins in enumerate(binning)]
        else:
            return binning
    
class PDFEvaluator(Calculation):

    def __init__(self,name,pdf,data):
        super().__init__(name,[pdf,data])
        self.pdf = pdf
        
    def calculate(self,inputs,verbose=False):
        systs,x_ij = inputs
        
        x_kj = cp.ascontiguousarray(cp.asarray(x_ij))
        if cp == np:
            return self.pdf.eval_pdf_multi(x_kj,systs=systs)
        else:
            return self.pdf.eval_pdf_multi(x_kj,systs=systs,get=False)
    
class PDFBinner(Calculation):

    def __init__(self,name,pdf,binning):
        from .signal import BinnedPDF
        super().__init__(name,[pdf])
        self.pdf = pdf        
        if isinstance(pdf,BinnedPDF) and pdf.binning == binning:
            self.binned_correctly = True
        else:
            self.binned_correctly = False
            self.bin_edges = binning_to_edges(binning)
            self.bin_edges = cp.ascontiguousarray(cp.asarray(self.bin_edges)) #FIXME this won't work with different number of bins in each dimension
            self.a_kj = cp.ascontiguousarray(cp.asarray(list(it.product(*self.bin_edges[:, :-1]))))
            self.b_kj = cp.ascontiguousarray(cp.asarray(list(it.product(*self.bin_edges[:,1:  ]))))
            self.bin_vol = cp.ascontiguousarray(cp.prod(self.b_kj-self.a_kj,axis=1))
        self.last_systs = None
        self.bin_ints = None
        
    def calculate(self,inputs,verbose=False):
        if self.last_systs is not inputs[0]:
            systs = inputs[0]
            if self.binned_correctly:
                self.bin_ints = systs
            else:
                norm = self.pdf.int_pdf(self.bin_edges[:,0],self.bin_edges[:,-1],systs=systs) 
                self.bin_ints = self.pdf.int_pdf_multi(self.a_kj,self.b_kj,systs=systs)/norm
            self.last_systs = systs
        return self.bin_ints
            
