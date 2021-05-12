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

def binning_to_edges(binning,lows=None,highs=None):
    if type(binning) == int:
        return [cp.linspace(lows[j],highs[j],binning) for j in range(len(lows))]
    else:
        if type(binning[0]) == int:
            return [cp.linspace(lows[j],highs[j],bins) for j,bins in enumerate(binning)]
        else:
            return [cp.asarray(edges) for edges in binning]
            
def edges_to_points(bin_edges):
    a_kj = cp.ascontiguousarray(cp.asarray([cp.asarray(x) for x in it.product(*[edges[:-1] for edges in bin_edges])]))
    b_kj = cp.ascontiguousarray(cp.asarray([cp.asarray(x) for x in it.product(*[edges[1: ] for edges in bin_edges])]))
    return a_kj,b_kj
    
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
            self.a_kj, self.b_kj = edges_to_points(self.bin_edges)
            self.bin_vol = cp.ascontiguousarray(cp.prod(self.b_kj-self.a_kj,axis=1))
        self.last_systs = None
        self.bin_ints = None
        
    def calculate(self,inputs,verbose=False):
        if self.last_systs is not inputs[0]:
            systs = inputs[0]
            if self.binned_correctly:
                self.bin_ints = systs
            else:
                norm = self.pdf.int_pdf([edges[0] for edges in self.bin_edges],[edges[-1] for edges in self.bin_edges],systs=systs) 
                self.bin_ints = self.pdf.int_pdf_multi(self.a_kj,self.b_kj,systs=systs)/norm
            self.last_systs = systs
        return self.bin_ints
            
