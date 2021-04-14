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
try:
    import cupy as cp
    from cupyx.scipy.special import erf
except:
    cp = np # Use numpy to emulate cupy on CPU
    from scipy.special import erf
from .calculate import Calculation
    
class Signal(Calculation):
    '''
    Represents the monte-carlo data that is used to build a PDF for a single 
    class of events, and contains the logic to evaluate the PDF using an 
    adaptive kernel density estimation algorithm.
    
    The Signal is a function of the systematics it defines, and the result of 
    `calculate` should be passed as the `systs` arugment to eval_pdf functions.
    '''

    def __init__(self,name,observables,value=1.0):
        self.observables = observables
        self.a = cp.asarray([l for l in self.observables.lows])
        self.b = cp.asarray([h for h in self.observables.highs])
        self.nev_param = self.observables.analysis.add_parameter(name+'_nev',value=value,constant=False)
        self.systematics = [syst for dim_systs in zip(observables.scales,observables.shifts,observables.resolutions) for syst in dim_systs]
        super().__init__(name,self.systematics)
        
        
    def load_mc(self,mc_files):
        t_nij = []
        for fname in mc_files:
            t_nij.append(self.observables.read_file(fname))
        self.t_ij = cp.ascontiguousarray(cp.asarray(np.concatenate(t_nij)))
        self.sigma_j = cp.std(self.t_ij,axis=0)
        self.w_i = cp.ones(self.t_ij.shape[0])
        self.h_ij = self.adapt_bandwidth()
        

    _inv_sqrt_2pi = 1/cp.sqrt(2*cp.pi)

    def _kdpdf0(x_j,t_ij,h_j,w_i):
        '''
        x_j is the j-dimensional point to evaluate the PDF at
        t_ij are the i events in the PDF at j-dimensional points
        h_j are the bandwidths for all PDF events in dimension j
        '''
        w = cp.sum(w_i)
        h_j_prod = cp.prod(Signal._inv_sqrt_2pi/h_j)
        res = h_j_prod*cp.sum(w_i*cp.exp(-0.5*cp.sum(cp.square((x_j-t_ij)/h_j),axis=1)))/w
        return res if np == cp else res.get()
    
    _kdpdf0_multi = cp.RawKernel(r'''
        extern "C" __global__
        void _kdpdf0_multi(const double* x_kj, const double* t_ij, const double* h_j, const double* w_i, 
                           const int n_i, const int n_j, const int n_k, double* pdf_k) {
            int k = blockDim.x * blockIdx.x + threadIdx.x;
            if (k >= n_k) return;
            double pdf = 0.0;
            for (int i = 0; i < n_i; i++) {
                double prod = 1.0;
                double a = 0;
                for (int j = 0; j < n_j; j++) {
                    prod /= h_j[j]*2.5066282746310007;
                    double b = (x_kj[k*n_j+j]-t_ij[i*n_j+j])/h_j[j];
                    a += b * b;
                }
                pdf += w_i[i]*prod*exp(-0.5*a);
            }
            pdf_k[k] = pdf;
        }
        ''', '_kdpdf0_multi') if cp != np else None
        
    def _estimate_pdf(self,x_j,w_i=None):
        return self._estimate_pdf_multi([x_j],w_i=w_i)[0]
    
    def _estimate_pdf_multi(self,x_kj,w_i=None,get=True):
        if w_i == None:
            w_i = self.w_i
        n = self.t_ij.shape[0]
        h_j = (4/3/n)**(1/5)*self.sigma_j
        if cp == np:
            return np.asarray([Signal._kdpdf0(x_j,self.t_ij,h_j,self.w_i) for x_j in x_kj])
        else:
            x_kj = cp.asarray(x_kj)
            h_j = cp.ascontiguousarray(cp.asarray(h_j))
            pdf_k = cp.empty(x_kj.shape[0])
            block_size = 64
            grid_size = x_kj.shape[0]//block_size+1
            Signal._kdpdf0_multi((grid_size,),(block_size,),(x_kj,self.t_ij,h_j,w_i,
                                                             self.t_ij.shape[0],self.t_ij.shape[1],x_kj.shape[0],
                                                             pdf_k))
            pdf_k = pdf_k/cp.sum(self.w_i)
            return pdf_k.get() if get else pdf_k
        
    def adapt_bandwidth(self,w_i=None):
        '''
        Calculates and returns bandwidths for all pdf events.
        '''
        n = self.t_ij.shape[0]
        d = len(self.observables.dimensions)
        sigma = cp.prod(self.sigma_j)**(1/d)
        h_i = (4/(d+2))**(1/(d+4)) \
               * n**(-1/(d+4)) \
               / sigma \
               / self._estimate_pdf_multi(self.t_ij,w_i=w_i,get=False)**(1/d)
        h_ij = cp.outer(h_i,self.sigma_j)
        cp.cuda.Stream.null.synchronize()
        return cp.ascontiguousarray(h_ij)
    
    _sqrt2 = cp.sqrt(2)

    def _int_kdpdf1(a_j,b_j,t_ij,h_ij,w_i,get=True):
        '''
        Integrates the PDF evaluated by _kdpdf1 and _kdpdf1_multi.
        
        a_j and b_j are the j-dimensional points represneting the lower and
            upper bounds of integration
        t_ij are the i events in the PDF at j-dimensional points
        h_ij are the bandwidths of each PDF event i in dimension j
        w_i are the weights of each PDF event
        '''
        w = cp.sum(w_i)
        n = t_ij.shape[0]
        d = t_ij.shape[1]
        res = cp.sum(w_i*cp.prod(
                erf((b_j-t_ij)/h_ij/Signal._sqrt2) - erf((a_j-t_ij)/h_ij/Signal._sqrt2)
            ,axis=1))/w/(2**d)
        return res.get() if get else res
    
    _int_kdpdf1_multi = cp.RawKernel(r'''
        #define sqrt2 1.4142135623730951
        extern "C" __global__
        void _int_kdpdf1_multi(const double* a_kj, const double* b_kj, const double* t_ij, const double* h_ij, const double* w_i, 
                               const int n_i, const int n_j, const int n_k, double* int_k) {
            /*
            2D arrays are passed to several of these arguments with row-major memory layout. 
            
            This CUDA kernel integrates a Gaussian Kernel Density PDF in regions bounded
            by a_j and b_j in the lists a_kj and b_kj.
                k - region index
                j - dimension index
            t_ij is the events used to build the kernel density PDF
                i - pdf point index
                j - dimension index
            h_ij is the bandwidth used to build the kernel density PDF
                i - pdf point index
                j - dimension index
                
            w_i is the weight of each event
            n_i, n_j, n_k are the size of each index
            
            The resulting value is not normalized by the sum of weights, but otherwise
            is normalized from (-infty,+infty), and stored in int_k.
                k - data point index.
            */
            int k = blockDim.x * blockIdx.x + threadIdx.x;
            if (k >= n_k) return;
            double integral = 0.0;
            for (int i = 0; i < n_i; i++) {
                double prod = w_i[i];
                for (int j = 0; j < n_j; j++) {
                    prod *= erf((b_kj[k*n_j+j]-t_ij[i*n_j+j])/h_ij[i*n_j+j]/sqrt2) 
                          - erf((a_kj[k*n_j+j]-t_ij[i*n_j+j])/h_ij[i*n_j+j]/sqrt2);
                }
                integral += prod;
            }
            double power = 1.0;
            for (int j = 0; j < n_j; j++) power *= 2.0;
            int_k[k] = integral/power;
        }
        ''', '_int_kdpdf1_multi') if cp != np else None
        
    def int_pdf(self,a_j,b_j,systs=None):
        if systs is None:
            t_ij,h_ij,w_i = self.t_ij,self.h_ij,self.w_i
        else:
            t_ij,h_ij,w_i = systs
        return Signal._int_kdpdf1(a_j,b_j,t_ij,h_ij,w_i)
    
    def int_pdf_multi(self,a_kj,b_kj,systs=None,get=False):
        if systs is None:
            t_ij,h_ij,w_i = self.t_ij,self.h_ij,self.w_i
        else:
            t_ij,h_ij,w_i = systs
        if cp == np:
            return [Signal._int_kdpdf1(a_j,b_j,t_ij,h_ij,w_i) for a_j,b_j in zip(a_kj,b_kj)]
        else:
            int_k = cp.empty(a_kj.shape[0])
            block_size = 64
            grid_size = a_kj.shape[0]//block_size+1
            Signal._int_kdpdf1_multi((grid_size,),(block_size,),(a_kj,b_kj,t_ij,h_ij,w_i,
                                                                 t_ij.shape[0],
                                                                 t_ij.shape[1],
                                                                 a_kj.shape[0],
                                                                 int_k))
            int_k = int_k/cp.sum(self.w_i)
            return int_k.get() if get else int_k
        
    def _normalization(self,a=None,b=None,t_ij=None,h_ij=None,w_i=None):
        '''
        Calls _norm_kdpdf1 wit the defaults set to the observable bounds and 
        loaded mc data, with no systematics.
        '''
        if a is None:
            a=self.a
        if b is None:
            b=self.b
        if t_ij is None:
            t_ij=self.t_ij
        if h_ij is None:
            h_ij=self.h_ij
        if w_i is None:
            w_i=self.w_i
        return Signal._int_kdpdf1(a,b,t_ij,h_ij,w_i)
    
        
    def _kdpdf1(x_j,t_ij,h_ij,w_i):
        '''
        Evaluate a the normalized PDF at a single point using generic NumPy/CuPy
        code instead of a dedicated CUDA kernel.
        
        x_j is the j-dimensional point to evaluate the PDF at
        t_ij are the i events in the PDF at j-dimensional points
        h_ij are the bandwidths of each PDF event i in dimension j
        w_i are the weights of each PDF event
        '''
        res = cp.sum(w_i*cp.prod(Signal._inv_sqrt_2pi/h_ij,axis=1)*cp.exp(-0.5*cp.sum(cp.square((x_j-t_ij)/h_ij),axis=1)))
        return res if np == cp else res.get()

    _kdpdf1_multi = cp.RawKernel(r'''
        #define sqrt2pi 2.5066282746310007
        extern "C" __global__
        void _kdpdf1_multi(const double* x_kj, const double* t_ij, const double* h_ij, const double* w_i, 
                           const int n_i, const int n_j, const int n_k, double* pdf_k) {
            /*
            2D arrays are passed to several of these arguments with row-major memory layout. 
            
            This CUDA kernel evaluates a Gaussian Kernel Density PDF at datapoints x_kj.
                k - data point index
                j - dimension index
            t_ij is the events used to build the kernel density PDF
                i - pdf point index
                j - dimension index
            h_ij is the bandwidth used to build the kernel density PDF
                i - pdf point index
                j - dimension index
                
            w_i is the weight of each event
            n_i, n_j, n_k are the size of each index
            
            The resulting value is not normalized by the sum of weights, but otherwise
            is normalized from (-infty,+infty), and stored in pdf_k.
                k - data point index.
            */
            int k = blockDim.x * blockIdx.x + threadIdx.x;
            if (k >= n_k) return;
            double pdf = 0.0;
            for (int i = 0; i < n_i; i++) {
                double prod = w_i[i];
                double a = 0;
                for (int j = 0; j < n_j; j++) {
                    prod /= h_ij[i*n_j+j]*sqrt2pi;
                    double b = (x_kj[k*n_j+j]-t_ij[i*n_j+j])/h_ij[i*n_j+j];
                    a += b * b;
                }
                pdf += prod*exp(-0.5*a);
            }
            pdf_k[k] = pdf;
        }
        ''', '_kdpdf1_multi') if cp != np else None
    
    def eval_pdf(self, x_j, systs=None):
        '''
        Evaluates the signal's normalized PDF at one point. (Calls eval_pdf_multi.)
        '''
        return self.eval_pdf_multi([x_j],systs=systs)[0]
    
    def eval_pdf_multi(self, x_kj, systs=None, get=True):
        '''
        Evaluates the signal's normalized PDF at a list-like series of points. 
        
        If CuPy is present on the system, a CUDA kernel will be used to run this
        calculation on the default GPU. (See: Signal._kdpdf1_multi)
        '''
        if systs is None:
            t_ij,h_ij,w_i = self.t_ij,self.h_ij,self.w_i
        else:
            t_ij,h_ij,w_i = systs
        x_kj = cp.asarray(x_kj)
        norm = cp.asarray(self._normalization(t_ij=t_ij,h_ij=h_ij,w_i=w_i))
        if np == cp:
            return np.asarray([Signal._kdpdf1(x_j,t_ij,h_ij,w_i) for x_j in x_kj])/norm
        else:
            pdf_k = cp.empty(x_kj.shape[0])
            block_size = 64
            grid_size = x_kj.shape[0]//block_size+1
            Signal._kdpdf1_multi((grid_size,),(block_size,),(x_kj,t_ij,h_ij,w_i,
                                                             t_ij.shape[0],
                                                             t_ij.shape[1],
                                                             x_kj.shape[0],
                                                             pdf_k))
            pdf_k = pdf_k/cp.sum(self.w_i)/norm
            return pdf_k.get() if get else pdf_k
    
    def _transform_syst(self,inputs):
        '''
        Scales and shifts datapoints by those systematics. Shift is in the units
        of the scaled dimension.
        '''
        scales = inputs[0:3*len(self.observables.scales):3]
        shifts = inputs[1:3*len(self.observables.shifts):3]
        return scales*self.t_ij+shifts
        
    def _weight_syst(self,inputs):
        '''
        Reweight events. (e.g. neutrino survival probability would be 
        implemented here.)
        
        Note: this returns the weights unmodified. if weights are modified, 
        one should call adapt_bandwidth since the zeroth order estimate would
        change.
        '''
        return self.w_i,self.h_ij
        
    def _conv_syst(self,inputs,h_ij=None):
        '''
        Convolves the bandwidths with the resolutions scaled by the scale
        systematics. Resolutions are in the units of the scaled dimension.
        '''
        if h_ij is None:
            h_ij = self.h_ij
        scales = inputs[0:3*len(self.observables.scales):3]
        resolutions = inputs[2:3*len(self.observables.shifts):3]
        return cp.sqrt(cp.square(scales*h_ij) + cp.square(resolutions))
        
    def calculate(self,inputs,verbose=False):
        '''
        Calculates the systematically transformed PDF weights, events, and 
        bandwidths. 
        '''
        inputs = cp.asarray(inputs,dtype=cp.float64)
        w_i,h_ij = self._weight_syst(inputs)
        t_ij = self._transform_syst(inputs)
        h_ij = self._conv_syst(inputs)
        return t_ij,h_ij,w_i
