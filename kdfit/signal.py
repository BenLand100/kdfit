import numpy as np
try:
    import cupy as cp
    from cupyx.scipy.special import erf
except:
    cp = np # Use numpy to emulate cupy on CPU
    from scipy.special import erf
    
class Signal:
    '''
    Represents the monte-carlo data that is used to build a PDF for a single 
    class of events, and contains the logic to evaluate the PDF using an 
    adaptive kernel density estimation algorithm.
    '''

    def __init__(self,name,observables,guess=1.0):
        self.name = name
        self.observables = observables
        self.a = cp.asarray([l for l in self.observables.lows])
        self.b = cp.asarray([h for h in self.observables.highs])
        self.nev_param = self.observables.analysis.add_parameter(name+'_nev',guess=guess,constant=False)
        
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
        
    def _estimate_pdf(self,x_j):
        return self._estimate_pdf_multi([x_j])[0]
    
    def _estimate_pdf_multi(self,x_kj,get=True):
        n = self.t_ij.shape[0]
        h_j = (4/3/n)**(1/5)*self.sigma_j
        '''
        t_ij = [self.T(t_j,syst) for t_j in self.t_ij]
        w_i = [self.W(t_j,syst) for t_j in t_ij]
        h_ij = [self.C(t_j,h_j,syst) for t_j in t_ij]
        return Signal._kdpdf1(x_j,t_ij,h_ij,w_i)
        '''
        if cp == np:
            return np.asarray([Signal._kdpdf0(x_j,self.t_ij,h_j,self.w_i) for x_j in x_kj])
        else:
            x_kj = cp.asarray(x_kj)
            h_j = cp.ascontiguousarray(cp.asarray(h_j))
            pdf_k = cp.empty(x_kj.shape[0])
            block_size = 64
            grid_size = x_kj.shape[0]//block_size+1
            Signal._kdpdf0_multi((grid_size,),(block_size,),(x_kj,self.t_ij,h_j,self.w_i,
                                                             self.t_ij.shape[0],self.t_ij.shape[1],x_kj.shape[0],
                                                             pdf_k))
            pdf_k = pdf_k/cp.sum(self.w_i)
            return pdf_k.get() if get else pdf_k
        
    def adapt_bandwidth(self):
        n = self.t_ij.shape[0]
        d = len(self.observables.dimensions)
        sigma = cp.prod(self.sigma_j)**(1/d)
        h_i = (4/(d+2))**(1/(d+4)) \
               * n**(-1/(d+4)) \
               / sigma \
               / self._estimate_pdf_multi(self.t_ij,get=False)**(1/d)
        h_ij = cp.outer(h_i,self.sigma_j)
        cp.cuda.Stream.null.synchronize()
        return cp.ascontiguousarray(h_ij)
    
    _sqrt2 = cp.sqrt(2)

    def _norm_kdpdf1(a_j,b_j,t_ij,h_ij,w_i):
        '''
        a_j and b_j are the j-dimensional points represneting the lower and upper bounds of the PDF
        t_ij are the i events in the PDF at j-dimensional points
        h_ij are the bandwidths of each PDF event i in dimension j
        '''
        w = cp.sum(w_i)
        n = len(t_ij)
        d = len(t_ij[0])
        res = cp.sum(w_i*cp.prod(
                erf((b_j-t_ij)/h_ij/Signal._sqrt2) - erf((a_j-t_ij)/h_ij/Signal._sqrt2)
            ,axis=1))/w/(2**d)
        return res if np == cp else res.get()
    
    def _normalization(self):
        '''
        t_ij = [self.T(t_j,syst) for t_j in self.t_ij]
        w_i = [self.W(t_j,syst) for t_j in t_ij]
        h_ij = [self.C(t_j,h_j,syst) for t_j,h_j in zip(t_ij,self.h_ij)]
        return Signal._norm_kdpdf1(self.a,self.b,t_ij,h_ij,w_i)
        '''
        return Signal._norm_kdpdf1(self.a,self.b,self.t_ij,self.h_ij,self.w_i)
    
        
    def _kdpdf1(x_j,t_ij,h_ij,w_i):
        '''
        x_j is the j-dimensional point to evaluate the PDF at
        t_ij are the i events in the PDF at j-dimensional points
        h_ij are the bandwidths of each PDF event i in dimension j
        '''
        w = cp.sum(w_i)
        res = cp.sum(w_i*cp.prod(Signal._inv_sqrt_2pi/h_ij,axis=1)*cp.exp(-0.5*cp.sum(cp.square((x_j-t_ij)/h_ij),axis=1)))/w
        return res if np == cp else res.get()

    _kdpdf1_multi = cp.RawKernel(r'''
        extern "C" __global__
        void _kdpdf1_multi(const double* x_kj, const double* t_ij, const double* h_ij, const double* w_i, 
                           const int n_i, const int n_j, const int n_k, double* pdf_k) {
            int k = blockDim.x * blockIdx.x + threadIdx.x;
            if (k >= n_k) return;
            double pdf = 0.0;
            for (int i = 0; i < n_i; i++) {
                double prod = 1.0;
                double a = 0;
                for (int j = 0; j < n_j; j++) {
                    prod /= h_ij[i*n_j+j]*2.5066282746310007;
                    double b = (x_kj[k*n_j+j]-t_ij[i*n_j+j])/h_ij[i*n_j+j];
                    a += b * b;
                }
                pdf += w_i[i]*prod*exp(-0.5*a);
            }
            pdf_k[k] = pdf;
        }
        ''', '_kdpdf1_multi') if cp != np else None
    
    def eval_pdf(self, x_j):
        '''
        t_ij = [self.T(t_j,syst) for t_j in self.t_ij]
        w_i = [self.W(t_j,syst) for t_j in t_ij]
        h_ij = [self.C(t_j,h_j,syst) for t_j,h_j in zip(t_ij,self.h_ij)]
        return Signal._kdpdf1(x_j,t_ij,h_ij,w_i)/self._normalization(syst)
        '''
        return self.eval_pdf_multi([x_j])[0]
    
    def eval_pdf_multi(self, x_kj, get=True):
        x_kj = cp.asarray(x_kj)
        norm = cp.asarray(self._normalization())
        if np == cp:
            return np.asarray([Signal._kdpdf1(x_j,self.t_ij,self.h_ij,self.w_i) for x_j in x_kj])/norm
        else:
            pdf_k = cp.empty(x_kj.shape[0])
            block_size = 64
            grid_size = x_kj.shape[0]//block_size+1
            Signal._kdpdf1_multi((grid_size,),(block_size,),(x_kj,self.t_ij,self.h_ij,self.w_i,
                                                             self.t_ij.shape[0],self.t_ij.shape[1],x_kj.shape[0],
                                                             pdf_k))
            pdf_k = pdf_k/cp.sum(self.w_i)/norm
            return pdf_k.get() if get else pdf_k
