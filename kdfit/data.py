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

from .calculate import Calculation

import numpy as np
try:    
    import uproot4
except:
    pass
try:
    import h5py
except:
    pass

class DataLoader(Calculation):
    '''
    This is a generic data input Calculation. Create subclasses of Calculation
    to load data of other formats. Subclasses should load data when called.
    '''
    
    def __init__(self,name):
        super().__init__(name, [], constant=True)
        
    def calculate(self,inputs,verbose=False):
        return self


class HDF5Data(DataLoader):
    '''
    Assumes each dimension of an Observable is stored as a dataset in an HDF5
    file. The datasets should be one dimensional and indexed by event number.
    All should be the same shape. Will chain together multiple files into one
    larger dataset.
    '''

    def __init__(self,name,filenames,datasets,max_events=None):
        super().__init__(name)
        self.filenames = filenames
        self.datasets = datasets
        self.max_events = max_events
    
    def __call__(self):
        print('Loading:',', '.join(self.filenames))
        data = [[] for ds in self.datasets]
        for fname in self.filenames:
            with h5py.File(fname,'r') as hf:
                total = len(data[0])
                for j,ds in enumerate(self.datasets):
                    if self.max_events is not None:
                        to_read = self.max_events - total 
                        ds = hf[ds]
                        if total > ds.shape[0]:
                            data[j].extend(ds[:])
                        else:
                            data[j].extend(ds[:to_read])
                    else:
                        data[j].extend(hf[ds][:])
            if self.max_events is not None and len(data[0]) >= self.max_events:
                break
        if self.max_events is not None:
            return np.asarray(data)[:,:self.max_events].T
        else:
            return np.asarray(data).T
        
class NPYData(DataLoader):
    '''
    Assumes each dimension of an Observable is stored as a dataset in an HDF5
    file. The datasets should be one dimensional and indexed by event number.
    All should be the same shape. Will chain together multiple files into one
    larger dataset.
    '''

    def __init__(self,name,filenames,indexes,ordering='ij'):
        super().__init__(name)
        self.filenames = filenames
        self.indexes = np.asarray(indexes,dtype=np.int32)
        self.ordering = ordering
        
    def __call__(self):
        print('Loading:',', '.join(self.filenames))
        x_nij = []
        for fname in self.filenames:
            events = np.load(fname)
            if self.ordering == 'ij':
                x_ = events[:,self.indexes]
                x_nij.append(x_)
            elif self.ordering == 'ji':
                x_ = events[self.indexes,:]
                x_nij.append(x_.T)
            else:
                raise Exception('Unknown ordering')
        return np.concatenate(x_nij)
                    
class SNOPlusNTuple(DataLoader):

    def __init__(self,name,filenames,branches,max_events=None):
        super().__init__(name)
        self.filenames = filenames
        self.branches = branches
        self.max_events = max_events
        
    def __call__(self):
        print('Loading:',', '.join(self.filenames))
        x_nij = []
        events = 0
        for fname in self.filenames:
            try:
                with uproot4.open(fname) as froot:
                    x_ji = np.asarray([froot['output'][branch].array() for branch in self.branches])
                    x_nij.append(x_ji.T)
                    events += x_ji.shape[1]
                if self.max_events is not None and events > self.max_events:
                    break
            except:
                print('Couldn''t read',fname)
        print('Loaded',events,'events')
        return np.concatenate(x_nij)
