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

import h5py
import numpy as np

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

    def __init__(self,name,filenames,datasets):
        super().__init__(name)
        self.filenames = filenames
        self.datasets = datasets
    
    def __call__(self):
        print('Loading:',', '.join(self.filenames))
        data = [[] for ds in self.datasets]
        for fname in self.filenames:
            with h5py.File(fname,'r') as hf:
                for j,ds in enumerate(datasets):
                    data[j].extend(hf[ds])
        return np.asarray(data)
        
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
                    
