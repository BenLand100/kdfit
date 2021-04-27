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
from collections import deque

class Calculation:
    '''Represents stage of a calculation'''
        
    def __init__(self, name, parents, constant=False):
        '''
        parents should be a list of Calculation objects this calculation depends on
        constant sets the assumption that calculate(...) is always valid and 
        never changes (no parents, but not an input)
        '''
        self.name = name
        self.parents = parents
        self.constant = constant
        
    def calculate(self, inputs):
        '''
        inputs is a list of parent results in the order of self.parents
        '''
        pass 
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name

class Parameter(Calculation):
    '''
    Represents an input to a System
    
    value is the initial or fixed value of the Parameter
    constant sets whether the parameter is fixed (True) or floated (False)
    '''
    def __init__(self, name, value=None, fixed=True):
        super().__init__(name, [], constant=False)
        self.name = name
        self.value = value
        self.fixed = fixed
        
    def link(self,param):
        if param is None:
            self.parents = []
        else:
            self.parents = [param]
            self.constant = False
            
    def calculate(self, inputs, verbose=False):
        if len(self.parents) == 1:
            return inputs[0]
        raise Exception('Should not calculate unlinked parameters')

class System:
    '''
    Represents a series of calculations as a a collection of connected dependencies
    '''
    
    def __init__(self,outputs=[],verbose=False):
        '''
        output is a list of output Calculations (order used for result of calculate)
        '''
        
        self.outputs = outputs # output structures
        
        parts = [] # sequentially stores all calcualtions in network
        children_indexes = [] # indexes of child instances in parts for each instance in parts
        parents_indexes = [] # indexes of parent instances in parts for each instance in parts
        input_indexes = [] # indexes of head (potential input) instances
        
        stack = deque()
        for child in self.outputs: # iterate over outputs to walk up tree
            if child not in parts:
                child_index = len(parts)
                parts.append(child)
                if len(child.parents) == 0:
                    input_indexes.append(child_index)
                children_indexes.append([])
                parents_indexes.append([])
            else:
                child_index = parts.index(child)
            for parent in child.parents:
                stack.append((child_index,child,parent))
        while len(stack) > 0: 
            child_index,child,parent = stack.popleft()
            if parent not in parts:
                index = len(parts)
                parts.append(parent)
                if len(parent.parents) == 0:
                    input_indexes.append(index)
                children_indexes.append([])
                parents_indexes.append([])
            else:
                index = parts.index(parent)
                #This prevents the same Calculation from appearing multiple
                #times in the input list, but is necessary to avoid double
                #counting if the tree is not trivial
                if child_index in children_indexes[index]:
                    continue #already mapped this branch
            if verbose:
                print(parent,'=>',child)
            parents_indexes[child_index].append(index)
            children_indexes[index].append(child_index)
            if parent.parents is not None:
                for grandparent in parent.parents:
                    stack.append((index,parent,grandparent))
                
        self.parts = np.asarray(parts,dtype=object)
        self.output_indexes = np.asarray([parts.index(output) for output in self.outputs],dtype=np.int32) # indexes of output instances
        self.input_indexes = np.asarray(input_indexes,dtype=np.int32)
        self.children_indexes = [np.asarray(child_indexes,dtype=np.int32) for child_indexes in children_indexes]
        self.parents_indexes = [np.asarray(parent_indexes,dtype=np.int32) for parent_indexes in parents_indexes]
        
        self.state = np.asarray([None for p in self.parts],dtype=object)
        self.not_evaluated = np.ones_like(self.parts,dtype=bool)
        
    def classify_inputs(self):
        '''
        Sets all unlinked non-fixed Parameters to floated inputs,  and unlinked Calculations to fixed inputs. Returns a
        list of both types.
        '''
        floated = []
        fixed = []
        floated_indexes = []
        fixed_indexes = []
        constant_indexes = []
        for input_index in self.input_indexes:
            p = self.parts[input_index]
            if len(p.parents) == 0 and not p.constant:
                if not isinstance(p,Parameter):
                    raise Exception('Non-constant calculations with no parents must be parameters')
                if not p.fixed:
                    floated.append(p)
                    floated_indexes.append(input_index)
                else:
                    fixed.append(p)
                    fixed_indexes.append(input_index)
            else:
                constant_indexes.append(input_index)
        self.floated_indexes = np.asarray(floated_indexes,dtype=np.int32)
        self.fixed_indexes = np.asarray(fixed_indexes,dtype=np.int32)
        self.constant_indexes = np.asarray(constant_indexes,dtype=np.int32)
        return floated,fixed
    
    # FIXME Perhaps System should be agnostic to floated and fixed Parameters, 
    # and instead require all Parameters as input, making Analysis distinguish 
    # between floated and fixed, building a parameter list for both.
    
    def calculate(self,floated,verbose=False):
        if type(verbose) is bool:
            verbose = 1 if verbose else 0
        recompute = []
        for indexes,values in [(self.floated_indexes,floated),(self.fixed_indexes,[p.value for p in self.parts[self.fixed_indexes]])]:
            for index,value in zip(indexes,values):
                if self.state[index] is not value:
                    if verbose > 1:
                        print('Changed input:',self.parts[index],self.state[index],'=>',value)
                    self.not_evaluated[index] = False
                    self.state[index] = value
                    recompute.extend(self.children_indexes[index])
        recompute.extend([constant_index for constant_index in self.constant_indexes if self.state[constant_index] is None])
        recompute = np.unique(np.asarray(recompute,dtype=np.uint32))
        if verbose > 2:
            print('Top-level recompute:',self.parts[recompute])
        
        not_queued = np.ones_like(self.parts,dtype=bool)
        not_queued[recompute] = False
        invalidate_queue = deque(recompute)
        while len(invalidate_queue) > 0:
            index = invalidate_queue.popleft()
            not_queued[index] = True
            if not self.not_evaluated[index]:
                if verbose > 2:
                    print('Invalidate:',self.parts[index])
                self.not_evaluated[index] = True
                children = self.children_indexes[index]
                children = children[not_queued[children]]
                not_queued[children] = False
                invalidate_queue.extend(children)
        
        not_queued = np.ones_like(self.parts,dtype=bool)
        not_queued[recompute] = False
        recompute_queue = deque(recompute)
        while len(recompute_queue) > 0:
            index = recompute_queue.popleft()
            not_queued[index] = True
            parents = self.parents_indexes[index]
            inputs_not_evaluated = self.not_evaluated[parents]
            if verbose > 2:
                print('Testing',self.parts[index],'parents:',
                      ', '.join(['%s:%s'%(self.parts[i],self.not_evaluated[i]) for i in parents]))
            if np.any(inputs_not_evaluated): 
                continue #not ready yet
            if verbose:
                print('Calculating',self.parts[index])
            self.state[index] = self.parts[index].calculate(self.state[parents],verbose=verbose)
            self.not_evaluated[index] = False
            children = self.children_indexes[index]
            children = children[not_queued[children]]
            if verbose > 1:
                print('Queuing children:',self.parts[children])
            not_queued[children] = False
            recompute_queue.extend(children)

        outputs = [self.state[index] for index in self.output_indexes]
        return outputs
