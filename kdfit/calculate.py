import numpy as np
from collections import deque

class Calculation:
    '''Represents stage of a calculation'''
        
    def __init__(self, name, parents, constant=False):
        '''
        parents should be a list of Calculation objects this calculation depends on
        constant sets the assumption that calculate(...) is always valid (no parents, but not an input)
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
        
class System:
    '''
    Represents a series of calculations as a a collection of connected dependencies
    '''
    
    def __init__(self,inputs=[],outputs=[],verbose=False):
        '''
        inputs is a list of input Calculations (order used for argument to calculate)
        output is a list of output Calculations (order used for result of calculate)
        
        outputs should, in principle, depend on the inputs in some way
        '''
    
        self.inputs = inputs # input structures 
        self.outputs = outputs # output structures
        
        parts = [] # sequentially stores all calcualtions in network
        children_indexes = [] # indexes of child instances in parts for each instance in parts
        parents_indexes = [] # indexes of parent instances in parts for each instance in parts
        constant_indexes = []
        
        stack = deque()
        for child in self.outputs: # iterate over outputs to walk up tree
            if child not in parts:
                child_index = len(parts)
                parts.append(child)
                if child.constant:
                    constant_indexes.append(child_index)
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
                if parent.constant:
                    constant_indexes.append(index)
                children_indexes.append([])
                parents_indexes.append([])
            else:
                index = parts.index(parent)
            if verbose:
                print(parent,'=>',child)
            parents_indexes[child_index].append(index)
            children_indexes[index].append(child_index)
            if parent.parents is not None:
                for grandparent in parent.parents:
                    stack.append((index,parent,grandparent))
                
        self.parts = np.asarray(parts,dtype=object)
        self.input_indexes = np.asarray([parts.index(input) for input in self.inputs],dtype=np.int32) # indexes of input instances
        self.output_indexes = np.asarray([parts.index(output) for output in self.outputs],dtype=np.int32) # indexes of output instances
        self.constant_indexes = np.asarray(constant_indexes,dtype=np.int32) # indexes constant instances
        self.children_indexes = [np.asarray(child_indexes,dtype=np.int32) for child_indexes in children_indexes]
        self.parents_indexes = [np.asarray(parent_indexes,dtype=np.int32) for parent_indexes in parents_indexes]
        
        #FIXME eventually going to want to _avoid_ recalculating things that haven't changed
        evaluation_order = []
        input_children = np.unique(np.concatenate([self.children_indexes[i] for i in np.concatenate([self.input_indexes,self.constant_indexes])]))
        not_evaluated  = np.ones_like(self.parts,dtype=bool)
        not_evaluated[self.input_indexes] = False
        not_evaluated[self.constant_indexes] = False
        not_queued = np.ones_like(self.parts,dtype=bool)
        not_queued[input_children] = False
        recompute_queue = deque(input_children)
        while len(recompute_queue) > 0:
            index = recompute_queue.popleft()
            not_queued[index] = True
            inputs_not_evaluated = not_evaluated[self.parents_indexes[index]]
            if np.any(inputs_not_evaluated): 
                continue #not ready yet
            evaluation_order.append(index)
            not_evaluated[index] = False
            children = self.children_indexes[index]
            children = children[not_queued[children]]
            not_queued[children] = False
            recompute_queue.extend(children)
        self.evaluation_order = np.asarray(evaluation_order,dtype=np.int32)
        if verbose:
            print('Evaluation order:',self.parts[self.evaluation_order])
            
    def calculate(self,inputs,verbose=False):
        state = np.asarray([None for p in self.parts],dtype=object)
        
        for constant_index in zip(self.constant_indexes):
            state[constant_index] = self.parts[constant_index].calculate(None,verbose=verbose)
            if verbose:
                print(self.parts[constant_index],state[constant_index])
            
        for input_index,input in zip(self.input_indexes,inputs):
            state[input_index] = input
            if verbose:
                print(self.parts[input_index],state[input_index])
        
        for index in self.evaluation_order:
            state[index] = self.parts[index].calculate(state[self.parents_indexes[index]],verbose=verbose)
            if verbose:
                print(self.parts[index],state[index])

        outputs = [state[index] for index in self.output_indexes]
        return outputs
