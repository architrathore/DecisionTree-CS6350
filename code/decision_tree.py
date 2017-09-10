'''
Implementation of decision tree for Machine Learning-CS6350
Author: Archit Rathore
'''
import numpy as np
from collections import Counter


class DataRecord:
    '''Class to store individual data points read from train and test files'''

    def __init__(self, label, name):
        self.label = 1 if label == '+' else 0
        self.name = ' '.join(name).lower()

    def __repr__(self):
        return str(self.label) + ' ' + self.name
    
class DecisionNode:
    '''Represent a node in the decision tree'''
    def __init__(self, attr_index):
        self.attr_index = attr_index    # index of attribute that is test in the 
                                        # feature array, -1 for leaf nodes
        self.prediction = None          # the prediction from this node, None for all non-leaf nodes
        self.branches = {}              # dictionary holding subtrees of the tree, empty for leaf nodes
        
    def __repr__(self):
        if self.prediction is not None:
            return '[{0: ^4} -> "{1}"]'.format(self.attr_index, self.prediction)
        else:
            return '[{0: ^4}]'.format(str(self.attr_index))


def read_dataset(filepath):
    '''Read train or test file and return a list DataRecord objects'''
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip().split()
            label, name = line[0], line[1:]
            data.append(DataRecord(label, name))
    return data

def build_features(dataset, func_list):
    '''Apply a sequence of functions to the dataset to get a feature array
    for each data point'''
    n, d = len(dataset), len(func_list)
    X = np.empty((n, d), dtype=int)
    for idx, datapoint in enumerate(dataset):
        X[idx] = np.array([f(datapoint.name) for f in func_list])
    return X

def check_same_label(arr):
    '''Check if all elements in arr are identical and return it'''
    return len(set(arr)) == 1

def find_best_split(dataset, attributes, labels):
    # Return the first attribute for now
    return list(attributes)[0]

def ID3(dataset, attributes, labels):
    '''ID3 algorithm for building a decision tree'''
    
    # If all examples have the same label
    if check_same_label(labels) and len(attributes) != 0:
        # Return a single node tree with the label
        leaf_node = DecisionNode(-1)
        leaf_node.prediction = labels[0]
        return leaf_node
    
    else:
        # Create a root node for tree
        root_node = DecisionNode(-1)
        
        # print("Attributes", attributes)
        # Find attribute A that best classifies the dataset S
        splitting_attr = find_best_split(dataset, attributes, labels)
        root_node.attr_index = splitting_attr
        
        # For each value v that the A can take:
        for split_val in set(dataset[:, splitting_attr]):
            # Find subset of examples S_v with A = v
            
            # Get indices with splitting_att = split_val
            indices = dataset[:, splitting_attr] == split_val    
            # indices is a boolean indicator array 
            
            # If S_v is empty
            if sum(indices) == 0:
                # Find the most common label in S
                most_common_label = Counter(labels).most_common(1)
                # Create a leaf node with this label
                leaf_node = DecisionNode(-1)
                leaf_node.prediction = most_common_label
                # Add the leaf node to this branch of node
                root_node.branches[split_val] = leaf_node
                
            else:
                # Add subtree ID3(S_v, Attributes - {A}, Label_v)
                S_v = dataset[indices]
                label_v = labels[indices]
                remaining_attr = attributes - {splitting_attr}
                root_node.branches[split_val] = ID3(S_v, remaining_attr, label_v)
                
        return root_node

dataset = read_dataset('../Dataset/updated_train.txt')

func_list = [lambda x: x[0] in ['a','e','i','o','u'],    # first letter is vowel
             lambda x: x[-1] in ['a','e','i','o','u'],   # last letter is vowel
             lambda x: len(x)%2 == 0                     # length of name is even
            ]

X = build_features(dataset, func_list)
y = np.array([d.label for d in dataset])

X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])
# a = ID3(X, {0, 1, 2}, y)
a = ID3(X_and, {0,1}, y_and)
print(a)