#!/bin/python
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
        self.attr_index = attr_index    # index of attribute that is tested in the 
                                        # feature array, -1 for leaf nodes
        self.prediction = None          # the prediction from this node, None for all non-leaf nodes
        self.branches = {}              # dictionary holding subtrees of the tree, empty for leaf nodes
        
    def __repr__(self):
        if self.attr_index == -1:
            return '["{0}"]'.format(self.prediction)
        else:
            return '[{}]'.format(str(self.attr_index))

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


def get_entropy(labels):
    prob_pos, prob_neg = sum(labels == 1)/len(labels), sum(labels == 0)/len(labels)
    if prob_pos == 0 or prob_neg == 0:
        return 0
    entropy = prob_pos * np.log2(prob_pos) + prob_neg * np.log2(prob_neg)
    return -1.0 * entropy
    
    
def find_best_split(S, attributes, labels):
    # Return the first attribute for now
    # return list(attributes)[0]
    '''Implements information gain based on reduction in entropy
    for feature selection'''
    
    if len(attributes) == 1:
        return next(iter(attributes))
    max_info_gain = 0
    best_split_attribute = None
    root_node_entropy = get_entropy(labels)
    for attribute in attributes:
        attribute_value_set = set(S[:, attribute])
        weighted_entropy = 0
        for value in attribute_value_set:
            value_indices = S[:, attribute] == value
            value_labels = labels[value_indices]
            weighted_entropy += len(value_labels) * get_entropy(value_labels)
        info_gain = root_node_entropy - weighted_entropy/len(S)
        if info_gain >= max_info_gain:
            best_split_attribute = attribute
            max_info_gain = info_gain
    return best_split_attribute


def ID3(S, attributes, labels, depth=0, max_depth=100):
    '''ID3 algorithm for building a decision tree'''

    # If all examples have the same label
    if check_same_label(labels) or len(attributes) == 0 or depth >= max_depth:
        # Return a single node tree with the label
        leaf_node = DecisionNode(-1)
        leaf_node.prediction = labels[0]
        return leaf_node

    else:
        # Create a root node for tree
        root_node = DecisionNode(-1)

        # print("Attributes", attributes)
        # Find attribute A that best classifies the dataset S
        splitting_attr = find_best_split(S, attributes, labels)
        root_node.attr_index = splitting_attr
        
        # For each value v that the A can take:
        for split_val in set(S[:, splitting_attr]):
            # Find subset of examples S_v with A = v

            # Get indices with splitting_att = split_val
            indices = S[:, splitting_attr] == split_val
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
                S_v = S[indices]
                label_v = labels[indices]
                remaining_attr = attributes - {splitting_attr}
                root_node.branches[split_val] = ID3(S_v, remaining_attr, label_v, 
                                                    depth=depth + 1, max_depth=max_depth)

        return root_node


def print_tree(decision_tree):
    '''Do a level order traversal to print the decision tree'''
    # Create two queues, one holds the current depth nodes
    # and the other holds the nodes of the next level
    curr_level, next_level = [], []
    
    # Append the root to current level's queue
    curr_level.append(decision_tree)
    
    # While there are still nodes to visit
    while (len(curr_level) > 0):
        # Get the first node in queue
        curr_node = curr_level.pop(0)
        
        # Print it's data
        print(curr_node, end=' ')
        
        # For each child of the current node
        for branch in curr_node.branches:
            # Add all children to the next level's queue
            next_level.append(curr_node.branches[branch])
            
        # If the current level's queue is empty
        if len(curr_level) == 0:
            print()
            # Swap current and next level queues
            curr_level, next_level = next_level, curr_level


def predict(decision_tree, x):
    '''Given a decision tree and input, find the prediction'''

    # print('Attrib: ', decision_tree.attr_index)
    # If the given node is leaf
    if decision_tree.attr_index == -1:
        return decision_tree.prediction
    
    # Recursively call prediction based on current node's splitting attribute
    else:
        decision_attr_val = x[decision_tree.attr_index]
        # print('Dec_attr_index:', decision_tree.attr_index)
        # print('Dec attrib value: ', decision_attr_val)
        # print('Branches: ', decision_tree.branches)
        # print('------------------------')
        if decision_attr_val in decision_tree.branches:
            return predict(decision_tree.branches[decision_attr_val], x)
        else:
            try:
                return decision_tree.branches[not decision_tree].prediction
            except:
                return True

        
def accuracy(decision_tree, X, y_true):
    '''Find the accuracy of given decision tree on dataset X with 
    ground truth y_true'''

    # y_pred = np.apply_along_axis(lambda x: predict(decision_tree, x), 1, X)
    y_pred = []
    for x in X:
        try:
            y_pred.append(predict(decision_tree, x))
        except:
            y_pred.append(True)

    return np.average(y_pred == y_true)

def cross_validate(function_list, depth):
    splits = {0, 1, 2, 3}
    accuracies = []
    for validation_idx in splits:
        validation_data = read_dataset('../Dataset/Updated_CVSplits/updated_training0{}.txt'.format(validation_idx))
        training_data = []
    
        for training_idx in splits - {validation_idx}:
            training_data += read_dataset('../Dataset/Updated_CVSplits/updated_training0{}.txt'.format(training_idx))
        
        X_train = build_features(training_data, function_list)
        y_train = np.array([d.label for d in training_data])
        X_validate = build_features(validation_data, function_list)
        y_validate = np.array([d.label for d in validation_data])
        attributes = set(range(0, (X_train.shape[1])))        
        
        dec_tree = ID3(X_train, attributes, y_train, max_depth=depth)
        accuracies.append(accuracy(dec_tree, X_validate, y_validate))

    return accuracies

def q1():
    training_data = read_dataset('../Dataset/updated_train.txt')
    test_data = read_dataset('../Dataset/updated_test.txt')

    X_train = build_features(training_data, func_list)
    y_train = np.array([d.label for d in training_data])
    attributes = set(range(0, (X_train.shape[1])))

    dec_tree = ID3(X_train, attributes, y_train)

    X_test = build_features(test_data, func_list)
    y_test = np.array([d.label for d in test_data])
    print("=====================================================")
    print("1(c) : Training accuracy = ", accuracy(dec_tree, X_train, y_train))
    print("=====================================================\n")
    print("======================= 1 (d) =======================")
    print("1(d) : Test accuracy     = ", accuracy(dec_tree, X_test, y_test))
    print("=====================================================\n")
    print("=====================================================")
    print("1(e) : Max depth         = ", len(func_list))
    print("=====================================================\n")

def q2(depths):
    print(" ___________________________________________________ ")
    print("|              Cross Validation results             |")
    print("|---------------------------------------------------|")
    print("|{: ^16}|{: ^16}|{: ^17}|".format("Depth", "Mean", "Std dev"))
    print("|---------------------------------------------------|")

    for depth in depths:
        accuracies = cross_validate(func_list, depth)
        mean, std_dev = np.mean(accuracies), np.std(accuracies)
        print("|{: ^16}|{: ^16.5}|{: ^17.5}|".format(depth, mean, std_dev))
    print("|___________________________________________________|")

    training_data = read_dataset('../Dataset/updated_train.txt')
    test_data = read_dataset('../Dataset/updated_test.txt')

    X_train = build_features(training_data, func_list)
    y_train = np.array([d.label for d in training_data])
    attributes = set(range(0, (X_train.shape[1])))

    dec_tree = ID3(X_train, attributes, y_train, max_depth=5)

    X_test = build_features(test_data, func_list)
    y_test = np.array([d.label for d in test_data])
    print()
    print("Using max_depth = 5 from the cross validation results")
    print("=====================================================")
    print("2(b) : Training accuracy = ", accuracy(dec_tree, X_train, y_train))
    print("=====================================================\n")
    print("======================= 1 (d) =======================")
    print("2(c) : Test accuracy     = ", accuracy(dec_tree, X_test, y_test))
    print("=====================================================\n")


func_list = [lambda x: len(x.split()[0]) > len(x.split()[-1]),     # length of first name more than last name
             lambda x: len(x.split()) > 2,                         # do they have a middle name
             lambda x: x.split()[0][0] == x.split()[-1][0],        # first letter of first and last name are same
             lambda x: x.split()[0][0] < x.split()[-1][0],         # first letter is smaller than last name             
             lambda x: x[1] in ['a','e','i','o','u'],              # second letter of first name is vowel
             lambda x: len(x.split()[-1])%2 == 0,                  # number of letters in last name is even
             lambda x: sum(map(ord, x))%2 == 0,
             lambda x: x[2] in {'a', 'e', 'i', 'o', 'u'},
             lambda x: x[3] in {'a', 'e', 'i', 'o', 'u'},
             lambda x: x[4] in {'a', 'e', 'i', 'o', 'u'},
             lambda x: x[5] in {'a', 'e', 'i', 'o', 'u'},
             lambda x: len(set(x)) > 4,
             lambda x: len(set(x)) > 8,
             lambda x: len(list(filter(lambda x: x in {'a', 'e', 'i', 'o', 'u'}, x)))%2==0,
             lambda x: 'a' < x[0] and x[0] <= 'k',
             lambda x: 'k' < x[0] and x[0] <= 'r',
             lambda x: 'r' < x[0] and x[0] <= 'z',
             lambda x: 'a' < x[1] and x[1] <= 'k',
             lambda x: 'k' < x[1] and x[1] <= 'r',
             lambda x: 'r' < x[1] and x[1] <= 'z',
             lambda x: 'a' < x[2] and x[2] <= 'k',
             lambda x: 'k' < x[2] and x[2] <= 'r',
             lambda x: 'r' < x[2] and x[2] <= 'z'
             ]

q1()
q2([1, 2, 3, 4, 5, 10, 15, 20])