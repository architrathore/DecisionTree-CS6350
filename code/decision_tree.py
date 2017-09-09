'''
Implementation of decision tree for Machine Learning-CS6350
Author: Archit Rathore
'''

class DataRecord:
    '''Class to store individual data points read from train and test files'''

    def __init__(self, label, firstname, lastname):
        self.label = 1 if label = '+' else 0
        self.firstname = firstname
        self.lastname = lastname

def read_dataset(filepath):
    '''Read train or test file and return a list DataRecord objects'''
    data = []
	with open(filepath, 'r') as f:
        for line in f:
            label, fname, lname = line.rstrip().split()
            record = DataRecord(label, fname, lname)
            data.append(record)

