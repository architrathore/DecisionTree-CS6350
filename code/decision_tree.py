'''
Implementation of decision tree for Machine Learning-CS6350
Author: Archit Rathore
'''

class DataRecord:
    '''Class to store individual data points read from train and test files'''

    def __init__(self, label, name):
        self.label = 1 if label == '+' else 0
        self.name = name

    def __repr__(self):
        return str(self.label) + ' ' + ' '.join(self.name)

def read_dataset(filepath):
    '''Read train or test file and return a list DataRecord objects'''
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip().split()
            label, name = line[0], line[1:]
            data.append(DataRecord(label, name))
    return data

print(read_dataset('../Dataset/updated_train.txt')[0:3])
