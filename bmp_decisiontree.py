import csv
import operator
import random

# Checks if string is float
#
def isfloat(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# Checks if string is int
#
def isint(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

# Reads csv dataset and returns it
#
def readdataset(filename):
    data = []
    with open(filename) as f:
        csvreader = csv.reader(f, delimiter=',')
        for row in csvreader:
            for i in range(len(row)):
                if isint(row[i]):
                    row[i] = int(row[i])
                elif isfloat(row[i]):
                    row[i] = float(row[i])
            data.append(row)
    return data

# Returns true if val is int or float
#
def isnumeric(val):
    return isinstance(val, int) or isinstance(val, float)

# The row is a row from the dataset
# the question is a tuple (index in row, value to compare)
# Function assumes >= for numeric data
#
def askquestion(row, question, op = operator.ge):
    valonrow = row[question[0]]

    if isnumeric(valonrow):
        return op(valonrow, question[1])
    else:
        return valonrow == question[1]

# Returns dictionary with a count of all classes in the data
#
def countclasses(data):
    res = {}
    for row in data:
        label = row[-1]
        if label in res:
            res[label] += 1
        else:
            res[label] = 1
    return res

# Gets the gini impurity of some dataset with class labes as last element in each row
#
def gini(data):
    classcount = countclasses(data)
    return 1.0 - sum([(classcount[c]/float(len(data)))**2 for c in classcount])

# Calculates info gain for split
#
def infogain(parentval, truelist, falselist):
    numelements = float(len(falselist) + len(truelist))
    # These are the weights
    pfalse = float(len(falselist)) / numelements
    ptrue = float(len(truelist)) / numelements
    # parent uncertainty - weighted avrage of child unertainty
    return parentval - (pfalse * gini(falselist) + ptrue * gini(truelist))

# Splits data for a question
#
def splitdata(data, question):
    truelist = []
    falselist = []
    for row in data:
        if askquestion(row, question):
            truelist.append(row)
        else:
            falselist.append(row)

    return truelist, falselist

# Finds the question with the highest info gain
#
def findbestquestion(impurity, data):
    bestgain = 0.0
    bestquestion = None
    numattribs = len(data[0]) - 1 # number of cols - 1

    for col in range(numattribs):
        uniquevals = set([row[col] for row in data]) # gets unique values in the current column

        for val in uniquevals:
            question = (col, val)

            truelist, falselist = splitdata(data, question)

            if len(truelist) == 0 or len(falselist) == 0:
                continue

            gain = infogain(impurity, truelist, falselist)

            if gain >= bestgain:
                bestgain = gain;
                bestquestion = question

    return bestquestion, bestgain

# Gets an empty node
#
def newnode():
    node = {
            'type' : 'node',
			'id' : 0,
            'truechild' : None,
            'falsechild' : None,
            'question' : None,
            'giniindex' : 0.0,
            'infogain' : 0.0
    }
    return node.copy()

# Gets an empty leaf
#
def newleaf():
    leaf = {
            'type' : 'leaf',
			'id' : 0,
            'classes' : None,
            'prediction' : None
    }
    return leaf.copy()

# Builds the tree using CART
#
def buildtree(data, maxdepth = 10, minsplitsize = 2, minleafsize = 1):
    def getprediction(classes):
        bestclass = None
        bestscore = 0
        for c in classes:
            if classes[c] > bestscore:
                bestscore = classes[c]
                bestclass = c
        return bestclass


    node = newnode()

    dataimpurity = gini(data)
    question, gain = findbestquestion(dataimpurity, data)

    if gain == 0 or maxdepth == 0 or len(data) < minsplitsize:
        leaf = newleaf()
        leaf['classes'] = countclasses(data)
        leaf['prediction'] = getprediction(leaf['classes'])
        return leaf

    node['question'] = question
    node['infogain'] = gain
    node['giniindex'] = dataimpurity

    truelist, falselist = splitdata(data, question)

    if len(truelist) < minleafsize or len(falselist) < minleafsize:
        leaf = newleaf()
        leaf['classes'] = countclasses(data)
        leaf['prediction'] = getprediction(leaf['classes'])
        return leaf

    node['truechild'] = buildtree(truelist, maxdepth-1, minsplitsize, minleafsize)
    node['falsechild'] = buildtree(falselist, maxdepth-1, minsplitsize, minleafsize)

    return node


# Predict a row without class value
#
def predictrow(treenode, row):
    if treenode['type'] == 'leaf':
        return treenode['prediction']

    # Assume type is node
    question = treenode['question']

    if askquestion(row, question):
        return predictrow(treenode['truechild'], row)
    return predictrow(treenode['falsechild'], row)

# Predicts many rows and returns the results in an array
#
def predict(treenode, data):
    res = []
    for row in data:
        res.append(predictrow(treenode, row))

    return res

# Prints the tree
#
def printtree(node, spacing=""):
    if node['type'] == 'leaf':
        #print(spacing + "Predict ", node['prediction'])
        print(spacing + "Predict ", node['classes'])
        return

    print(spacing + "Is col " + str(node['question'][0]) + "==" + str(node['question'][1]) + '?')

    print(spacing + '---> True:')
    printtree(node['truechild'], spacing + "  ")

    print(spacing + '---> False:')
    printtree(node['falsechild'], spacing + "  ")

