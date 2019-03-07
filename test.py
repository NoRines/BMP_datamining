from bmp_decisiontree import buildtree, predict, readdataset, countclasses, mergeleafs
from bmp_randomforest import buildrandomforest, predictforest
from bmp_graphtree import treetograph
from math import sqrt

dataset = readdataset('data/adult_wo_fnlwgt.csv')

split = int(len(dataset) * 0.7)

train = dataset[0:split]
test = dataset[split:len(dataset)]

testx = [row[0:-1] for row in test]
testy = [row[-1] for row in test]

ntrees = 100
maxdepth = 7
minsize = 400
nattribs = 7

forest = buildrandomforest(train, ntrees, nattribs, minsize, 1.0, maxdepth)

res = predictforest(forest, testx, ['<=50K', '>50K'])


correct = 0
for i in range(len(testy)):
    if testy[i] == res[i]:
        correct += 1


print(correct, '/', len(testy))
print(float(correct) / len(testy))

