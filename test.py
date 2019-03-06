from bmp_decisiontree import buildtree, predict, readdataset, countclasses, mergeleafs
from bmp_graphtree import treetograph
import random


dataset = readdataset('data/adult_wo_fnlwgt.csv')

classes = countclasses(dataset)
print(classes)
print(float(classes['<=50K']) / len(dataset))

#random.shuffle(dataset)


split = int(len(dataset) * 0.7)

train = dataset[0 : split]
test = dataset[split : len(dataset)]


tree = buildtree(train, minsplitsize=200, minleafsize=100, maxdepth=13)

mergeleafs(tree)

#attribnames = ['age','education-num','marital-status','hours-per-week','native-country','Class']
#attribnames = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'hours-per-week', 'native-country', 'Class']
attribnames = ['age', 'workclass', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'hours-per-week', 'native-country', 'Class']

treetograph(tree, 'data/adult', attribnames)


testx = [row[0:-1] for row in test]
testy = [row[-1] for row in test]


predy = predict(tree, testx)

correct = 0
for i in range(len(testy)):
    if testy[i] == predy[i]:
        correct += 1

print(correct, '/', len(testy))
print(float(correct) / len(testy))
