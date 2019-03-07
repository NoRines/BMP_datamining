from random import randrange
from bmp_decisiontree import buildtree, mergeleafs, predictrow

# Gets a random subsample from the dataset
#
def subsample(data, samplesize):
    sample = []
    numsamples = int(len(data) * samplesize)
    while len(sample) < numsamples:
        i = randrange(len(data))
        sample.append(data[i])
    return sample


# Creates a random forest
#
def buildrandomforest(data, ntrees, nattribs, minsize, samplesize, maxdepth):
    forest = []
    for i in range(ntrees):
        sample = subsample(data, samplesize)
        tree = buildtree(sample, maxdepth, 2, minsize, nattribs)
        mergeleafs(tree)
        forest.append(tree)

    return forest


def baggingpredict(forest, row, classes):
    predictions = [predictrow(tree, row) for tree in forest]
    pcouts = {}

    for pred in predictions:
        for c in classes:
            if pred == c:
                if c not in pcouts:
                    pcouts[c] = 0
                pcouts[c] += 1

    bestpred = None
    bestcount = 0
    for c in pcouts:
        if bestcount < pcouts[c]:
            bestpred = c
            bestcount = pcouts[c]

    return bestpred

def predictforest(forest, data, classes):
    res = []
    for row in data:
        res.append(baggingpredict(forest, row, classes))
    return res
