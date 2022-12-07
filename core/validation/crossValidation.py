from random import randrange, seed

class KFoldValidation:

    def __init__(self, numberOfFolds) -> None:
        pass

    def split (dataset, numberOfFolds):
        dataset_split=list() #creo una lista vuota chiamata dataset_split
        dataset_copy = list(dataset) #creo una copia del dataset

        foldSize = int(len(dataset)/numberOfFolds)
        print("foldsize:" + str(foldSize))
        print("numberoffolds:" + str(numberOfFolds))
        for i in range(numberOfFolds):
            print(i)
            singleFold = list()
            while len(singleFold)<foldSize:
                index = randrange(len(dataset_copy))
                singleFold.append(dataset_copy.pop(index))
            dataset_split.append(singleFold)
       
        return dataset_split


class Holdout:
    def __init__(self) -> None:
        pass
    
    def split (self, dataset, splitPercentage, validationSplitPercentage=0):
        trainSet = list()
        validationSet=list()
        trainSetSize = splitPercentage * len(dataset)
        testSet = list(dataset)
        while len(trainSet) < trainSetSize:
            index = randrange(len(testSet))
            trainSet.append(testSet.pop(index))
        if (validationSplitPercentage >0):
            validationSetSize = validationSplitPercentage * len(trainSet)
            while len(validationSet)<validationSetSize:
                 index = randrange(len(trainSet))
                 validationSet.append(trainSet.pop(index))

        return trainSet, testSet, validationSet



# test cross validation split
seed(1)
dataset = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
numFolds=3
folds = KFoldValidation.split(dataset, numFolds)
for i in range(len(folds)):
    print(folds[i])
train,test,validation =Holdout.split(Holdout,dataset,0.70)
print("Holdout")
print(train)
print(test)
print(validation)