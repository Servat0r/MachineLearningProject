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
    
    def split (dataset, splitPercentage):
        trainset = list()
        train_size = splitPercentage * len(dataset)
        dataset_copy = list(dataset)
        while len(trainset) < train_size:
            index = randrange(len(dataset_copy))
            trainset.append(dataset_copy.pop(index))
        return trainset, dataset_copy



# test cross validation split
seed()
dataset = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
#dataset = [["pippo","pluto","paperino"],"topolino",["minnie","paperoga"],"de paperoni"]
numFolds=3
folds = KFoldValidation.split(dataset, numFolds)
for i in range(len(folds)):
    print(folds[i])
train,test=Holdout.split(dataset,0.20)
print("Holdout")
print(train)
print(test)