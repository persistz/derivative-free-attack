#Instance class, each sample is an instance
import copy

class Instance:

    __feature = []        #feature value in each dimension
    __fitness = 0         #fitness of objective function under those features

    def __init__(self, dim):
        self.__feature = range(dim.getSize())
        self.__fitness = 0
        self.__dimension = dim

    #return feature value in index-th dimension
    def getFeature(self, index):
        return self.__feature[index]

    #return features of all dimensions
    def getFeatures(self):
        return self.__feature

    #set feature value in index-th dimension
    def setFeature(self, index, v):
        self.__feature[index] = v

    #set features of all dimension
    def setFeatures(self, v):
        self.__feature = v

    #return fitness under those features
    def getFitness(self):
        return self.__fitness

    #set fitness
    def setFitness(self, fit):
        self.__fitness = fit

    #
    def Equal(self, ins):
        if len(self.__feature) != len(ins.__feature):
            return False
        for i in range(len(self.__feature)):
            if self.__feature[i] != ins.__feature[i]:
                return False
        return True

    #copy this instance
    def CopyInstance(self):
        copy_ = Instance(self.__dimension)
        features = copy.copy(self.__feature)
        copy_.setFeatures(features)
        copy_.setFitness(self.__fitness)
        return copy_

    def CopyFromInstance(self, ins_):
        self.__feature = []
        self.__feature = copy.copy(ins_.getFeatures())


#Dimension class
#dimension message
class Dimension:

    __size = 0       #dimension size
    __max = 0
    __min = 0

    def __init__(self):
        return

    def setDimensionSize(self, s):
        self.__size = s
        return

    def setMax(self, m):
        self.__max = m
        return

    def setMin(self, m):
        self.__min = m
        return

    def getMax(self):
        return self.__max

    def getMin(self):
        return self.__min

    def getSize(self):
        return self.__size