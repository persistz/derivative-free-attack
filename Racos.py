from Components import Instance
from Components import Dimension
import random
import numpy as np

from utils import *
import time
# from keras.preprocessing.image import save_img as imsave
from scipy.misc import imsave

class RacosOptimization:

    def __init__(self, dim, model, img_rows, img_cols, img_rgb, dimsize,
                resize_mode, ss, mt, pn, ub, to=1800):
        # basic parameters
        self.Pop = []             # population set
        self.PosPop = []          # positive sample set
        self.Optimal = []         # the best sample so far
        self.NextPop = []         # the next population set
        self.label = []
        self.dimsize = dimsize
        self.resize_mode = resize_mode
        self.SampleSize = ss       # the instance number of sampling in an iteration
        self.MaxIteration = mt     # the number of iterations
        self.PositiveNum = pn      # the set size of PosPop
        self.UncertainBits = ub    # the dimension size that is sampled randomly
        self.TimeOut = to
        
        self.dimension = dim
        self.region = np.zeros((dim.getSize(),2))
        self.region[:,0] += dim.getMin()   #min
        self.region[:,1] += dim.getMax()   #max
        # model and size of input
        self.model = model
        self.ImgRows = img_rows
        self.ImgCols = img_cols
        self.ImgRgb = img_rgb
        # addition
        self.querys = 0
        self.oriImg = None
        self.oriLabel = None
        self.tarLabel = None
        return

    def Clear(self):
        self.Pop = []
        self.PosPop = []
        self.Optimal = []
        self.NextPop = []
        self.querys = 0
        return

    # Return optimal
    def getOptimal(self):
        return self.Optimal

    # Generate an instance randomly
    def RandomInstance(self, dim, region):
        #completely random
        inst = Instance(dim)
        ins = []
        ins = np.random.uniform(region[0][0], region[0][1], dim.getSize())
        inst.setFeatures(ins)
        return inst

    # reset model
    def ResetModel(self):
        self.region[:,0] = self.dimension.getMin()
        self.region[:,1] = self.dimension.getMax()
        self.label = []
        return

    # Update PosPop list according to new Pop list generated latterly
    def UpdatePosPopAndOptimal(self):
        self.NextPop.sort(key=lambda instance: instance.getFitness())
        self.PosPop, self.Pop = [], []
        for i in range(self.PositiveNum):
            self.PosPop.append(self.NextPop[i])
        for i in range(self.SampleSize):
            self.Pop.append(self.NextPop[self.PositiveNum+i])
        if(self.Optimal.getFitness() > self.PosPop[0].getFitness()):
            self.Optimal = self.PosPop[0].CopyInstance()
        return

    # generate an instance randomly
    def PosRandomInstance(self, dim, region, label, pos):
        ins = Instance(dim)
        ins.CopyFromInstance(pos)
        for i in range(len(label)):
            temp = random.uniform(region[label[i]][0],region[label[i]][1])
            ins.setFeature(label[i], temp)
        return ins

    # Initialize Pop, PosPop and Optimal
    def Initialize(self):
        temp = []
        self.ResetModel()
        batch_size = 150
        self.querys += 150
        for i in range(batch_size):
            ins = []
            ins = self.RandomInstance(self.dimension, self.region)
            temp.append(ins)
        self.RunOnce(temp)
        temp.sort(key=lambda instance: instance.getFitness())
        # initialize PosPop and Pop
        i = 0
        while(i<self.PositiveNum):
            self.PosPop.append(temp[i])
            i += 1
        while(i<self.PositiveNum+self.SampleSize):
            self.Pop.append(temp[i])
            i += 1
        # initialize optimal
        self.Optimal = self.PosPop[0].CopyInstance()
        return

    def Opt(self, ori_img, label, target=None):
        self.oriImg = ori_img
        self.iniLabel = label
        self.tarLabel = target
        self.Clear()
        self.ResetModel()
        time_begin = time.time()
        self.Initialize()
        # self.save_img_after_initialize(ori_img)
        print('initialize:'+str(time.time()-time_begin))
        time_begin = time.time()
        ori_uncertainbits = self.UncertainBits
        time_all = time.time()
        for itera in range(self.MaxIteration - 1):
            if itera%50 == 0:
                print(itera, self.Optimal.getFitness(), time.time()-time_begin, time.time()-time_all)
                # self.save_img_after_indexth_iteration(ori_img, itera)
            print(itera, self.Optimal.getFitness(), time.time()-time_begin, time.time()-time_all)
            if time.time()-time_all > self.TimeOut:
                break
            if self.Optimal.getFitness() == -10000000:
                break
            time_begin = time.time()
            self.NextPop = []
            for sam in range(self.SampleSize):
                self.ResetModel()
                ChosenPos = random.randint(0, self.PositiveNum - 1)
                self.ContinueShrinkModel(self.PosPop[ChosenPos])
                ins = self.PosRandomInstance(self.dimension, self.region,self.label, self.PosPop[ChosenPos])
                self.NextPop.append(ins)
            self.RunOnce(self.NextPop)
            self.querys += self.SampleSize
            self.NextPop = self.NextPop + self.PosPop + self.Pop
            self.UpdatePosPopAndOptimal()
        # self.save_img_after_indexth_iteration(ori_img, itera)
        return

    def ContinueShrinkModel(self, ins):
        opt_number = 0
        while(opt_number<self.UncertainBits):
            ChosenDim = random.randint(0, self.dimension.getSize()-1)
            greater, less, max_, min_ = 0, 0, -255, 255
            stand = ins.getFeature(ChosenDim)
            for i in range(0, self.SampleSize):
                temp = self.Pop[i].getFeature(ChosenDim)
                if temp >= stand:
                    less += 1
                    min_ = temp if (temp<min_) else min_
                else:
                    greater += 1
                    max_ = temp if (temp>max_) else max_
            if greater >= less:
                self.region[ChosenDim][0] = random.uniform(max_, stand)
            else:
                self.region[ChosenDim][1] = random.uniform(stand, min_)
            self.label.append(ChosenDim)
            opt_number+=1
        self.label.sort()
        return

    def getQuerys(self):
        return self.querys

    def RunOnce(self, popSet):
        batch_size = len(popSet)
        batch_input = np.zeros((batch_size, self.ImgRows, self.ImgCols, self.ImgRgb))
        for i in range(batch_size):
            if not self.dimsize == self.ImgRows:
                noise = noise_resize(popSet[i], self.dimsize, self.ImgRows, self.ImgRgb, self.dimension.getMax(), self.resize_mode)
            else:
                noise = np.around(np.array(popSet[i].getFeatures()).reshape(self.ImgRows, self.ImgCols, self.ImgRgb))
            batch_input[i] = noise + self.oriImg
        batch_input = np.clip(batch_input, 0, 255)
        batch_pred = input_to_prediction(batch_input, self.model, batch_size=batch_size)
        for i in range(batch_size):
            popSet[i].setFitness(self.computeDis(batch_pred[i]))
        return

    def computeDis(self, pred):
        n = np.argmax(pred)
        # print("pred", pred)
        # print(pred.shape)
        if self.tarLabel:
            # return -10000000 if (n == self.tarLabel) else (pred[n]-pred[self.tarLabel])
            return -10000000 if (n == self.tarLabel) else (-pred[self.tarLabel])
        else:
            temp = np.argsort(pred)
            # print("temp", temp)
            # return -10000000 if (not n == self.iniLabel) else (pred[temp[-1]]-pred[temp[-2]])
            return -10000000 if (not n == self.iniLabel) else (-pred[temp[-2]])

    # save_img and save_ini
    # for generate the examples imgs in the paper
    def save_img_after_indexth_iteration(self, ori_img, index):
        # noise = noise_resize(self.Optimal, self.dimsize, self.ImgRows, self.ImgRgb, self.dimension.getMax(), self.resize_mode)
        noise = np.array(self.Optimal.getFeatures()).reshape(self.ImgRows, self.ImgCols, self.ImgRgb)
        noise = np.around(noise)
        temp = noise.astype('uint8')
        imsave('./demo/perturbation/'+str(index)+'.png', temp)

        gen_img = noise+ori_img
        gen_img = np.clip(gen_img, 0, 255).astype('uint8')
        imsave('./demo/example/'+str(index)+'.png', gen_img)

    def save_img_after_initialize(self, ori_img):
        for i in range(self.PositiveNum):
            # noise = noise_resize(self.PosPop[i], self.dimsize, self.ImgRows, self.ImgRgb, self.dimension.getMax(), self.resize_mode)
            noise = np.array(self.PosPop[i].getFeatures()).reshape(self.ImgRows, self.ImgCols, self.ImgRgb)
            noise = np.around(noise)
            temp = noise.astype('uint8')
            imsave('./demo/perturbation/'+str(i)+'_1.png', temp)
            gen_img = noise+ori_img
            gen_img = np.clip(gen_img, 0, 255).astype('uint8')
            imsave('./demo/example/'+str(i)+'_1.png', gen_img)
        for i in range(self.SampleSize):
            # noise = noise_resize(self.Pop[i], self.dimsize, self.ImgRows, self.ImgRgb, self.dimension.getMax(), self.resize_mode)
            noise = np.array(self.Pop[i].getFeatures()).reshape(self.ImgRows, self.ImgCols, self.ImgRgb)
            noise = np.around(noise)
            temp = noise.astype('uint8')
            imsave('./demo/perturbation/'+str(i)+'_sam1.png', temp)
            gen_img = noise+ori_img
            gen_img = np.clip(gen_img, 0, 255).astype('uint8')
            imsave('./demo/example/'+str(i)+'_sam1.png', gen_img)

# def find_k_max(pred, k):
#     temp = pred.copy()
#     temp = np.argsort(temp)
#     for i in range(k):
#         temp_label = temp[0][-1-i]
#         print(temp_label, pred[0][temp_label])
#     return temp[0][-1-k]