#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: emanuele
"""

#for nn
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.backends.cudnn as cudnn

from PIL import Image
#import tensorboard
from torch.utils.tensorboard import SummaryWriter

import wandb


# for data visualization
import matplotlib.pyplot as plt

#to set network architecture
import torch.nn as nn
import torch.nn.functional as F

import psutil
from inspect import currentframe



#classes for INaturalist datset customized
import os.path
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
from torchvision.datasets.vision import VisionDataset

import operator


#Whatever you want to do with shells like running an application, copying files etc., you can do with subprocess. It has run function which does it for you!
import subprocess 
"""
#accelerate the training
from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter
"""

#packages iclude by default (don't have to manually install in your virtual env)

#manage the input of values from outside the script 
import argparse
#manage the folder creation
import os
import time

#deep copy mutable object
import copy

import math

import random

from collections import Counter

torch.set_printoptions(precision=17)

#%% CHECKING CLASS
#load the dataset and compute mean and std for the standardization of its element (by now only implemented for cifar10 dataset)
#torch.set_default_tensor_type(torch.DoubleTensor)


class DatasetMeanStd:
    
    def __init__(self, DatasetName, ClassesDict):
        """
        This class is a tool to compute mean and std to standardise (or check) your dataset
        the init function load the training and test dataset 
        Parameters
        ----------
        DatasetName : string
        this is a string that encode the dataset that will be used

        Returns
        -------
        None.

        """
        
        

        self.DatasetName = DatasetName
        self.ClassesList = ClassesDict.keys()
        self.transform = transforms.ToTensor()
        if(self.DatasetName=='CIFAR10'):
            self.train_data = datasets.CIFAR10(root = self.params['DataFolder'], train = True, download = True, transform = self.transform)
            self.test_data = datasets.CIFAR10(root = self.params['DataFolder'], train = False, download = True, transform = self.transform)
        elif(self.DatasetName=='MNIST'):
            self.train_data = datasets.MNIST(root = 'data_nobackup', train = True, download = True, transform = self.transform)
            self.test_data = datasets.MNIST(root = 'data_nobackup', train = False, download = True, transform = self.transform) 
        elif(self.DatasetName=='CIFAR100'):
            self.train_data = datasets.CIFAR100(root = self.params['DataFolder'], train = True, download = True, transform = self.transform)
            self.test_data = datasets.CIFAR100(root = self.params['DataFolder'], train = False, download = True, transform = self.transform)
    #TODO: include also the option to calculate mean and std for test sets and MNIST dataset
    
    def Mean(self):
        """
        Compute the mean of the dataset (only Cifar10 for now) for the standardization (image vectors normalization)
        Returns
        -------
        list
            mean value for each channel

        """
        
        
        
        if (self.DatasetName == 'CIFAR10'):
            imgs = [item[0] for item in self.train_data if item[1] in self.ClassesList]  # item[0] and item[1] are image and its label
            imgs = torch.stack(imgs, dim=0).numpy()
            
            # calculate mean over each channel (r,g,b)
            mean_r = imgs[:,0,:,:].mean()
            mean_g = imgs[:,1,:,:].mean()
            mean_b = imgs[:,2,:,:].mean()   

            return [mean_r, mean_g, mean_b]
        elif (self.DatasetName == 'CIFAR100'):
            imgs = [item[0] for item in self.train_data if item[1] in self.ClassesList]  # item[0] and item[1] are image and its label
            imgs = torch.stack(imgs, dim=0).numpy()
            
            # calculate mean over each channel (r,g,b)
            mean_r = imgs[:,0,:,:].mean()
            mean_g = imgs[:,1,:,:].mean()
            mean_b = imgs[:,2,:,:].mean()   

            return [mean_r, mean_g, mean_b]
    def Std(self):
        
        """
        Compute the std of the dataset (only Cifar10 for now) for the standardization (image vectors normalization)

        Returns
        -------
        list
            std value for each channel

        """
        
        if (self.DatasetName == 'CIFAR10'):
            imgs = [item[0] for item in self.train_data] # item[0] and item[1] are image and its label
            imgs = torch.stack(imgs, dim=0).numpy()
            
            # calculate std over each channel (r,g,b)
            std_r = imgs[:,0,:,:].std()
            std_g = imgs[:,1,:,:].std()
            std_b = imgs[:,2,:,:].std()  
            
            return(std_r, std_g, std_b)
        elif (self.DatasetName == 'CIFAR100'):
            imgs = [item[0] for item in self.train_data] # item[0] and item[1] are image and its label
            imgs = torch.stack(imgs, dim=0).numpy()
            
            # calculate std over each channel (r,g,b)
            std_r = imgs[:,0,:,:].std()
            std_g = imgs[:,1,:,:].std()
            std_b = imgs[:,2,:,:].std()  
            
            return(std_r, std_g, std_b)           
            


#%% DEFINING TIMES CLASS
class Define:
    def __init__(self, params, n_epochs, NSteps, NBatches, StartPoint, PreviousTimes):
        """
        This class contain methods to create list of logarithmically equispced times, and the ones for the correlations computation


        Parameters
        ----------
        params : dict 
            store all the relevant parameter defined in the main code
        n_epochs : int
            express the total number of epoches
        NSteps : int
            express the number of the final time arrow (this arrow will be used to set the times where to evaluate state of the training (store train/valuation/(test) performances))
        NBatches : int
            Number of batches in the dataset (fixed by the dataset size and batch size)
        StartPoint : int
            starting point for the beginning of the simulation (different for new/retrieved runs)
        PreviousTimes : array
            array of previous times to be attached (in the RETRIEVE mode) at the beginning of the time vector

        Returns
        -------
        None.

        """
        self.params = params.copy()
        self.n_epochs = n_epochs
        self.NSteps = NSteps
        self.NBatches = NBatches
        self.PreviousTimes = PreviousTimes
            
        #we differenciate the 2 cases (because we cannot use as start point log(0))
        if (self.params['StartMode']=='BEGIN'):
            self.StartPoint = StartPoint #in this case we substitute into np.logspace() (below) directly the input (since it is 0) 2**0=1
        if (self.params['StartMode']=='RETRIEVE'):
            self.StartPoint = np.log2(StartPoint) #here instead we substitute into np.logspace() (below) the log(input), where the input correspond to the end (MaxStep) of the last simulation 
        
    #this method return an array of Nsteps dimension with logaritmic equispaced steps : we use it for stocastic algortim as SGD and PCNSGD
    def StocasticTimes(self):
        '''
        define the time time vector for the evaluation stops (stochastic case)
        NOTE: in case of RETRIEVE mode the new times will be equispaced but, in general, with a different spacing between consecutive times with respect to the old vector
        Returns
        -------
        Times : numpy array
            return the logarithmic equispaced steps for stochastic algorithms (as SGD).

        '''    
        MaxStep = self.n_epochs*self.NBatches #the last factor is due to the fact that in the PCNSGD we trash some batches (so the number of steps is slightly lower); 0.85 is roughtly set thinking that batch size, also in the unbalance case will be chosen to not esclude more than 15% of batches
        Times = np.logspace(self.StartPoint, np.log2(MaxStep),num=self.NSteps, base=2.) 
        Times = np.rint(Times).astype(int)   
        
        if (self.params['StartMode']=='BEGIN'):
            for ind in range(0,2): #put the initial times linear to store initial state
                Times[ind] = ind+1    
        
        for steps in range (0, self.NSteps-1):
            while Times[steps] >= Times[steps+1]:
                Times[steps+1] = Times[steps+1]+1
        
        if (self.params['StartMode']=='RETRIEVE'): #in case of continuing previous simulation we concatenate the new times sequence at the end of the old one
            Times = np.concatenate((self.PreviousTimes,Times[1:]), axis=0)


        #here we just reproduce the same time vector to compare the single piece simulation withe the interrupt one (to test the checkpoint)
        if (self.params['CheckMode']=='ON'):
            MaxStep = self.params['n_epochsComp']*self.NBatches
            Times = np.logspace(0, np.log2(MaxStep),num=self.params['NStepsComp'], base=2.) 
            Times = np.rint(Times).astype(int)
       
            for ind in range(0,4): #put the initial times linear to store initial state
                Times[ind] = ind+1    
            
            for steps in range (0, self.NSteps-1):
                while Times[steps] >= Times[steps+1]:
                    Times[steps+1] = Times[steps+1]+1
                
        return Times
          
        
 
    #if we are using a full batch approach we fix the equispaced times with the numbers of epoches
    def FullBatchTimes(self):       
        '''
        define the time vector for the evaluation stops (full batch case)
        NOTE: in case of RETRIEVE mode the new times will be equispaced but, in general, with a different spacing between consecutive times with respect to the old vector
        Returns
        -------
        Times : numpy array
            return the logarithmic equispaced steps for full batch algorithms (as SGD)..

        '''
        MaxStep = self.n_epochs
        Times = np.logspace(self.StartPoint, np.log2(MaxStep),num=self.NSteps, base=2.) 
        Times = np.rint(Times).astype(int)

        if (self.params['StartMode']=='BEGIN'):        
            for ind in range(0,4): #put the initial times linear to store initial state
                Times[ind] = ind+1    
        
        for steps in range (0, self.NSteps-1):
            while Times[steps] >= Times[steps+1]:
                Times[steps+1] = Times[steps+1]+1

        if (self.params['StartMode']=='RETRIEVE'): #in case of continuing previous simulation we concatenate the new times sequence at the end of the old one
            print('previous times are ', self.PreviousTimes, 'new ones ',  Times)
            Times = np.concatenate((self.PreviousTimes,Times[1:]), axis=0)                
            print('the- concatenation of the 2 ', Times)
            
            
        #here we just reproduce the same time vector to compare the single piece simulation withe the interrupt one (to test the checkpoint)
        if (self.params['CheckMode']=='ON'):
            MaxStep = self.params['n_epochsComp']
            Times = np.logspace(0, np.log2(MaxStep),num=self.params['NStepsComp'], base=2.) 
            Times = np.rint(Times).astype(int)
       
            for ind in range(0,4): #put the initial times linear to store initial state
                Times[ind] = ind+1    
            
            for steps in range (0, self.NSteps-1):
                while Times[steps] >= Times[steps+1]:
                    Times[steps+1] = Times[steps+1]+1
            
            
        return Times

    
    #TODO: adapt correlation times to the logic of RETRIEVE
    #setting the times for the computation of correlations:
    #i set tw log-equispaced on the bigger interval possible and t log equispaced in a sub interval whose size is under the difference (MaxStep-last tw)
    def CorrTimes(self, Ntw, Nt, tw, t):
        """
        This method combine the 2 vectors (tw and t) into a 2-D matrix to express correlation times
        NOTE: correlation measure on the last version is not stable yet
        Parameters
        ----------
        Ntw : int
            number of first times for the correlation computation
        Nt : int
            number of second times for the correlation computation.
        MaxStep : int
            Number of maximum steps associated with the fixed number of epoches of the simulation.
        tw : array 
            array of starting points.
        t : array
            array of second points for the 2 point correlation computation.

        Returns
        -------
        CorrTimes : 2-D array
            matrix of 2-point correlation times.

        """
        
        
        #defining the correlation times matrix
        CorrTimes = np.zeros((Ntw, Nt))
        
        for i in range(0, Ntw):
            for j in range(0, Nt):
                CorrTimes[i][j] = tw[i] + t[j]
        return CorrTimes
    
    #correlation are calculated between 2 times, this function return the list of starting times
    def FirstCorrTimes(self, Ntw, MaxStep):
        """
        Create the vector of starting points for the 2-point correlation computation
        
        Parameters
        ----------
        Ntw : int
            number of first times for the correlation computation.
        MaxStep : int
            Number of maximum steps associated with the fixed number of epoches of the simulation..

        Returns
        -------
        None.

        """
        
        tw =  np.logspace(self.StartPoint, np.log2(MaxStep*0.5),num=(Ntw +1), base=2.) 
        tw = np.rint(tw).astype(int)
        #shift forward the equal sizes
        for steps in range (0, Ntw):
            while tw[steps] >= tw[steps+1]:
                tw[steps+1] = tw[steps+1]+1
        
        return tw
        
    #correlation are calculated between 2 times, this function return the list of arriving (second) times
    def SecondCorrTimes(self,Ntw, Nt, tw, MaxStep, spacing_mode = 'log'):     
        """
        Create the vector of second points for the 2-point correlation computation

        Parameters
        ----------
        Ntw : int
            number of first times for the correlation computation
        Nt : int
            number of second times for the correlation computation.

        tw : array 
            array of starting points.
            
        MaxStep : int
            Number of maximum steps associated with the fixed number of epoches of the simulation.
            
        spacing_mode: string
            the spacing mode: linear o logharithmic (default)
        Returns
        -------
        None.

        """
        if (spacing_mode == 'log'):
            #saving 0 and 1 as first ts to evaluate correlation and overlap with the same config. and with the time soon after

            t = np.logspace(3, np.log2(MaxStep-tw[Ntw]),num=(Nt +1), base=2.) 
            t = np.rint(t).astype(int)
            t[0] = 0
            t[1] = 1
            
        elif(spacing_mode == 'linear'):

            t = np.linspace(2, MaxStep-tw[Ntw], num = Nt+1, dtype = int)
            t[0] = 0
            t[1] = 1
        else:
            print('Invalid spacing_mode given as input to SecondCorrTimes function', file=self.params['WarningFile'])
            
        #shift forward the equal sizes
        for steps in range (0, Nt):
            while t[steps] >= t[steps+1]:
                t[steps+1] = t[steps+1]+1
        return t
           

#%% VARIABLE CLASS
#start creating a class which contain all variable used in the others, so these last ones can inherethe them
class NetVariables:
    """
    This class is the container of all the relevant variables that will be stored on files
    """
    #we initialize variable using none as default value to set empty variable because is not good to use directly [] (mutable object in jeneral as default value (deepen this concept)) 
    #so we set None as default value for all variables we don't want to pass
    def __init__(self, params, TrainLoss = None, TestLoss = None, ValidLoss = None , TrainAcc = None ,TestAcc = None, ValidAcc = None , WeightNorm = None, GradientNorm = None,
                 TP = None, TN = None, FP = None, FN = None, TotP = None, Prec = None, Recall = None, FMeasure = None, TrainAngles = None,
                 PCGAngles = None, GradAnglesNormComp = None, StepGradientClassNorm = None, TrainClassesLoss = None, TrainClassesAcc = None, TestAngles = None, TestClassesLoss = None, TestClassesAcc = None, ValidClassesLoss = None, ValidClassesAcc = None,
                 RepresentationClassesNorm = None, ClassesGradientNorm = None,  
                 TwWeights = None, TwoTimesOverlap= None, TwoTimesDistance=None ):
                 
            self.num_classes = params['n_out']
            #self.NSteps = params['NSteps']
            #if you want to have measures at each step (not only the ones corresponding to "Times"' list we modify the above line to modify the shape of the corresponding arraies)
            self.NSteps = params['NSteps']
            self.n_epochs = params['n_epochs']
            self.Ntw = params['Ntw']
            self.Nt = params['Nt']
            

            #prepare variable for accuracy and loss plot
            if TrainLoss is None:
                self.TrainLoss = []
            else:
                 self.TrainLoss = TrainLoss
                 
            if TestLoss is None:
                self.TestLoss = []
            else:
                 self.TestLoss = TestLoss            
            
            if TrainAcc is None:
                self.TrainAcc = []
            else:
                 self.TrainAcc = TrainAcc            
            
            if TestAcc is None:
                self.TestAcc = []
            else:
                 self.TestAcc = TestAcc    

            if ValidLoss is None:
                self.ValidLoss = []
            else:
                 self.ValidLoss = ValidLoss            
            
            if ValidAcc is None:
                self.ValidAcc = []
            else:
                 self.ValidAcc = ValidAcc        

            
            if WeightNorm is None: #MEASURE TO ADD TO RECALL LOGIC
                self.WeightNorm = []
            else:
                 self.WeightNorm = WeightNorm    
            
            if GradientNorm is None: #MEASURE TO ADD TO RECALL LOGIC
                self.GradientNorm = []            
            else:
                 self.GradientNorm = GradientNorm    
            
            #DEFINE SOME VARIABLE CONTAINERS
            #vector for true positive and false positive (for the precision) 
            if TP is None:
                if (self.params['StartMode']=='BEGIN'):
                    self.TP = torch.zeros((self.num_classes , (self.NSteps + 1)))
                elif (self.params['StartMode']=='RETRIEVE'): #restarting in RETRIEVE mode we don't need the measure at step 0 (before the training start)
                    self.TP = torch.zeros((self.num_classes , (self.NSteps)))
            else:
                 self.TP = TP    
            
            if TN is None:
                if (self.params['StartMode']=='BEGIN'):
                    self.TN = torch.zeros((self.num_classes , (self.NSteps + 1)))
                elif (self.params['StartMode']=='RETRIEVE'): #restarting in RETRIEVE mode we don't need the measure at step 0 (before the training start)
                    self.TN = torch.zeros((self.num_classes , (self.NSteps)))
            else:
                 self.TN = TN    
            
            if FP is None:
                if (self.params['StartMode']=='BEGIN'):
                    self.FP = torch.zeros((self.num_classes , (self.NSteps + 1)))
                elif (self.params['StartMode']=='RETRIEVE'): #restarting in RETRIEVE mode we don't need the measure at step 0 (before the training start)
                    self.FP = torch.zeros((self.num_classes , (self.NSteps)))
            else:
                 self.FP = FP    
            
            if FN is None:
                if (self.params['StartMode']=='BEGIN'):
                    self.FN = torch.zeros((self.num_classes , (self.NSteps + 1)))
                elif (self.params['StartMode']=='RETRIEVE'): #restarting in RETRIEVE mode we don't need the measure at step 0 (before the training start)
                    self.FN = torch.zeros((self.num_classes , (self.NSteps)))
            else:
                 self.FN = FN    
            
            if TotP is None:
                self.TotP = torch.zeros((self.num_classes , self.NSteps))
            else:
                 self.TotP = TotP    
            
            if Prec is None:
                self.Prec = torch.zeros((self.num_classes , self.NSteps))
            else:
                 self.Prec = Prec    
            
            if Recall is None:
                self.Recall = torch.zeros((self.num_classes , self.NSteps))
            else:
                 self.Recall = Recall    
            
            if FMeasure is None:
                self.FMeasure = torch.zeros((self.num_classes , self.NSteps))
            else:
                 self.FMeasure = FMeasure    
            
            #angules variables
            #NOTE: IT IS IMPORTANT TO FILL THE ARRAY WITH THE CORRECT TYPE, BECAUSE THIS WILL BE THE DEFAULT DATA TYPE OF THE ARRAY USED; TOSPECIFY EXPLICITLY THE DATA TYPE USE THE OPTION dtype=...
            #we add one more component (n_epochs + 1 instead of n_epochs) because we also store the starting state to check the randomness of representation state at the beginning
            if TrainAngles is None:
                if (self.params['StartMode']=='BEGIN'):
                    self.TrainAngles = np.full((int(self.num_classes*(self.num_classes-1)/2), (self.NSteps + 1)), 1000.)
                elif (self.params['StartMode']=='RETRIEVE'): #restarting in RETRIEVE mode we don't need the measure at step 0 (before the training start)
                    self.TrainAngles = np.full((int(self.num_classes*(self.num_classes-1)/2), (self.NSteps)), 1000.)                
            else:
                 self.TrainAngles = TrainAngles    
            
            if PCGAngles is None:
                if (self.params['StartMode']=='BEGIN'):
                    self.PCGAngles = np.full((int(self.num_classes*(self.num_classes-1)/2), (self.NSteps + 1)), 1000.)
                elif (self.params['StartMode']=='RETRIEVE'): #restarting in RETRIEVE mode we don't need the measure at step 0 (before the training start)
                    self.PCGAngles = np.full((int(self.num_classes*(self.num_classes-1)/2), (self.NSteps)), 1000.)
            else:
                 self.PCGAngles = PCGAngles   
                 
            if GradAnglesNormComp is None:
                self.GradAnglesNormComp = np.full((int(self.num_classes*(self.num_classes-1)/2), (self.NSteps)), 1000.)
            else:
                self.GradAnglesNormComp = GradAnglesNormComp
                
                 
            if StepGradientClassNorm is None:
                if (self.params['StartMode']=='BEGIN'):
                    self.StepGradientClassNorm = np.full((self.num_classes, (self.NSteps+1)), 1000.)
                elif (self.params['StartMode']=='RETRIEVE'):
                    self.StepGradientClassNorm = np.full((self.num_classes, (self.NSteps)), 1000.)
            else:   
                self.StepGradientClassNorm = StepGradientClassNorm
            
            if TrainClassesLoss is None:
                if (self.params['StartMode']=='BEGIN'):
                    self.TrainClassesLoss = np.zeros((self.num_classes, (self.NSteps + 1)))
                elif (self.params['StartMode']=='RETRIEVE'):
                    self.TrainClassesLoss = np.zeros((self.num_classes, (self.NSteps)))
            else:
                 self.TrainClassesLoss = TrainClassesLoss    
            
            if TrainClassesAcc is None:
                if (self.params['StartMode']=='BEGIN'):
                    self.TrainClassesAcc = np.full((int(self.num_classes), (self.NSteps+1)), 1000.)
                elif (self.params['StartMode']=='RETRIEVE'):
                    self.TrainClassesAcc = np.full((int(self.num_classes), (self.NSteps)), 1000.)
            else:
                 self.TrainClassesAcc = TrainClassesAcc    
                 
            #same measures for test set
            if TestAngles is None:
                self.TestAngles = np.full((int(self.num_classes*(self.num_classes-1)/2), self.NSteps), 10.)
            else:
                 self.TestAngles = TestAngles    
            
            if TestClassesLoss is None:
                if (self.params['StartMode']=='BEGIN'):
                    self.TestClassesLoss = np.zeros((self.num_classes, (self.NSteps + 1)))
                elif (self.params['StartMode']=='RETRIEVE'):
                    self.TestClassesLoss = np.zeros((self.num_classes, (self.NSteps)))
            else:
                 self.TestClassesLoss = TestClassesLoss    
            
            if TestClassesAcc is None:
                if (self.params['StartMode']=='BEGIN'):
                    self.TestClassesAcc = np.full((int(self.num_classes), (self.NSteps+1)), 1000.)
                elif (self.params['StartMode']=='RETRIEVE'):
                    self.TestClassesAcc = np.full((int(self.num_classes), (self.NSteps)), 1000.)
            else:
                 self.TestClassesAcc = TestClassesAcc    
            
            if ValidClassesLoss is None:
                if (self.params['StartMode']=='BEGIN'):
                    self.ValidClassesLoss = np.zeros((self.num_classes, (self.NSteps + 1)))
                elif (self.params['StartMode']=='RETRIEVE'):
                    self.ValidClassesLoss = np.zeros((self.num_classes, (self.NSteps)))
            else:
                 self.ValidClassesLoss = ValidClassesLoss  

            if ValidClassesAcc is None:
                if (self.params['StartMode']=='BEGIN'):
                    self.ValidClassesAcc = np.full((int(self.num_classes), (self.NSteps+1)), 1000.)
                elif (self.params['StartMode']=='RETRIEVE'):
                    self.ValidClassesAcc = np.full((int(self.num_classes), (self.NSteps)), 1000.)
            else:
                 self.ValidClassesAcc = ValidClassesAcc             
            
            if RepresentationClassesNorm is None:
                if (self.params['StartMode']=='BEGIN'):
                    self.RepresentationClassesNorm = np.zeros((self.num_classes, (self.NSteps + 1)))
                elif (self.params['StartMode']=='RETRIEVE'):
                    self.RepresentationClassesNorm = np.zeros((self.num_classes, (self.NSteps)))
            else:
                 self.RepresentationClassesNorm = RepresentationClassesNorm    
            
            if ClassesGradientNorm is None:
                self.ClassesGradientNorm = np.zeros((self.n_epochs, self.num_classes))
            else:
                 self.ClassesGradientNorm = ClassesGradientNorm    
            
   

            #define the list where we will put the copies of the weights
            if (params['SphericalConstrainMode']=='ON'):
                if TwWeights is None:     
                    self.TwWeights = []
                if TwoTimesOverlap is None:     
                    self.TwoTimesOverlap = torch.zeros((self.Ntw, self.Nt))       
                if TwoTimesDistance is None:     
                    self.TwoTimesDistance = torch.zeros((self.Ntw, self.Nt))  
        







#%% USEFUL METHOD FOR THE NETWORK CLASSES
#HERE WE DEFINE A CLASSES THAT COINTAIN USEFUL TOOL FOR THE NET CLASSES


class OrthoInit:
    """
    #ORTHOGONAL CONDITION: since we are using a more extended CNN we add the possibility to initialize the weight according to the method proposed in the following article:
    #Dynamical Isometry and a Mean Field Theory of CNNs: How to Train 10,000-Layer Vanilla Convolutional Neural Networks
    #below some methods used for this purpose  
    """
    def __init__(self):
        pass
    
    ######################################Generating 2D orthogonal initialization kernel####################################
    #generating uniform orthogonal matrix
    def _orthogonal_matrix(self, dim):
        a = torch.zeros((dim, dim)).normal_(0, 1)
        q, r = torch.linalg.qr(a)
        d = torch.diag(r, 0).sign()
        diag_size = d.size(0)
        d_exp = d.view(1, diag_size).expand(diag_size, diag_size)
        q.mul_(d_exp)
        return q
    
    #generating orthogonal projection matrix,i.e. the P,Q of Algorithm1 in the original
    def _symmetric_projection(self, n):
        """Compute a n x n symmetric projection matrix.
        Args:
          n: Dimension.
        Returns:
          A n x n orthogonal projection matrix, i.e. a matrix P s.t. P=P*P, P=P^T.
        """
        q = self._orthogonal_matrix(n)
        # randomly zeroing out some columns
        # mask = math.cast(random_ops.random_normal([n], seed=self.seed) > 0,
        # #                      self.dtype)
        mask = torch.randn(n)
    
        c = torch.mul(mask,q)
        U,_,_= torch.svd(c)
        U1 = U[:,0].view(len(U[:,0]),1)
        P = torch.mm(U1,U1.t())
        P_orth_pro_mat = torch.eye(n)-P
        return P_orth_pro_mat
    
    #generating block matrix the step2 of the Algorithm1 in the original
    def _block_orth(self, p1, p2):
        """Construct a 2 x 2 kernel. Used to construct orthgonal kernel.
        Args:
          p1: A symmetric projection matrix (Square).
          p2: A symmetric projection matrix (Square).
        Returns:
          A 2 x 2 kernel [[p1p2,         p1(1-p2)],
                          [(1-p1)p2, (1-p1)(1-p2)]].
        Raises:
          ValueError: If the dimensions of p1 and p2 are different.
        """
        if p1.shape != p2.shape:
            raise ValueError("The dimension of the matrices must be the same.")
        kernel2x2 = {}#Block matrices are contained by a dictionary
        eye = torch.eye(p1.shape[0])
        kernel2x2[0, 0] = torch.mm(p1, p2)
        kernel2x2[0, 1] = torch.mm(p1, (eye - p2))
        kernel2x2[1, 0] = torch.mm((eye - p1), p2)
        kernel2x2[1, 1] = torch.mm((eye - p1), (eye - p2))
    
        return kernel2x2
    
    #compute convolution operator of equation2.17 in the original
    def _matrix_conv(self, m1, m2):
        """Matrix convolution.
        Args:
          m1: A k x k dictionary, each element is a n x n matrix.
          m2: A l x l dictionary, each element is a n x n matrix.
        Returns:
          (k + l - 1) * (k + l - 1) dictionary each element is a n x n matrix.
        Raises:
          ValueError: if the entries of m1 and m2 are of different dimensions.
        """
    
        n = m1[0, 0].shape[0]
        if n != m2[0, 0].shape[0]:
            raise ValueError("The entries in matrices m1 and m2 "
                             "must have the same dimensions!")
        k = int(np.sqrt(len(m1)))
        l = int(np.sqrt(len(m2)))
        result = {}
        size = k + l - 1
        # Compute matrix convolution between m1 and m2.
        for i in range(size):
            for j in range(size):
                result[i, j] = torch.zeros(n,n)
                for index1 in range(min(k, i + 1)):
                    for index2 in range(min(k, j + 1)):
                        if (i - index1) < l and (j - index2) < l:
                            result[i, j] += torch.mm(m1[index1, index2],
                                                            m2[i - index1, j - index2])
        return result
    
    def _dict_to_tensor(self, x, k1, k2):
        """Convert a dictionary to a tensor.
        Args:
          x: A k1 * k2 dictionary.
          k1: First dimension of x.
          k2: Second dimension of x.
        Returns:
          A k1 * k2 tensor.
        """
        return torch.stack([torch.stack([x[i, j] for j in range(k2)])
                                for i in range(k1)])
    
    #generating a random 2D orthogonal Convolution kernel
    def _orthogonal_kernel(self, tensor):
        """Construct orthogonal kernel for convolution.
        Args:
          ksize: Kernel size.
          cin: Number of input channels.
          cout: Number of output channels.
        Returns:
          An [ksize, ksize, cin, cout] orthogonal kernel.
        Raises:
          ValueError: If cin > cout.
        """
        ksize = tensor.shape[2]
        cin = tensor.shape[1]
        cout = tensor.shape[0]
        if cin > cout:
            raise ValueError("The number of input channels cannot exceed "
                             "the number of output channels.")
        orth = self._orthogonal_matrix(cout)[0:cin, :]#这就是算法1中的H
        if ksize == 1:
            return torch.unsqueeze(torch.unsqueeze(orth,0),0)
    
        p = self._block_orth(self._symmetric_projection(cout),
                             self._symmetric_projection(cout))
        for _ in range(ksize - 2):
            temp = self._block_orth(self._symmetric_projection(cout),
                                    self._symmetric_projection(cout))
            p = self._matrix_conv(p, temp)
        for i in range(ksize):
            for j in range(ksize):
                p[i, j] = torch.mm(orth, p[i, j])
        tensor.copy_(self._dict_to_tensor(p, ksize, ksize).permute(3,2,1,0))
        return tensor
    
    #defining 2DConvT orthogonal initialization kernel
    def ConvT_orth_kernel2D(self,tensor):
        ksize = tensor.shape[2]
        cin = tensor.shape[0]
        cout = tensor.shape[1]
        if cin > cout:
            raise ValueError("The number of input channels cannot exceed "
                             "the number of output channels.")
        orth = self._orthogonal_matrix(cout)[0:cin, :]  # 这就是算法1中的H
        if ksize == 1:
            return torch.unsqueeze(torch.unsqueeze(orth, 0), 0)
    
        p = self._block_orth(self._symmetric_projection(cout),
                        self._symmetric_projection(cout))
        for _ in range(ksize - 2):
            temp = self._block_orth(self._symmetric_projection(cout),
                               self._symmetric_projection(cout))
            p = self._matrix_conv(p, temp)
        for i in range(ksize):
            for j in range(ksize):
                p[i, j] = torch.mm(orth, p[i, j])
        tensor.copy_(self._dict_to_tensor(p, ksize, ksize).permute(2, 3, 1, 0))
        return tensor
    #Call method
    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.weight.shape[0] > m.weight.shape[1]:
                    self._orthogonal_kernel(m.weight.data)
                    m.bias.data.zero_()
                else:
                    nn.init.orthogonal_(m.weight.data)
                    m.bias.data.zero_()
    
            elif isinstance(m, nn.ConvTranspose2d):
                if m.weight.shape[1] > m.weight.shape[0]:
                    self.ConvT_orth_kernel2D(m.weight.data)
                   # m.bias.data.zero_()
                else:
                    nn.init.orthogonal_(m.weight.data)
                   # m.bias.data.zero_()
    
               # m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                #m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.zero_()
    '''
    Algorithm requires The number of input channels cannot exceed the number of output channels.
     However, some questions may be in_channels>out_channels. 
     For example, the final dense layer in GAN. If counters this case, Orthogonal_kernel is replaced by the common orthogonal init'''
    '''
    for example,
    net=nn.Conv2d(3,64,3,2,1)
    net.apply(Conv2d_weights_orth_init)
    '''
    
    def makeDeltaOrthogonal(self, in_channels=3, out_channels=64, kernel_size=3, gain=torch.Tensor([1])):
        weights = torch.zeros(out_channels, in_channels, kernel_size, kernel_size)
        out_channels = weights.size(0)
        in_channels = weights.size(1)
        if weights.size(1) > weights.size(0):
            raise ValueError("In_filters cannot be greater than out_filters.")
        q = self._orthogonal_matrix(out_channels)
        q = q[:in_channels, :]
        q *= torch.sqrt(gain)
        beta1 = weights.size(2) // 2
        beta2 = weights.size(3) // 2
        weights[:, :, beta1, beta2] = q
        return weights
    #Calling method is the same as the above _orthogonal_kernel
    ######################################################END###############################################################



#%% DEFINE NN ARCHITECTURE

class Net(nn.Module, NetVariables):
    def __init__(self, params):
        """
        Network class: this is a simple toy network  ()MLP with 2 hidden layer 
            In general to understand the architecture of the network is useful to read the forward method below

        Parameters
        ----------
        params : dict
            dict of parameters imported from Main code

        Returns
        -------
        None.

        """
        
        
        
        #super(Net,self).__init__()
        #super(Net, self).__init__(self, params['n_out'], params['NSteps'], params['n_epochs'])
        
        self.params = params.copy()
        
        nn.Module.__init__(self)
        NetVariables.__init__(self, self.params)        
        
        
        # number of hidden nodes in each layer (512)
        hidden_1 = 32
        hidden_2 = 32

        # linear layer (784 -> hidden_1)
        if  (params['Dataset']=='MNIST'):
            self.fc1 = nn.Linear(28*28, hidden_1)
        #NOTE: images in MNIST dataset are b&w photos with 28*28 pixels, CIFAR10 instead contain with colored photos (3 colour channel) with 32*32 pixels 
        elif  (params['Dataset']=='CIFAR10'):
            self.fc1 = nn.Linear(32*32*3, hidden_1)
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(hidden_1,hidden_2)
        # linear layer (n_hidden -> 10)
        self.fc3 = nn.Linear(hidden_2, self.num_classes)
        
        #weights initialization (this step can also be put below in a separate def)
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.xavier_normal_(self.fc3.weight)
        #initialize the bias to 0
        nn.init.constant_(self.fc3.bias, 0)
    #I return from the forward a dictionary to get the output after each layer not only the last one
    #this is useful for example in the inter-classes angles (calculated throught the scalar product between inner layers representation)
    def forward(self,x):
        
        outs = {}
        # flatten image input
        if  (self.params['Dataset']=='MNIST'):
            x = x.view(-1,28*28)
        #NOTE: images in MNIST dataset are b&w photos with 28*28 pixels, CIFAR10 instead contain with colored photos (3 colour channel) with 32*32 pixels 
        elif  (self.params['Dataset']=='CIFAR10'):
            x = x.view(-1,32*32*3)
        # add hidden layer, with relu activation function
        Fc1 = F.relu(self.fc1(x))
        
        outs['l1'] = Fc1
        
        # add hidden layer, with relu activation function
        #Fc2 = F.relu(self.fc2(Fc1))
        #if you want to use a tanh activation function don't use F.tanh wich is deprecated; instead substitute the line above with the one below
        Fc2 = torch.tanh(self.fc2(Fc1))
        
        
        outs['l2'] = Fc2
        
        # add output layer
        Out = self.fc3(Fc2)
        
        outs['out'] = Out
        
        return outs







class ConvNet(nn.Module, NetVariables, OrthoInit):

    def __init__(self, params):
        """
        Network class: this is a prototipe of simple CNN with 2 hidden layer 
            In general to understand the architecture of the network is useful to read the forward method below
    
        Parameters
        ----------
        params : dict
            dict of parameters imported from Main code
    
        Returns
        -------
        None.
    
        """

        self.params = params.copy()
        #this is an example of multiple inherance class
        #we need a single "super" call for each parent class
        """
        super().__init__()
        super().__init__(self.params)
        """
        nn.Module.__init__(self)
        NetVariables.__init__(self, self.params)
        OrthoInit.__init__(self)
        
        if  (self.params['Dataset']=='MNIST'):
            self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.fc = nn.Linear(7*7*32, self.num_classes)
        elif  (self.params['Dataset']=='CIFAR10' or self.params['Dataset']=='INATURALIST' or self.params['Dataset']=='CIFAR100'):  
            
            self.l1=[nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2)]
            if self.params['IGB_flag'] == 'ON':
                self.l1.append(nn.ReLU())
                self.l1.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif self.params['IGB_flag'] == 'OFF':    
                self.l1.append(nn.Tanh())
                self.l1.append(nn.AvgPool2d(kernel_size=2, stride=2))     
                
            self.layer1 = nn.Sequential(*self.l1)
            
            self.l2=[nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5, stride=1, padding=2)]
            if self.params['IGB_flag'] == 'ON':
                self.l2.append(nn.ReLU())
                self.l2.append(nn.MaxPool2d(kernel_size=4, stride=4))
            elif self.params['IGB_flag'] == 'OFF':    
                self.l2.append(nn.Tanh())
                self.l2.append(nn.AvgPool2d(kernel_size=4, stride=4))     
                
            self.layer2 = nn.Sequential(*self.l2)
            


            #note the difference in values from the MNIST case in the following line; it is due to the different image size
            #in fact, a generic image (regardless of the number of channels) with size X*Y changes its extension in the 2 directions in the following way:
            #In the convolutional layer: X -> 1+(X-kernel_size + 2*padding)/stride
            # In the pooling layer: X -> 1+(X-kernel_size)/stride
            self.fc = nn.Linear(4*4*64, self.num_classes)                
            
        self.initialize_weights() #calling the function below to initialize weights
        #self.weights_init() #call the orthogonal initial condition    
    def initialize_weights(self):
        #modules is the structure in which pytorch saves all the layers that make up the network
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            
            

        
    def forward(self, x):
        outs = {}
        
        L1 = self.layer1(x)
        outs['l1'] = L1
        L2 = self.layer2(L1)
        #After pooling or CNN convolution requires connection full connection layer, it is necessary to flatten the multi-dimensional tensor into a one-dimensional,
        #Tensor dimension convolution or after pooling of (batchsize, channels, x, y), where x.size (0) means batchsize value, and finally through x.view (x.size (0), -1) will be in order to convert the structure tensor (batchsize, channels * x * y), is about (channels, x, y) straightened, can then be connected and fc layer
        outs['l2'] = L2
        Out = L2.reshape(L2.size(0), -1)
        Out = self.fc(Out)
        outs['out'] = Out
        outs['pred'] = torch.argmax(Out, dim=1)
        return outs











class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out



class ResNet(nn.Module, NetVariables):
    def __init__(self, block, layers, params, num_classes = 10):
        
        self.params = params.copy()

        nn.Module.__init__(self)
        NetVariables.__init__(self, self.params)        
        self.num_classes = num_classes
        
        #super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        outs = {}
        
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        outs['l2'] = x
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        outs['out'] = x
        outs['pred'] = torch.argmax(x, dim=1)
        return outs















#WARNING: VGG is designed only for CIFAR10 data    

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    
    #'VGG16': [64, 'D3', 64, 'M', 128, 'D4', 128, 'M', 256, 'D4', 256, 'D4', 256, 'M', 512,'D4', 512,'D4', 512, 'M', 512,'D4', 512,'D4', 512, 'M', 'D5'], #with dropout
    #'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512,512, 'M'], #original
    #'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512,512, 'A'], #original but with an averaging pool at the end instead of maxpool
    #'VGG16': [64, 'D3', 64, 'M', 128, 'D4', 128, 'M', 256, 'D4', 256, 'D4', 256, 'M', 512,'D4', 512,'D4', 512, 'M', 512,'D4', 512,'D4', 512, 'A', 'D5'], #original but with an averaging pool at the end instead of maxpool and with dropout
    'VGG16': [64, 'Dp', 64, 'M', 128, 'Dp', 128, 'M', 256, 'Dp', 256, 'Dp', 256, 'M', 512,'Dp', 512,'Dp', 512, 'M', 512,'Dp', 512,'Dp', 512, 'A', 'Dp'], #dropouts dependent from a single parameter (useful for hyper-par optim.) 
    'BENCHMARKVGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module, NetVariables, OrthoInit): 
    #below lines modified to uniform classes input to CNN and MLP case (vgg_name is fixed to 'VGG16')
    #def __init__(self, vgg_name, n_out):
    def __init__(self, params):
        """
        Network class: this is a prototipe of more complex (deep) CNN: the sequence of layers can be chosen from one of the above dict item (cfg) 
            In general to understand the architecture of the network is useful to read the forward method below
        
        Note: all the module defined in Init method will be automatically charged on device (and so will be present on self.model.parameters); 
        This means that if you define modules in Init but don't use them in forward they will have Null grad (this will raise an error during the cloning of Grad) and waste useful memory for unused modules
        Parameters
        ----------
        params : dict
            dict of parameters imported from Main code
    
        Returns
        -------
        None.
    
        """
        #this is an example of multiple inherance class
        #we need a single "super" call for each parent class 
        self.params = params.copy()
        
        """        
        super(VGG, self).__init__()
        super(VGG, self).__init__(self, params['n_out'], params['NSteps'], params['n_epochs'])
        """
        nn.Module.__init__(self)
        NetVariables.__init__(self, self.params)
        OrthoInit.__init__(self)
        
        
        
        #self.features = self._make_layers(cfg[vgg_name])
        self.features = self._make_layers(cfg['VGG16']) #TODO:  SUBSTITUTE 'D' CONFIG WITH 'VGG16' (USED ONLY TO COPY BENCHMARK)
        self.classifier = nn.Linear(512, self.num_classes)
        """
        self.BenchClassifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
        """

        
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                #nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    #m.bias.detach().zero_()
                    nn.init.constant_(m.bias, 0)
        
        #self.weights_init() #call the orthogonal initial condition
        


    def forward(self, x):
        outs = {} 
        L2 = self.features(x)
        outs['l2'] = L2
        Out = L2.view(L2.size(0), -1)
        Out = self.classifier(Out)
        #Out = self.BenchClassifier(Out) #TODO: UNCOMMENT ABOVE LINE AND COMMENT THIS ONE (USED ONLY TO COPY BENCHMARK)
        outs['out'] = Out
        return outs

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                
            elif x=='A':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
                
            elif x == 'D3':
                layers += [nn.Dropout(0.3)]

            elif x == 'D4':
                layers += [nn.Dropout(0.4)]            

            elif x == 'D5':
                layers += [nn.Dropout(0.5)]   
                
            elif x == 'Dp':
                layers += [nn.Dropout(self.params['dropout_p'])] 
                
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           #nn.ReLU(inplace=True)
                           #nn.LeakyReLU(negative_slope=0.1, inplace=False),
                           nn.Tanh()  #put it back after banch runs
                           #,nn.BatchNorm2d(x)   #For now Batch Norm is excluded because it is incompatible with PCNGD, GD, PCNSGD where I forward sample by sample
                           ,nn.GroupNorm(int(x/self.params['group_factor']), x) #put it back after benchmark run
                           #,nn.GroupNorm(int(1), x)
                           ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        #see * operator (Unpacking Argument Lists)
        #The special syntax *args in function definitions in python is used to pass a variable number of arguments to a function. It is used to pass a non-key worded, variable-length argument list.
        #The syntax is to use the symbol * to take in a variable number of arguments; by convention, it is often used with the word args.
        #What *args allows you to do is take in more arguments than the number of formal arguments that you previously defined.
        return nn.Sequential(*layers) 




class VGG_Custom_Dropout(nn.Module, NetVariables, OrthoInit):
    #below lines modified to uniform classes input to CNN and MLP case (vgg_name is fixed to 'VGG16')
    #def __init__(self, vgg_name, n_out):
    def __init__(self, params):

        """
        Network class: this is a prototipe of more complex (deep) CNN: the sequence of layers can be chosen from one of the above dict item (cfg) 
            In general to understand the architecture of the network is useful to read the forward method below
            The only difference with respect to the above VGG is that here you have control on the dropout layer defined by a mask.
            The right dropout is the one of class VGG (a different mask has to be applied on each image of the dataset (both for stochastic and deterministic algorithms)); this constitute an extension used for example to show that with conventional dropout PCNGD monotony prediction breaks (also for small lr)
    
        Parameters
        ----------
        params : dict
            dict of parameters imported from Main code
    
        Returns
        -------
        None.
    
        """        
        #this is an example of multiple inherance class
        #we need a single "super" call for each parent class 
        self.params = params.copy()
        
        """        
        super(VGG, self).__init__()
        super(VGG, self).__init__(self, params['n_out'], params['NSteps'], params['n_epochs'])
        """
        nn.Module.__init__(self)
        NetVariables.__init__(self, self.params)
        OrthoInit.__init__(self)
        
        
        
        #self.features = self._make_layers(cfg[vgg_name])
        self.ModuleDict = self._make_layers(cfg['VGG16'])
        self.classifier = nn.Linear(512, self.num_classes)
        self.mask_dict = {}


        """
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                #nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    #m.bias.detach().zero_()
                    nn.init.constant_(m.bias, 0)
        """
        self.weights_init() #call the orthogonal initial condition
        


    def forward(self, x, Mask_Flag):
        outs = {} 
        #iterate over the ModuleDict substituting where necessary identities appropriately (we use a flag to figure out when to update the masks)
        #flagghetta=0
        
        for key in self.ModuleDict:
            if key.startswith('!'):
                
                if not self.training: #if we are in eval mode the dropout is substitute by a identity layer
                    x = self.ModuleDict[key](x)
                else:
                
                
                    if Mask_Flag==1: #this flag trigger the update of masks 
                        self.mask_dict[key] = torch.distributions.Bernoulli(probs=(1-self.params['dropout_p'])).sample(x.size())
                        self.mask_dict[key] = self.mask_dict[key].to(self.params['device']) #load the mask used for the below tensor multiplication on the same device
                    """
                    if flagghetta==0:
                        print('x prima', x[0][0][0])
                        print('a maschera', self.mask_dict[key][0][0][0])                   
                    """

                    x = x * self.mask_dict[key] * 1/(1-self.params['dropout_p']) #dropout layer
                    
                    """
                    if flagghetta==0:                 
                        print('x dopo', x[0][0][0])
                    flagghetta=1                   
                    """
                    
            else: #for modules different from dropout regular forward
                x = self.ModuleDict[key](x)
        
        L2 = x
        outs['l2'] = L2
        Out = L2.view(L2.size(0), -1)
        Out = self.classifier(Out)
        outs['out'] = Out
        return outs

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        ModuleDict = nn.ModuleDict()
        NumberKey=0
        
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                
            elif x=='A':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
                 
                
            elif x == 'Dp':
                
                ModuleDict[str(NumberKey)] = nn.Sequential(*copy.deepcopy(layers)) 
                NumberKey+=1
                ModuleDict['!'+str(NumberKey)] = nn.Identity()
                NumberKey+=1
                layers = []
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           #nn.ReLU(inplace=True),
                           #nn.LeakyReLU(negative_slope=0.1, inplace=False),
                           nn.Tanh()
                           #,nn.BatchNorm2d(x)   #For now Batch Norm is excluded because it is incompatible with PCNGD, GD, PCNSGD where I forward sample by sample
                           ,nn.GroupNorm(int(x/self.params['group_factor']), x)
                           #,nn.GroupNorm(int(1), x)
                           ]
                in_channels = x
        NumberKey+=1
        ModuleDict[str(NumberKey)] = nn.AvgPool2d(kernel_size=1, stride=1)
        #see * operator (Unpacking Argument Lists)
        #The special syntax *args in function definitions in python is used to pass a variable number of arguments to a function. It is used to pass a non-key worded, variable-length argument list.
        #The syntax is to use the symbol * to take in a variable number of arguments; by convention, it is often used with the word args.
        #What *args allows you to do is take in more arguments than the number of formal arguments that you previously defined.
        return ModuleDict






#below the class for the ResNet18 implementation
#   https://github.com/liao2000/ML-Notebook/blob/main/ResNet/ResNet_PyTorch.ipynb

#the only difference with the usual implementation is that BatchNorm2d are substituited by GroupNorm layer 
#note: as for batchNorm also GroupNorm require an input greater of size 1 to calculate a meaningful mean and std 
#(since we have torch.nn.AdaptiveAvgPool2d(1) in ResNet this translate in self.params['group_factor']>1)


"""
class ResBlock(nn.Module, NetVariables):
    def __init__(self, in_channels, out_channels, params, downsample):
        
        self.params = params.copy()

        nn.Module.__init__(self)
        NetVariables.__init__(self, self.params)
        
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                #nn.BatchNorm2d(out_channels)
                nn.GroupNorm(int(out_channels/self.params['group_factor']), out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        #self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn1 = nn.GroupNorm(int(out_channels/self.params['group_factor']), out_channels)
        #self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.GroupNorm(int(out_channels/self.params['group_factor']), out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = input + shortcut
        return nn.ReLU()(input)



class ResNet18(nn.Module, NetVariables):
    def __init__(self, params, in_channels, resblock, outputs):

        self.params = params.copy()
        
        nn.Module.__init__(self)
        NetVariables.__init__(self, self.params)
        
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            #nn.BatchNorm2d(64),
            nn.GroupNorm(int(64./self.params['group_factor']), 64),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            resblock(64, 64, self.params, downsample=False),
            resblock(64, 64, self.params, downsample=False)
        )

        self.layer2 = nn.Sequential(
            resblock(64, 128, self.params, downsample=True),
            resblock(128, 128, self.params, downsample=False)
        )

        self.layer3 = nn.Sequential(
            resblock(128, 256, self.params, downsample=True),
            resblock(256, 256, self.params, downsample=False)
        )


        self.layer4 = nn.Sequential(
            resblock(256, 512, self.params, downsample=True),
            resblock(512, 512, self.params, downsample=False)
        )

        self.gap = torch.nn.AdaptiveAvgPool2d(1) #with this module you don't have to recompute the out dimension from ,convolutional layer varying the input size
        self.fc = torch.nn.Linear(512, outputs)

    def forward(self, input):
        outs = {}

        #print("la taglia all'inizio è: ", input.size())
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.gap(input)
        #print("la taglia prima del flatten è: ", input.size())
        #input = torch.flatten(input) #flatten is not good with batch because it eliminate even the batch dimension
        input = input.view(input.size(0), -1)
        #print("la taglia prima del flatten è: ", input.size())
        outs['l2'] = input
        input = self.fc(input)
        #print("il valore salvato è: ", input)
        outs['out'] = input
        
        return outs

"""


#Imagenet-LT dataset
class LT_Dataset(torch.utils.data.Dataset): #see for example https://github.com/naver-ai/cmo/blob/main/imbalance_data/lt_data.py

    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform

        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        self.targets = self.labels  # Sampler needs to use targets

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        # return sample, label, path
        return sample, label



CATEGORIES_2021 = ["kingdom", "phylum", "class", "order", "family", "genus"]

DATASET_URLS = {
    "2017": "https://ml-inat-competition-datasets.s3.amazonaws.com/2017/train_val_images.tar.gz",
    "2018": "https://ml-inat-competition-datasets.s3.amazonaws.com/2018/train_val2018.tar.gz",
    "2019": "https://ml-inat-competition-datasets.s3.amazonaws.com/2019/train_val2019.tar.gz",
    "2021_train": "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.tar.gz",
    "2021_train_mini": "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.tar.gz",
    "2021_valid": "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz",
}

DATASET_MD5 = {
    "2017": "7c784ea5e424efaec655bd392f87301f",
    "2018": "b1c6952ce38f31868cc50ea72d066cc3",
    "2019": "c60a6e2962c9b8ccbd458d12c8582644",
    "2021_train": "38a7bb733f7a09214d44293460ec0021",
    "2021_train_mini": "db6ed8330e634445efc8fec83ae81442",
    "2021_valid": "f6f6e0e242e3d4c9569ba56400938afc",
}


class INaturalist(VisionDataset):
    """`iNaturalist <https://github.com/visipedia/inat_comp>`_ Dataset.

    Args:
        root (string): Root directory of dataset where the image files are stored.
            This class does not require/use annotation files.
        version (string, optional): Which version of the dataset to download/use. One of
            '2017', '2018', '2019', '2021_train', '2021_train_mini', '2021_valid'.
            Default: `2021_train`.
        target_type (string or list, optional): Type of target to use, for 2021 versions, one of:

            - ``full``: the full category (species)
            - ``kingdom``: e.g. "Animalia"
            - ``phylum``: e.g. "Arthropoda"
            - ``class``: e.g. "Insecta"
            - ``order``: e.g. "Coleoptera"
            - ``family``: e.g. "Cleridae"
            - ``genus``: e.g. "Trichodes"

            for 2017-2019 versions, one of:

            - ``full``: the full (numeric) category
            - ``super``: the super category, e.g. "Amphibians"

            Can also be a list to output a tuple with all specified target types.
            Defaults to ``full``.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    
    #NOTE: FOR NOW ONLY _INIT_2021 METHOD HAS BEEN ADAPTED 
    #NOTE: FOR NOW THE SELF.TARGET ATTRIBUTE IS DEFINED ONLY FOR SINGLE LABEL USING; NOT IMPLEMENTED YET FOR MULTI-LABELS SITUATION
    
    def __init__(
        self,
        root: str,
        version: str = "2021_train",
        target_type: Union[List[str], str] = "full",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        self.version = verify_str_arg(version, "version", DATASET_URLS.keys())

        super().__init__(os.path.join(root, version), transform=transform, target_transform=target_transform)

        os.makedirs(root, exist_ok=True)
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        self.all_categories: List[str] = []

        # map: category type -> name of category -> index
        self.categories_index: Dict[str, Dict[str, int]] = {}

        # list indexed by category id, containing mapping from category type -> index
        self.categories_map: List[Dict[str, int]] = []

        if not isinstance(target_type, list):
            target_type = [target_type]
        if self.version[:4] == "2021":
            self.target_type = [verify_str_arg(t, "target_type", ("full", *CATEGORIES_2021)) for t in target_type]
            self._init_2021()
        else:
            self.target_type = [verify_str_arg(t, "target_type", ("full", "super")) for t in target_type]
            self._init_pre2021()

        # index of all files: (full category id, filename)
        self.index: List[Tuple[int, str]] = []
        
        self.targets = []
        
        for dir_index, dir_name in enumerate(self.all_categories):
            files = os.listdir(os.path.join(self.root, dir_name)) #list of all files inside each category folder
            for fname in files:
                self.index.append((dir_index, fname))
                
                
                #after the _init_2021 procedure we define a self.target list attribute
                #we use self.target_type[0] as we are using a single element; in this case self.target_type is a list with 1 element
                self.targets.append(self.categories_map[dir_index][self.target_type[0]]) 
                #print(dir_index, self.categories_map[dir_index])
        #print(self.targets)
    def _init_2021(self) -> None:
        """Initialize based on 2021 layout"""

        self.all_categories = sorted(os.listdir(self.root)) #the list of all categories folders

        # map: category type -> name of category -> index
        self.categories_index = {k: {} for k in CATEGORIES_2021}
        
        #we define a variable with same structure of self.categories_index 
        #while self.categories_index store the index self.categories_occourencesstore the number of occurencies of each class
        #we can then define self.categories_index  ordering according to the occourences
        self.categories_occourences = {k: {} for k in CATEGORIES_2021}

        for dir_index, dir_name in enumerate(self.all_categories):
            #we start counting the files into the selected path
            Num_Images_in_folder =  len(os.listdir(os.path.join(self.root, dir_name)))
            
            
            pieces = dir_name.split("_")
            if len(pieces) != 8:
                raise RuntimeError(f"Unexpected category name {dir_name}, wrong number of pieces")
            if pieces[0] != f"{dir_index:05d}":
                raise RuntimeError(f"Unexpected category id {pieces[0]}, expecting {dir_index:05d}")
                
            cat_map = {}
            #here we define the number of occurrencies in the dataset of each class
            for cat, name in zip(CATEGORIES_2021, pieces[1:7]):
                
                if name in self.categories_occourences[cat]:
                    self.categories_occourences[cat][name] += Num_Images_in_folder
                else:
                    self.categories_occourences[cat][name] = Num_Images_in_folder
               
              

        #here we define the class index according to the above ordering
        for cat in self.categories_occourences:
            self.categories_occourences[cat] = dict(sorted(self.categories_occourences[cat].items(), key=operator.itemgetter(1),reverse=True))
            CatCount=0
            for name in self.categories_occourences[cat]:
                self.categories_index[cat][name] = CatCount
                CatCount+=1
        
        print('occ', self.categories_occourences[self.target_type[0]]) 
        print('ind',self.categories_index[self.target_type[0]])      
 
        #define the mapping between each class and the corresponding label
        for dir_index, dir_name in enumerate(self.all_categories): 
            pieces = dir_name.split("_")
            if len(pieces) != 8:
                raise RuntimeError(f"Unexpected category name {dir_name}, wrong number of pieces")
            if pieces[0] != f"{dir_index:05d}":
                raise RuntimeError(f"Unexpected category id {pieces[0]}, expecting {dir_index:05d}")
            for cat, name in zip(CATEGORIES_2021, pieces[1:7]):                
                cat_map[cat] = self.categories_index[cat][name] 
            #print(cat_map)
            #we save for each of the 10000 classes a list of the corresponding mapping between category name and labels
            #NOTE: the deepcpoy is necessary since dict is a mutable object
            self.categories_map.append(copy.deepcopy(cat_map)) 

        #print('here the number of element per class', self.categories_occourences)


    def _init_pre2021(self) -> None:
        """Initialize based on 2017-2019 layout"""

        # map: category type -> name of category -> index
        self.categories_index = {"super": {}}

        cat_index = 0
        super_categories = sorted(os.listdir(self.root))
        for sindex, scat in enumerate(super_categories):
            self.categories_index["super"][scat] = sindex
            subcategories = sorted(os.listdir(os.path.join(self.root, scat)))
            for subcat in subcategories:
                if self.version == "2017":
                    # this version does not use ids as directory names
                    subcat_i = cat_index
                    cat_index += 1
                else:
                    try:
                        subcat_i = int(subcat)
                    except ValueError:
                        raise RuntimeError(f"Unexpected non-numeric dir name: {subcat}")
                if subcat_i >= len(self.categories_map):
                    old_len = len(self.categories_map)
                    self.categories_map.extend([{}] * (subcat_i - old_len + 1))
                    self.all_categories.extend([""] * (subcat_i - old_len + 1))
                if self.categories_map[subcat_i]:
                    raise RuntimeError(f"Duplicate category {subcat}")
                self.categories_map[subcat_i] = {"super": sindex}
                self.all_categories[subcat_i] = os.path.join(scat, subcat)

        # validate the dictionary
        for cindex, c in enumerate(self.categories_map):
            if not c:
                raise RuntimeError(f"Missing category {cindex}")

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where the type of target specified by target_type.
        """

        cat_id, fname = self.index[index]
        img = Image.open(os.path.join(self.root, self.all_categories[cat_id], fname))

        target: Any = []
        for t in self.target_type:
            if t == "full":
                target.append(cat_id)
            else:
                target.append(self.categories_map[cat_id][t])
        target = tuple(target) if len(target) > 1 else target[0]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self) -> int:
        return len(self.index)

    def category_name(self, category_type: str, category_id: int) -> str:
        """
        Args:
            category_type(str): one of "full", "kingdom", "phylum", "class", "order", "family", "genus" or "super"
            category_id(int): an index (class id) from this category

        Returns:
            the name of the category
        """
        if category_type == "full":
            return self.all_categories[category_id]
        else:
            if category_type not in self.categories_index:
                raise ValueError(f"Invalid category type '{category_type}'")
            else:
                for name, id in self.categories_index[category_type].items():
                    if id == category_id:
                        return name
                raise ValueError(f"Invalid category id {category_id} for {category_type}")


    def _check_integrity(self) -> bool:
        return os.path.exists(self.root) and len(os.listdir(self.root)) > 0

    def download(self) -> None:
        if self._check_integrity():
            raise RuntimeError(
                f"The directory {self.root} already exists. "
                f"If you want to re-download or re-extract the images, delete the directory."
            )

        base_root = os.path.dirname(self.root)

        download_and_extract_archive(
            DATASET_URLS[self.version], base_root, filename=f"{self.version}.tgz", md5=DATASET_MD5[self.version]
        )

        orig_dir_name = os.path.join(base_root, os.path.basename(DATASET_URLS[self.version]).rstrip(".tar.gz"))
        if not os.path.exists(orig_dir_name):
            raise RuntimeError(f"Unable to find downloaded files at {orig_dir_name}")
        os.rename(orig_dir_name, self.root)
        print(f"Dataset version '{self.version}' has been downloaded and prepared for use")

    """
    #INaturalist dataset customized class (see for example https://github.com/frank-xwang/RIDE-LongTailRecognition/blob/main/data_loader/inaturalist_data_loaders.py): 
        #the built-in class perform early load, but the dataset is quite big so we define a customized version that implement lazy data-loading
    class LT_Dataset(torch.utils.data.Dataset):
        
        def __init__(self, root, txt, transform=None):
            self.img_path = []
            self.labels = []
            self.transform = transform
            with open(txt) as f:
                for line in f:
                    self.img_path.append(os.path.join(root, line.split()[0]))
                    self.labels.append(int(line.split()[1]))
            self.targets = self.labels # Sampler needs to use targets
            
        def __len__(self):
            return len(self.labels)
            
        def __getitem__(self, index):
    
            path = self.img_path[index]
            label = self.labels[index]
            
            with open(path, 'rb') as f:
                sample = Image.open(f).convert('RGB')
            
            if self.transform is not None:
                sample = self.transform(sample)
    
            # return sample, label, path
            return sample, label
    """




class DatasetTrial:
    
    def __init__(self, params):
        self.params = params.copy()
    
    def DataTempCopy(self):
        """
        This is a copy of the DataLoad method in bricks; we use it (in absence of class selection) to test the dataset on a temp instance, count the number the clasees and create the mapping dict 
        Returns
        -------
        None.

        """
        
            
        # convert data to torch.FloatTensor
        #transform = transforms.ToTensor()
        
        #convert data to tensor and standardize them (rescale each channel of each image fixing the mean to 0 and the std to 1)
        
        
        
        #TODO: correct standardization for the mnist dataset below
        if (self.params['Dataset']=='MNIST'):
            self.transform = transforms.Compose([
                        transforms.ToTensor(), #NOTE: Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                        transforms.Normalize((0.), (1.))])     
        elif(self.params['Dataset']=='CIFAR10'):    
            self.transform = transforms.Compose([
                    transforms.ToTensor(), #NOTE: Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                    #NOTE: the standardization depend on the dataset that you use; if you use a subset of classes you have to calculate mean and std on the restricted dataset
                    transforms.Normalize((0.49236655, 0.47394478, 0.41979155), (0.24703233, 0.24348505, 0.26158768))]) 
        elif(self.params['Dataset']=='Imagenet-LT'): 
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        elif(self.params['Dataset']=='Places365'):
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
        elif(self.params['Dataset']=='CIFAR100'):    
            self.transform = transforms.Compose([
                    transforms.ToTensor(), #NOTE: Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                    #NOTE: the standardization depend on the dataset that you use; if you use a subset of classes you have to calculate mean and std on the restricted dataset
                    #wrong values #transforms.Normalize((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025)),
                    transforms.Normalize((0.5070746, 0.48654896, 0.44091788),(0.26733422, 0.25643846, 0.27615058))
            ])
        elif(self.params['Dataset']=='INATURALIST'):
            self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize([8,8])
            ])
            
        """    
        #to check the above values used for the dataset standardization you can use the function Mean and Std from DatasetMeanStd class (CodeBlocks module); below an example for the mean
        a = []
        a = DatasetMeanStd('CIFAR100', self.params['label_map']).Mean()
        print("mean of dataset for standardization", a,flush=True)
        a = []
        a = DatasetMeanStd('CIFAR100', self.params['label_map']).Std()
        print("Std of dataset for standardization", a,flush=True)        
        """
        # choose the training and testing datasets
        
        if (self.params['Dataset']=='MNIST'):
            print('this run used MNIST dataset', file = self.params['info_file_object'])
            self.train_data = datasets.MNIST(root = 'data_nobackup', train = True, download = True, transform = self.transform)
            self.test_data = datasets.MNIST(root = 'data_nobackup', train = False, download = True, transform = self.transform)
            self.valid_data = datasets.MNIST(root = 'data_nobackup', train = False, download = True, transform = self.transform)
        elif(self.params['Dataset']=='CIFAR10'):
            print('this run used CIFAR10 dataset', file = self.params['info_file_object'])
            self.train_data = datasets.CIFAR10(root = self.params['DataFolder'], train = True, download = True, transform = self.transform)
            self.test_data = datasets.CIFAR10(root = self.params['DataFolder'], train = False, download = True, transform = self.transform) 
            self.valid_data = datasets.CIFAR10(root = self.params['DataFolder'], train = False, download = True, transform = self.transform) 
        elif(self.params['Dataset']=='Imagenet-LT'):            
            self.train_data = LT_Dataset('./data_nobackup', 'ImageNet_LT/ImageNet_LT_train.txt', transform=self.transform)
            self.valid_data = LT_Dataset('./data_nobackup', 'ImageNet_LT/ImageNet_LT_test.txt', transform=self.transform)     
            kwargs = dict(histtype='stepfilled', alpha=0.3, bins='auto',density=True)
            plt.hist(self.train_data, **kwargs)
            plt.show()
            plt.savefig("train.pdf")
            plt.hist(self.valid_data, **kwargs)
            plt.savefig("valid.pdf")
            plt.show()
        elif(self.params['Dataset']=='Places365'):
            self.train_data = datasets.Places365(root = 'data_nobackup/Places365', train = True, download = True, transform = self.transform)
            self.test_data = datasets.Places365(root = 'data_nobackup/Places365', train = False, download = True, transform = self.transform)
            self.valid_data = datasets.Places365(root = 'data_nobackup/Places365', train = False, download = True, transform = self.transform)           
        elif(self.params['Dataset']=='CIFAR100'):
            self.train_data = datasets.CIFAR100(root = self.params['DataFolder'], train = True, download = True, transform = self.transform)
            self.test_data = datasets.CIFAR100(root = self.params['DataFolder'], train = False, download = True, transform = self.transform) 
            self.valid_data = datasets.CIFAR100(root = self.params['DataFolder'], train = False, download = True, transform = self.transform)            
        elif(self.params['Dataset']=='INATURALIST'):
            
            self.train_data = INaturalist(root = self.params['DataFolder'], version='2021_train_mini', target_type=self.params['label_type'], transform=self.transform, download=False)
            self.test_data = INaturalist(root = self.params['DataFolder'], version='2021_valid', target_type=self.params['label_type'], transform=self.transform, download=False) 
            self.valid_data = INaturalist(root = self.params['DataFolder'], version='2021_valid', target_type=self.params['label_type'], transform=self.transform, download=False)            
            
            """
            for key in self.train_data.categories_index[label]:
                print(key, self.train_data.categories_index[label][key])
            """
            
            """
            LenDataset=0
            for item in self.train_data:
                LenDataset+=1
            print(LenDataset)
            print(type(item[1]))
            
            """
            
            print('number of elements and classes (train)', len(self.train_data.targets),len(torch.unique(torch.Tensor(self.train_data.targets))), flush=True)

            print('number of elements and classes (test)', len(self.test_data.targets),len(torch.unique(torch.Tensor(self.test_data.targets))), flush=True)
          



#%% BLOCKS CLASS
class Bricks:
    """
    The following class has to be interpred as the bricks that will used to made up the main code 
    it contains all the blocks of code that performs simple tasks: the general structure structure is as follow:
        inside class Bricks we instantiate one of the Net classes 
        so in this case we will not use class inheritance but only class composition: 
            I don't use the previous class as super class but simply call them creating an istance inside the class itself; each of the Net classes inherit the class NetVariables (where important measures are stored)
            Notes for newcomers in python:
                
            Inheritance is used where a class wants to derive the nature of parent class and then modify or extend the functionality of it. 
            Inheritance will extend the functionality with extra features allows overriding of methods, but in the case of Composition, we can only use that class we can not modify or extend the functionality of it. It will not provide extra features.
            Warning: you cannot define a method that explicitly take as input one of the instance variables (variables defined in the class); it will not modify the variable value. 
            Instead if you perform a class composition as done for NetVariables you can give the variable there defined as input and effectively modify them           
                                
        NOTE: a part of the variable is defined in the class inhered from the network architecture class while a parts is definded in the following class itself.
        as a general rule the interesting variables are defined in the parent class, while the temp storing variables in the bricks of the following class
        THIS IS AN IMPORTANT POINT BECAUSE A COMMON MISTAKES WHEN DEALING WITH CLASSES IS TO REDIFINE VARIABLES SHELLINGTHE PARENT'S ONES
    """
    
    def __init__(self, params):
        #You are sharing a reference to a Python dictionary; use a copy if this was not intended; dict.copy() creates a shallow copy of a dictionary
        self.params = params.copy()
        """
        self.NetMode = params['NetMode']
        self.n_out = params['n_out']
        self.NSteps = params['NSteps']
        self.n_epochs = params['n_epochs']
        """
        # initialize the NN
        #create an istance of the object depending on NetMode
        if(self.params['NetMode']=='MultiPerceptron'):
            #self.model = Net(self.params['n_out'], self.params['NSteps'], self.params['n_epochs'])
            self.model = Net(self.params)
            
        elif(self.params['NetMode']=='CNN'):
            self.model = ConvNet(self.params)
            
        elif(self.params['NetMode']=='VGG16'):
            self.model = VGG(self.params)
        elif(self.params['NetMode']=='VGG_Custom_Dropout'):
            self.model = VGG_Custom_Dropout(self.params)
        elif(self.params['NetMode']=='ResNet18'):  
            self.model = ResNet18(self.params, in_channels=3, resblock=ResBlock, outputs=self.params['n_out'])
        elif(self.params['NetMode']=='ResNet34'): 
            self.model = ResNet(ResidualBlock, [3, 4, 6, 3], self.params, num_classes = self.params['n_out'])
            
        else:
            print('Architecture argument is wrong', file = self.params['WarningFile'])
        self.RoundSolveConst = 1e3 #NOTE: this is a scaling factor to prevent underflow problem in norm computation/dot product if you have a large vector with small components; If this is not the case remove it to avoid the opposite problem (overflow)
        self.Epsilon = 1e-6
        
        self.NormGrad1 = [[] for i in range(self.model.num_classes)] 
        self.NormGrad2 = [[] for i in range(self.model.num_classes)] 
        #self.NormGradOverlap =  np.zeros((self.model.num_classes, self.params['samples_grad']))
        self.NormGrad1Tot = []
        self.NormGrad2Tot = []
        self.cos_alpha = 0 #initialize the angle for the scheduling
        
        self.model.double() #Call .double() on the model and input, which will transform all parameters to float64:
    



    #prepare the dataset and load it on the device
    def DataLoad(self):
        """
        transform dataset (convert to tensor and standardize) and wrap it in dataloader.
        If specified a selection of class and a reduction of dataset is done, (the reduction can be different for the classes if class imbalance is triggered)
        to parallelize the computation (not forward sample by sample) here we define a dataloader for each class ; in this way:
            we can define, for each class a batch with only element of the same class (and so be able to forward the whole batch and computing the corresponding per class gradient)
            we can define very big batch since we forward every time sub-batches defined on the single classes
        Returns
        -------
        None.

        """
        
            
        # convert data to torch.FloatTensor
        #transform = transforms.ToTensor()
        
        #convert data to tensor and standardize them (rescale each channel of each image fixing the mean to 0 and the std to 1)
        
        
        
        #TODO: correct standardization for the mnist dataset below
        if (self.params['Dataset']=='MNIST'):
            self.transform = transforms.Compose([
                        transforms.ToTensor(), #NOTE: Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                        transforms.Normalize((0.), (1.))])     
        elif(self.params['Dataset']=='CIFAR10'): 
            if (self.params['NetMode']=='ResNet34'): 
                normalize = transforms.Normalize(
                        mean=[0.4914, 0.4822, 0.4465],
                        std=[0.2023, 0.1994, 0.2010],
                    )
                
                # define transforms
                self.transform = transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        normalize,
                ]) 
            else:
                self.transform = transforms.Compose([
                        transforms.ToTensor(), #NOTE: Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                        #NOTE: the standardization depend on the dataset that you use; if you use a subset of classes you have to calculate mean and std on the restricted dataset
                        transforms.Normalize((0.49236655, 0.47394478, 0.41979155), (0.24703233, 0.24348505, 0.26158768))]) 
        elif(self.params['Dataset']=='Imagenet-LT'): 
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        elif(self.params['Dataset']=='Places365'):
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
        elif(self.params['Dataset']=='CIFAR100'):    
            self.transform = transforms.Compose([
                    transforms.ToTensor(), #NOTE: Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                    #NOTE: the standardization depend on the dataset that you use; if you use a subset of classes you have to calculate mean and std on the restricted dataset
                    #wrong values #transforms.Normalize((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025)),
                    transforms.Normalize((0.5070746, 0.48654896, 0.44091788),(0.26733422, 0.25643846, 0.27615058))
            ])
        elif(self.params['Dataset']=='INATURALIST'):
            self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize([256,256]),
                    transforms.CenterCrop(224)
                    #transforms.Resize([8,8])
                    ,
                    transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192])
            ])
            
        """    
        #to check the above values used for the dataset standardization you can use the function Mean and Std from DatasetMeanStd class (CodeBlocks module); below an example for the mean
        a = []
        a = DatasetMeanStd('CIFAR100', self.params['label_map']).Mean()
        print("mean of dataset for standardization", a,flush=True)
        a = []
        a = DatasetMeanStd('CIFAR100', self.params['label_map']).Std()
        print("Std of dataset for standardization", a,flush=True)        
        """
        # choose the training and testing datasets
        
        if (self.params['Dataset']=='MNIST'):
            print('this run used MNIST dataset', file = self.params['info_file_object'])
            self.train_data = datasets.MNIST(root = self.params['DataFolder'] , train = True, download = True, transform = self.transform)
            self.test_data = datasets.MNIST(root = self.params['DataFolder'], train = False, download = True, transform = self.transform)
            self.valid_data = datasets.MNIST(root = self.params['DataFolder'], train = False, download = True, transform = self.transform)
        elif(self.params['Dataset']=='CIFAR10'):
            print('this run used CIFAR10 dataset', file = self.params['info_file_object'])
            self.train_data = datasets.CIFAR10(root = self.params['DataFolder'], train = True, download = True, transform = self.transform)
            self.test_data = datasets.CIFAR10(root = self.params['DataFolder'], train = False, download = True, transform = self.transform) 
            self.valid_data = datasets.CIFAR10(root = self.params['DataFolder'], train = False, download = True, transform = self.transform) 
        elif(self.params['Dataset']=='Imagenet-LT'):            
            self.train_data = LT_Dataset('./data_nobackup', 'ImageNet_LT/ImageNet_LT_train.txt', transform=self.transform)
            self.valid_data = LT_Dataset('./data_nobackup', 'ImageNet_LT/ImageNet_LT_test.txt', transform=self.transform)     
            kwargs = dict(histtype='stepfilled', alpha=0.3, bins='auto',density=True)
            plt.hist(self.train_data, **kwargs)
            plt.show()
            plt.savefig("train.pdf")
            plt.hist(self.valid_data, **kwargs)
            plt.savefig("valid.pdf")
            plt.show()
        elif(self.params['Dataset']=='Places365'):
            self.train_data = datasets.Places365(root = 'data_nobackup/Places365', train = True, download = True, transform = self.transform)
            self.test_data = datasets.Places365(root = 'data_nobackup/Places365', train = False, download = True, transform = self.transform)
            self.valid_data = datasets.Places365(root = 'data_nobackup/Places365', train = False, download = True, transform = self.transform)           
        elif(self.params['Dataset']=='CIFAR100'):
            self.train_data = datasets.CIFAR100(root = self.params['DataFolder'], train = True, download = True, transform = self.transform)
            self.test_data = datasets.CIFAR100(root = self.params['DataFolder'], train = False, download = True, transform = self.transform) 
            self.valid_data = datasets.CIFAR100(root = self.params['DataFolder'], train = False, download = True, transform = self.transform)            
        elif(self.params['Dataset']=='INATURALIST'):
            self.train_data = INaturalist(root = self.params['DataFolder'], version='2021_train_mini', target_type=self.params['label_type'], transform=self.transform, download=False)
            self.test_data = INaturalist(root = self.params['DataFolder'], version='2021_valid', target_type=self.params['label_type'], transform=self.transform, download=False) 
            self.valid_data = INaturalist(root = self.params['DataFolder'], version='2021_valid', target_type=self.params['label_type'], transform=self.transform, download=False)            
            
            """
            for key in self.train_data.categories_index[label]:
                print(key, self.train_data.categories_index[label][key])
            """
            
            """
            LenDataset=0
            for item in self.train_data:
                LenDataset+=1
            print(LenDataset)
            print(type(item[1]))
            
            """
            
            print('numero di elementi e classi', len(self.train_data.targets),len(torch.unique(torch.Tensor(self.train_data.targets))), flush=True)

            


            #print('numero di classi', len(self.train_data.classes), flush=True)
        
            #os._exit(0)
            #sys.exit("Error message")

        
        else:
            print('the third argument ypo passed to the python code is not valid', file = self.params['WarningFile'])
        
        
        
        
        

            
            
            
        #DATASET SELECTION FOR UNBALANCED CASE
        #define a variable to fix the unbalance rate between the 2 classes
        if(self.params['ClassImbalance'] == 'ON'):
            
            self.TrainDL = {}#dict to store data loader (one for each mapped class) for train set
            self.TestDL = {}#dict to store data loader (one for each mapped class) for test set
            self.ValidDL = {}#dict to store data loader (one for each mapped class) for valid set
            #define the batch sizr for each class such that their proportion will be near to "self.params['ImabalnceProportions']"
            #the advantage of proceding like that is that we can easly get the exact same number of batches per each class
            


            self.traintargets = torch.tensor(self.train_data.targets) #convert label in tensors
            self.validtargets = torch.tensor(self.valid_data.targets) #convert label in tensors
            self.testtargets = torch.tensor(self.test_data.targets) #convert label in tensors
            #first we cast the target label (originary a list) into a torch tensor
            #we then define a copy of them to avoid issue during the class mapping 
                #can happen for example, using only the above one that I want to map {0:1, 1:0} 
                #we map the 0s in 1 and then map 1s to 0 the list of 1s will include also the 0s mapped in the precedent steps; to avoid so we follow the following rule:
            self.train_data.targets = torch.tensor(self.train_data.targets)
            self.valid_data.targets = torch.tensor(self.valid_data.targets)
            self.test_data.targets = torch.tensor(self.test_data.targets)
                    
            self.TrainIdx = {}
            self.ValidIdx = {}
            self.TestIdx = {}
            print("WARNING: the alghoritm assume that each class has in total the same number of elements; if this is not the case you have to modify the block below (in codeblocks.py) expressing the number of classes with respect to the number of element of ept for the majority class")
            for key in self.params['label_map']:
                #we start collecting the index associated to the output classes togheter
                #TRAIN
                self.trainTarget_idx = (self.traintargets==key).nonzero() 
                #l0=int(900/MajorInputClassBS)*self.TrainClassBS[self.params['label_map'][key]] #just for debug purpose
                
                
                #l0 = int(len(self.trainTarget_idx)*self.params['ImabalnceProportions'][self.params['label_map'][key]])
                l0 = 250

                self.Trainl0 = l0
                print("the number of elements selected by the class {} loaded on the trainset is {}".format(key, self.Trainl0),flush=True, file = self.params['info_file_object'])
                #print(self.trainTarget_idx)
                ClassTempVar = '%s'%self.params['label_map'][key]
                
                #VALID
                if(self.params['Dataset']=='INATURALIST'): #in this dataset we have a different number of images per class: we then select the same number of images for each class
                    self.validTarget_idx = (self.validtargets==key).nonzero()
                    self.Validl0= int(20)
                else:
                    self.validTarget_idx = (self.validtargets==key).nonzero()
                    self.Validl0= int(len(self.validTarget_idx)/2) #should be less than 500 (since the total test set has 1000 images per class)                
                #TEST
                if (self.params['ValidMode']=='Test'): #if we are in testing mode we have to repeat it for a third dataset
                    if(self.params['Dataset']=='INATURALIST'):
      
                        self.testTarget_idx = (self.testtargets==key).nonzero()
                        self.Testl0= int(20)    
                    else:
                        self.testTarget_idx = (self.testtargets==key).nonzero()
                        self.Testl0= int(len(self.testTarget_idx)/2)  #should be less than 500 (since the total test set has 1000 images per class)
                
                if self.TrainIdx: #if the mapped class has already appeared, we concatenate the new indeces to the existing ones
                    self.TrainIdx['%s'%0] = torch.cat((self.TrainIdx['%s'%0], self.trainTarget_idx[:][0:self.Trainl0]),0)
                    self.ValidIdx['%s'%0] = torch.cat((self.ValidIdx['%s'%0], self.validTarget_idx[:][0:self.Validl0]),0)
                    if (self.params['ValidMode']=='Test'): #if we are in testing mode we have to repeat it for a third dataset
                        self.TestIdx['%s'%0] = torch.cat((self.TestIdx['%s'%0], self.testTarget_idx[:][-self.Testl0:]),0)                   
                else: #if, instead the class is selected for the first time, we simply charge it on the indeces dict
                    self.TrainIdx['%s'%0] = self.trainTarget_idx[:][0:self.Trainl0]
                    self.ValidIdx['%s'%0] = self.validTarget_idx[:][0:self.Validl0] #select the last indeces for the validation so we don't have overlap increasing the size
                    if (self.params['ValidMode']=='Test'): #if we are in testing mode we have to repeat it for a third dataset
                        self.TestIdx['%s'%0] = self.testTarget_idx[:][-self.Testl0:] #select the last indeces for the validation so we don't have overlap increasing the size
            #REMAP THE LABELS: now that the indexes are fixed we map the dataset to the new labels
            for key in self.params['label_map']:               
                self.train_data.targets[self.traintargets==key]= self.params['label_map'][key] 
                self.valid_data.targets[self.validtargets==key]=self.params['label_map'][key]
                if (self.params['ValidMode']=='Test'):
                    self.test_data.targets[self.testtargets==key]=self.params['label_map'][key]
                    
            print('indeces divided per classes',flush=True)

            #DATALOADER CREATION    
            #now we iterate over the mapped classes avoiding repetition

            #TRAIN
            self.train_sampler = SubsetRandomSampler(self.TrainIdx['%s'%0])  
            #if we are studing the class imbalance case we use the sampler option to select data
            #we load the dataloader corresponding to the mapped "self.params['label_map'][key]" class as a dict element
            self.TrainDL['Class%s'%0] = torch.utils.data.DataLoader(self.train_data, batch_size = self.params['batch_size'], 
                                                   sampler = self.train_sampler, num_workers = self.params['num_workers'])     

            #VALID
            self.valid_sampler = SubsetRandomSampler(self.ValidIdx['%s'%0])
            self.ValidDL['Class%s'%0] = torch.utils.data.DataLoader(self.valid_data, batch_size = self.params['batch_size'], #note that for test and valid the choice of the batch size is not relevant (we use these dataset only in eval mode)
                                                   sampler = self.valid_sampler, num_workers = self.params['num_workers']) 
            
            if (self.params['ValidMode']=='Test'):                    
                self.test_sampler = SubsetRandomSampler(self.TestIdx['%s'%0])                    
                self.TestDL['Class%s'%0] = torch.utils.data.DataLoader(self.test_data, batch_size = self.params['batch_size'], 
                                                       sampler = self.test_sampler, num_workers = self.params['num_workers'])     
    

                
        
        #identify the number of classes from the number of outcome (in the label vector of the training set) with frequency non-zero

        self.SamplesClass = np.zeros(self.params['n_out'])
        self.TestSamplesClass = np.zeros(self.params['n_out'])
        self.ValidSamplesClass = np.zeros(self.params['n_out'])
        
        self.TrainTotal = 0
        self.TestTotal = 0
        self.ValTotal = 0        
#         for data, label in self.valid_loader:
#             for im in range(0, len(label)):
#                 self.TestSamplesClass[label[im]] +=1

        print('dataloaders prepared', flush=True)
        
        
        train_classes = [self.train_data.targets[i] for i in self.train_sampler]
        Train_number_classes = Counter(i.item() for i in train_classes)
        print(Train_number_classes)
        for key in Train_number_classes:
            self.SamplesClass[key] = Train_number_classes[key]
            self.TrainTotal += self.SamplesClass[key]
        
            print("train  number of samples in  {} is {}".format(key, self.SamplesClass[key]), flush = True, file = self.params['info_file_object'])
        print("total train  number of samples is {}".format( self.TrainTotal), flush = True, file = self.params['info_file_object'])

        valid_classes = [self.valid_data.targets[i] for i in self.valid_sampler]
        Valid_number_classes = Counter(i.item() for i in valid_classes) 
        print(Valid_number_classes)
        for key in Valid_number_classes:
            self.ValidSamplesClass[key] = Valid_number_classes[key]
            self.ValTotal += self.ValidSamplesClass[key]
        
            print("train  number of samples in  {} is {}".format(key, self.ValidSamplesClass[key]), flush = True, file = self.params['info_file_object'])
        print("total train  number of samples is {}".format( self.ValTotal), flush = True, file = self.params['info_file_object'])

        if (self.params['ValidMode']=='Test'):  
            test_classes = [self.test_data.targets[i] for i in self.test_sampler]
            Test_number_classes = Counter(i.item() for i in test_classes) 
            print(Test_number_classes)
            for key in Test_number_classes:
                self.TestSamplesClass[key] = Test_number_classes[key]
                self.TestTotal += self.TestSamplesClass[key]
            
                print("train  number of samples in  {} is {}".format(key, self.TestSamplesClass[key]), flush = True, file = self.params['info_file_object'])
            print("total train  number of samples is {}".format( self.TestTotal), flush = True, file = self.params['info_file_object'])

 
        
                
        #display an element from each mapped class to chck
        """
        for MC in set(list(self.params['label_map'].values())):
            # Display image and label.
            trainfeature, trainlabel = next(iter(self.TrainDL['Class%s'%MC]))
            print(f"Feature Batch Shape: {trainfeature.size()}", flush = True)
            print(f"Label Batch Shape: {trainlabel.size()}", flush = True)
            imgdir = trainfeature[0].squeeze()
            labels = trainlabel[0]
            #plt.imshow(imgdir, cmap="gray")
            plt.imshow(imgdir.permute(1, 2, 0)) #reverse order to make image compatible with format of imshow argument
            plt.show()
            print(f"Labels: {labels}")
        """
    

        
    #load the model on the device and define the loss function and the optimizer object
    def NetLoad(self):      
        """
        This method load the Net on the specified device and specify the criterion (loss) and optimized to use

        Returns
        -------
        None.

        """
        #load model into the device
        self.model.to(self.params['device'])
        
        
        """
        # Pytorch will only use one GPU by default.  we need to make a model instance and check if we have multiple GPUs. If we have multiple GPUs, we can wrap our model using nn.DataParallel. Then we can put our model on GPUs by model.to(device)
        if torch.cuda.device_count() > 1:
          print("Let's use", torch.cuda.device_count(), "GPUs!")
          # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
          model = nn.DataParallel(model)
        """
                
        # specify loss function (categorical cross-entropy)
        #NOTE: reduction mode
        #l'output layer (che va confrontato con i label reali) ha, in generale una forma (BS, Nc, d1,...,dk), dove BS indica il numero di immagini nella batch, Nc il numero di classi,  d1,...,dk ulteriori eventuali dimensioni
        #reduction='mean' raggruppa tutte le loss della batch sommandole e le normalizza per il numero di elementi*eventuali dimensioni (BS*d1*..*dk)
        #reduction='sum' effettua semplicemente la somma delle loss associate ad ogni elemento della batch
        self.criterion = nn.CrossEntropyLoss(reduction='sum')#reduction='sum'
        self.SampleCriterion = nn.CrossEntropyLoss(reduction = 'none')
        self.mean_criterion = nn.CrossEntropyLoss(reduction='mean')
        # specify optimizer (stochastic gradient descent) and learning rate
        #To use torch.optim you have to construct an optimizer object, that will hold the current state and will update the parameters based on the computed gradients.
        #To construct an Optimizer you have to give it an iterable containing the parameters (all should be Variable s) to optimize. Then, you can specify optimizer-specific options such as the learning rate, weight decay, etc.
        #why not declare it in the init method?
        #If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it. Parameters of a model after .cuda() will be different objects with those before the call.
        #In general, you should make sure that optimized parameters live in consistent locations when optimizers are constructed and used.
        self.optimizer = torch.optim.SGD(self.model.parameters(),lr = self.params['learning_rate'], momentum = self.params['momentum'], weight_decay=self.params['weight_decay'])
        
    def DefineRetrieveVariables(self):
        """
        creation of the temp variables; it initialize a list of empty list (one for each class) that usually are initialized at the beginning of the run (see InitialState method)

        Returns
        -------
        None.

        """
        self.MeanRepresClass = [[] for i in range(self.params['n_out'])]
        self.MRC = [[] for i in range(self.params['n_out'])]       
        self.EvaluationVariablesReset() #reset of temp evaluation variables
        self.StoringGradVariablesReset()
    
    def InitialState(self):
        """
        Iterate over the train loader calculating the mean representation per class (MRC), the per class loss 
        and the representation norm at the beginning of the train to evaluate the initial state of the system
        This version i adapted for the case with dataset divided in classes

        Returns
        -------
        None.

        """        
        
        #Starting measure to see the initial state (without any training)
        self.model.eval() 
        #creation of the temp variables; i initialize a list of empty list (one for each class)
        self.MeanRepresClass = [[] for i in range(self.params['n_out'])]
        self.MRC = [[] for i in range(self.params['n_out'])]

        self.EvaluationVariablesReset() #reset of temp evaluation variables
        #we evaluate the training set at the times before training starts
        self.StoringGradVariablesReset()
        
        for EvalKey in self.TrainDL:
            SetFlag = 'Train' 
            for dataval,labelval in self.TrainDL[EvalKey]:
        
                Mask_Flag = 1
                
                dataval = dataval.double() 
                dataval = dataval.to(self.params['device'])
                labelval = labelval.to(self.params['device']) 

                if self.params['NetMode']=='VGG_Custom_Dropout':
                    
                    self.DropoutBatchForward(dataval, Mask_Flag)
                    Mask_Flag = 0
                else:
                    self.BatchForward(dataval)
                    
                    
                self.output = self.OutDict['out'].clone()
                last_layer_repr = self.OutDict['l2'].clone()  
                    
                self.BatchEvalLossComputation(labelval, 0, SetFlag) #computation of the loss function and the gradient (with backward call)

                #computation of quantity useful for precision accuracy,... measures
                self.CorrectBatchGuesses(labelval, 0, SetFlag)
                #Store the last layer mean representation and per classes loss function
                """
                #ADAPT COMP. OF THE LAST LAYER TO THE CASE OF BATCHES (also remember the layer compression line).
                if NetInstance.params['NetMode']=='VGG_Custom_Dropout':
                    NetInstance.MeanRepresClass[labelval[i]].append(NetInstance.OutDict['l2'].clone().double())
                else:
                    NetInstance.MeanRepresClass[labelval[i]].append(NetInstance.OutDict['l2'].clone().double())
                """

                self.loss.sum().backward()   # backward pass: compute gradient of the loss with respect to model parameters
                
                #TODO: so far we surpressed the computation of gradient of the per class gradient to speed up the simulation; you may want to reintroduce it later. In this case remember to consider that now we are using a single batch with mixed element coming from each class
                #self.GradCopyUpdate(labelval[0]) #funcion to be modified since the label differ inside a batch
                self.optimizer.zero_grad()
                        
            print('init performed', flush=True)      
                    
            #putting gradient to 0 before filling it with the per class normalized sum
            self.optimizer.zero_grad()
            """
            NetInstance.LastLayerRepresCompression()
            """

        self.LossAccAppend(0, SetFlag)
            
       
        
               
                
        #WANDB BLOCK


        for i in range(0, self.model.num_classes):
            wandb.log({'Performance_measures/Training_Loss_Class_{}'.format(i): self.model.TrainClassesLoss[i][0],
                       'Performance_measures/Training_Accuracy_Class_{}'.format(i): self.model.TrainClassesAcc[i][0],
                       'Performance_measures/Rescaled_Steps': (self.params['batch_size']/self.params['learning_rate']),
                       'Performance_measures/True_Steps_+_1': 1})                   
        
            wandb.log({'GradientAngles/Gradient_Single_batch_Norm_of_Classes_{}'.format(i): (self.model.StepGradientClassNorm[i][0] ),
                       'GradientAngles/Rescaled_Steps': (self.params['batch_size']/self.params['learning_rate']),
                       'Performance_measures/True_Steps_+_1': 1}) 


        
        

    
    
        for EvalKey in self.ValidDL:
            SetFlag = 'Valid' 
            for dataval,labelval in self.ValidDL[EvalKey]:
        
                Mask_Flag = 1
                
                dataval = dataval.double() 
                dataval = dataval.to(self.params['device'])
                labelval = labelval.to(self.params['device']) 

                if self.params['NetMode']=='VGG_Custom_Dropout':
                    
                    self.DropoutBatchForward(dataval, Mask_Flag)
                    Mask_Flag = 0
                else:
                    self.BatchForward(dataval)
                    
                    
                self.output = self.OutDict['out'].clone()

                    
                self.BatchEvalLossComputation(labelval, 0, SetFlag) #computation of the loss function and the gradient (with backward call)

                #computation of quantity useful for precision accuracy,... measures
                self.CorrectBatchGuesses(labelval, 0, SetFlag)
                #Store the last layer mean representation and per classes loss function
                """
                #ADAPT COMP. OF THE LAST LAYER TO THE CASE OF BATCHES (also remember the layer compression line).
                if NetInstance.params['NetMode']=='VGG_Custom_Dropout':
                    NetInstance.MeanRepresClass[labelval[i]].append(NetInstance.OutDict['l2'].clone().double())
                else:
                    NetInstance.MeanRepresClass[labelval[i]].append(NetInstance.OutDict['l2'].clone().double())
                """

                self.loss.sum().backward()   # backward pass: compute gradient of the loss with respect to model parameters

                self.optimizer.zero_grad()
                    
            #putting gradient to 0 before filling it with the per class normalized sum
            self.optimizer.zero_grad()
            """
            NetInstance.LastLayerRepresCompression()
            """
            
        print('initial loop on valid set performed', flush=True)    
        self.LossAccAppend(0, SetFlag)
        
        
        if self.params['ValidMode']=='Test':
    
            for EvalKey in self.TestDL:
                SetFlag = 'Test' 
                for dataval,labelval in self.TestDL[EvalKey]:
            
                    Mask_Flag = 1
                    
                    dataval = dataval.double() 
                    dataval = dataval.to(self.params['device'])
                    labelval = labelval.to(self.params['device']) 
    
                    if self.params['NetMode']=='VGG_Custom_Dropout':
                        
                        self.DropoutBatchForward(dataval, Mask_Flag)
                        Mask_Flag = 0
                    else:
                        self.BatchForward(dataval)
                        
                        
                    self.output = self.OutDict['out'].clone()
    
                        
                    self.BatchEvalLossComputation(labelval, 0, SetFlag) #computation of the loss function and the gradient (with backward call)
    
                    #computation of quantity useful for precision accuracy,... measures
                    self.CorrectBatchGuesses(labelval, 0, SetFlag)
                    #Store the last layer mean representation and per classes loss function
                    """
                    #ADAPT COMP. OF THE LAST LAYER TO THE CASE OF BATCHES (also remember the layer compression line).
                    if NetInstance.params['NetMode']=='VGG_Custom_Dropout':
                        NetInstance.MeanRepresClass[labelval[i]].append(NetInstance.OutDict['l2'].clone().double())
                    else:
                        NetInstance.MeanRepresClass[labelval[i]].append(NetInstance.OutDict['l2'].clone().double())
                    """
    
                    self.loss.sum().backward()   # backward pass: compute gradient of the loss with respect to model parameters
                    self.GradCopyUpdate(labelval[0]) #again we select one of the label scalar; since all the element in the batch represent the same class they are all equivalent
                    self.optimizer.zero_grad()
                        
                #putting gradient to 0 before filling it with the per class normalized sum
                self.optimizer.zero_grad()
                """
                NetInstance.LastLayerRepresCompression()
                """    
            self.LossAccAppend(0, SetFlag)  

     




        
    #reset some variables used for temp storing of gradient information    
    def StoringGradVariablesReset(self):
        """
        Clear the Norm of the gradient and the temp variable where we store it before assign it to the p.grad

        Returns
        -------
        None.

        """
        self.Norm = np.zeros(self.model.num_classes)
        self.GradCopy = [[] for i in range(self.model.num_classes)] 
            
    def CorrelationTempVariablesInit(self):
        """
        reset Temp variables for correlation computation

        Returns
        -------
        None.

        """
        self.TwWeightTemp = []
        self.OverlapTemp = []
        self.DistanceTemp = []
    
    #reset of the class repr. vector, the total norm and the vectors of correct guess
    def EvaluationVariablesReset(self):
        """
        reset the variable associated to evaluation measures (train and test correct guesses (total and per class))

        Returns
        -------
        None.

        """
             
            
        #reset Gradient Norm (calculated for each epoch)
        self.total_norm = 0        
        #prepare tensor to store the mean representation of the classes
        self.MeanRepresClass = [[] for i in range(self.model.num_classes)]       
        # variables for training accuracy
        self.TrainCorrect = 0
        self.ClassTrainCorrect = np.zeros(self.model.num_classes)        
        #self.TrainTotal = len(self.train_loader.sampler)   
        #since we are dividing the dataloader associated to different classes we calculate the Traintotal from the sum of the classes' dataloader       
        self.TestCorrect = 0
        self.ClassTestCorrect = np.zeros(self.model.num_classes)       
        #self.ValTotal = len(self.valid_loader.sampler)
        self.ValCorrect = 0
        self.ClassValCorrect = np.zeros(self.model.num_classes)  
         
        self.train_loss = 0
        self.valid_loss = 0
        self.test_loss = 0
        
    #this is a single line command; I put it here simply to recall easly the returned variable   
    def BatchForward(self, data):
        """
        forward propagation of the batch (data) across the Net

        Parameters
        ----------
        data : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.OutDict = self.model(data)
        self.output = self.OutDict['out'].clone()
 
    def DropoutBatchForward(self, data, MF):
        """
        define a method to use in case of models with customized dropout.
        The difference with normal forward is that in this case we have 2 more flag arguments to understand when is time to 
        update the masks and wheter we are in train or eval mode.
        
        Note: every time you write model() you make a forward step, meaning also gradient accumulation. For each step this command should be given only one time, for this reason we save all the output dict in a variable and recall from it instead of reuse model()

        Parameters
        ----------
        SampleData : TYPE
            DESCRIPTION.
        MF : TYPE
            DESCRIPTION.
        TMF : TYPE
            DESCRIPTION.

        Returns
        -------
        None.
        """
        self.OutDict = self.model(data, MF)
        self.output = self.OutDict['out'].clone()           
 
    


    def LastLayerRepr(self, data, label):
        """
        storing the mean representation (defined by the last layer weight's vector) for each classes; by this representation you can compute an angle between classes

        Parameters
        ----------
        data : TYPE
            DESCRIPTION.
        label : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        for im in range(0, len(label)):
            #print("THE L2", self.model(data)['l2'][im].shape, label[im])
            self.MeanRepresClass[label[im]].append(self.model(data)['l2'][im].clone().double())
        for index in range(0, self.model.num_classes):
            if self.MeanRepresClass[index]:
                #print("indice ",index, self.MeanRepresClass[index][0].shape, self.MeanRepresClass[index][1].shape )
                self.MRC[index]= sum(torch.stack(self.MeanRepresClass[index])).detach().clone()
                self.MeanRepresClass[index].clear()
                self.MeanRepresClass[index].append(self.MRC[index].detach().clone())            
        
    #TODO: LossComputation and  SampleLossComputation can be joined in a single method       
    def LossComputation(self, label):
        """
        compute the loss

        Parameters
        ----------
        label : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # calculate the loss; we use torch.reshape(label, (-1,)) to convert a tensor of singles scalaras into a 1-D tensor (shape required to use criterion method)
        self.loss = self.criterion(self.output,torch.reshape(label, (-1,)))
                        
        #SPHERICAL CONSTRAIN BLOCK
        """
        if (self.params['SphericalConstrainMode']=='ON'):                    
            #add the sphericl regularization constrain to the loss
            #following line constrain the L2 norm to be equal to the number of training parameters (tunable weights)
            #loss += SphericalRegulizParameter*(((sum(q.pow(2.0).sum() for q in model.parameters() if q.requires_grad))-(sum(p.numel() for p in model.parameters() if p.requires_grad))).pow(2.0))
            
            #following line constrain the L2 norm to be equal to the number of neurons (neurons for each layer + number of classes)
            if  (self.params['Dataset']=='MNIST'):
                TotalNeurons = 32 + 32 + torch.count_nonzero(self.train_data.targets.bincount()) 
            elif (self.params['Dataset']=='CIFAR10'):
                TotalNeurons = 32 + 32 + self.model.num_classes
            self.loss += self.params['SphericalRegulizParameter']*(((sum(q.pow(2.0).sum() for q in self.model.parameters() if q.requires_grad))-(TotalNeurons)).pow(2.0))
        """

    def BatchEvalLossComputation(self, label, TimeComp=None, SetFlag=None):
        """
        compute per class loss function and the total one
        if in eval mode the loss is added to the measure of the corresponding dataset
        Parameters
        ----------
        label : vector
            labels of the batch's elements.
        TimeComp : int
            time component of the array (second component).
        SetFlag : string
            specify (for the evaluation mode) which dataset we are using at the moment of the function's call (and so where to store the measures) 

        Returns
        -------
        None.

        """





        #print("label and output", torch.reshape(label, (-1,)))
        self.loss = self.SampleCriterion(self.output,torch.reshape(label, (-1,))) #reduction = 'none' is the option to get back a single loss for each sample in the batch (for this reason I use the SampleCriterion)

        label = torch.reshape(label, (-1,))

        
        #WARNING: we now don't have the problem of mixed classes inside the batch but you have to be carefull about the choice of the criterion mode; we don't want the mean over the batch but the sum (because the batches of dataloader associated to different classes may have a different size (if there is class imbalance))
        #we select one random label[i] (they are all the same) and assign the computed loss to the variable of the corresponding class
        #note that we check to be in eval mode (since both train and eval are computed there) 
        
        im_counted = 0
        index=0
        
        if not self.model.training:
            
            while im_counted<len(label):

                if SetFlag =='Train':
                    self.model.TrainClassesLoss[index][TimeComp] += ((label==index).int()*(self.loss)).sum().item()
                    #print('check loss', self.loss, (label==index).int(), ((label==index).int()*(self.loss)).sum().item())
                elif SetFlag=='Valid':                
                    self.model.ValidClassesLoss[index][TimeComp] += ((label==index).int()*(self.loss)).sum().item()
                elif SetFlag=='Test':                
                    self.model.TestClassesLoss[index][TimeComp] += ((label==index).int()*(self.loss)).sum().item()
                else:
                    print("WARNING: you set a wrong value for the set flag: only Train Valid and Test are allowed", file = self.params['WarningFile'])
                im_counted = im_counted + ((label==index).int()).sum().item()
                index+=1
        """
        #OLD APPROACH WITH ALL THE CLASSES MIXED INTO ONE SINGLE DATALOADER
        for i in range (0, self.model.num_classes):
            #print(((torch.reshape(label, (-1,))==i).int()*self.loss).sum().item(), (torch.reshape(label, (-1,))==i).int(), self.loss )
            vec[i][TimeComp] += ((label==i).int()*self.loss).sum().item() #sum up all term in loss belonging to the same class and add to the corresponding component 
        #print('LABEL LOSS', (label==i).int(), self.loss, ((label==i).int()*self.loss).sum().item()  )
        """
    def SampleLossComputation(self, label):
        """
        compute the loss on a single image of the batch as input

        Parameters
        ----------
        label : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        #reshape output in 1-d vector to make it compatible with criterion input shape
        self.loss = self.criterion(self.output.expand(1, self.model.num_classes),(label).expand(1))
        
        #SPHERICAL CONSTRAIN BLOCK
        """
        if (self.params['SphericalConstrainMode']=='ON'):
            
            #add the sphericl regularization constrain to the loss
            #following line constrain the L2 norm to be equal to the number of training parameters (tunable weights)
            #loss += SphericalRegulizParameter*(((sum(q.pow(2.0).sum() for q in model.parameters() if q.requires_grad))-(sum(p.numel() for p in model.parameters() if p.requires_grad))).pow(2.0))
            
            #following line constrain the L2 norm to be equal to the number of neurons (neurons for each layer + number of classes)
            if  (self.params['Dataset']=='MNIST'):
                TotalNeurons = 32 + 32 + torch.count_nonzero(self.train_data.targets.bincount()) 
            elif (self.params['Dataset']=='CIFAR10'):
                TotalNeurons = 32 + 32 + self.model.num_classes
            self.loss += self.params['SphericalRegulizParameter']*(((sum(q.pow(2.0).sum() for q in self.model.parameters() if q.requires_grad))-(TotalNeurons)).pow(2.0))
        """
            



        
    
    def CorrectBatchGuesses(self, label, TimesComponentCounter, SetFlag):
        """
        Evaluate how many guesses (Between the BS images forwarded ) were correctly assigned (are equal to the real label)

        Parameters
        ----------
        label : TYPE
            DESCRIPTION.
        TimesComponentCounter : TYPE
            DESCRIPTION.
        SetFlag : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        im_counted = 0
        i=0
                
        if SetFlag=='Train':
            
            _, self.TrainPred = torch.max(self.output, 1)
            label = torch.reshape(label, (-1,))
            
            while im_counted<len(label):
                self.ClassTrainCorrect[i] += ((label==i).int()*(self.TrainPred==i).int()).sum().item()
                self.TrainCorrect+= ((label==i).int()*(self.TrainPred==i).int()).sum().item()
                self.model.TP[i][TimesComponentCounter] += ((label==i).int()*(self.TrainPred==i).int()).sum().item()
                self.model.FP[i][TimesComponentCounter] += ((label!=i).int()*(self.TrainPred==i).int()).sum().item()
                self.model.FN[i][TimesComponentCounter] += ((label==i).int()*(self.TrainPred!=i).int()).sum().item()  
                im_counted = im_counted + ((label==i).int()).sum().item()
                i+=1
        elif SetFlag=='Valid':
            _, ValPred = torch.max(self.output, 1)
            label = torch.reshape(label, (-1,))

            while im_counted<len(label):
                self.ClassValCorrect[i] += ((label==i).int()*(ValPred==i).int()).sum().item()
                self.ValCorrect+= ((label==i).int()*(ValPred==i).int()).sum().item()
                im_counted = im_counted + ((label==i).int()).sum().item()   
                i+=1
        elif SetFlag=='Test':
            #calculating correct samples for accuracy
            _, TestPred = torch.max(self.output, 1)    
            label = torch.reshape(label, (-1,))
            
            while im_counted<len(label):
                self.ClassTestCorrect[i] += ((label==i).int()*(TestPred==i).int()).sum().item()
                self.TestCorrect+= ((label==i).int()*(TestPred==i).int()).sum().item() 
                   
                im_counted = im_counted + ((label==i).int()).sum().item()  
                i+=1
        
    def LossAccAppend(self, TimesComponentCounter, SetFlag):
        """
        Accuracy and Losses (total and per class) are computed and assigned to the corresponding measure variable
        NOTE: 
            -the loss function (both the total and the per-class) is normalized for the number of element over which is calculated
            - for such normalization is sufficient to use the number of elements (in the whole dataset or only in the dataloader corresponding to a  specific class)
             In the oversampled algorithms there is no difference because in any case the evaluation phase check the whole dataset one time (without repetition); in fact we are just evaluating, not forwarding to update

        Parameters
        ----------
        TimesComponentCounter : TYPE
            DESCRIPTION.
        SetFlag : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if SetFlag=='Train':
            self.model.TrainLoss.append(np.sum(self.model.TrainClassesLoss, axis=0)[TimesComponentCounter] / self.TrainTotal)   
            self.model.TrainAcc.append(100*self.TrainCorrect / self.TrainTotal)
                   
            for k in range(self.model.num_classes):
                self.model.TrainClassesAcc[k][TimesComponentCounter] = (100*self.ClassTrainCorrect[k] / self.SamplesClass[k])
                self.model.TrainClassesLoss[k][TimesComponentCounter] = self.model.TrainClassesLoss[k][TimesComponentCounter]/self.SamplesClass[k]
                print("Train Class Loss saved for class {} is {}".format(k, self.model.TrainClassesLoss[k][TimesComponentCounter]), file = self.params['EpochValues_file_object'])
        if SetFlag=='Test':
            self.model.TestLoss.append( np.sum(self.model.TestClassesLoss, axis=0)[TimesComponentCounter]/ self.TestTotal) 
            
            self.model.TestAcc.append(100*self.TestCorrect / self.TestTotal)
            
        
            for k in range(self.model.num_classes):
                self.model.TestClassesAcc[k][TimesComponentCounter] = (100*self.ClassTestCorrect[k] / self.TestSamplesClass[k])
                self.model.TestClassesLoss[k][TimesComponentCounter] = self.model.TestClassesLoss[k][TimesComponentCounter]/self.TestSamplesClass[k]
                print("Test Class Loss saved for class {} is {}".format(k, self.model.TestClassesLoss[k][TimesComponentCounter]), file = self.params['EpochValues_file_object'])
        if SetFlag=='Valid':
            self.model.ValidLoss.append( np.sum(self.model.ValidClassesLoss, axis=0)[TimesComponentCounter]/ self.ValTotal) 
            
            self.model.ValidAcc.append(100*self.ValCorrect / self.ValTotal)
            
        
            for k in range(self.model.num_classes):
                self.model.ValidClassesAcc[k][TimesComponentCounter] = (100*self.ClassValCorrect[k] / self.ValidSamplesClass[k])
                self.model.ValidClassesLoss[k][TimesComponentCounter] = self.model.ValidClassesLoss[k][TimesComponentCounter]/self.ValidSamplesClass[k]
                print("Vaid Class Loss saved for class {} is {}".format(k, self.model.ValidClassesLoss[k][TimesComponentCounter]), file = self.params['EpochValues_file_object'])
                           

        

    def GradCopyUpdate(self, label):
        """
        copy the gradient to a storing variable that will be used to implement the update prescribed by the algorithm

        Parameters
        ----------
        label : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        ParCount = 0
        if not self.GradCopy[label]:
            for p in self.model.parameters():   
                #self.GradCopy[label].append(p.grad.clone().double()) #we copy the gradient with double precision (.double()) to prevent rounding error in case of very small numbers                           
                self.GradCopy[label].append(p.grad.clone())
                
        elif self.GradCopy[label]:
            for p in self.model.parameters(): 
                self.GradCopy[label][ParCount] = self.GradCopy[label][ParCount].clone() + p.grad.clone()#self.GradCopy[label][ParCount].clone() + p.grad.clone().double()
                ParCount +=1


    def NormalizeGradVec(self):
        """
        normalize the gradient vector

        Returns
        -------
        None.

        """
        Norm =0
        for p in self.model.parameters():
            param_norm = p.grad.detach().data.norm(2)
            Norm += param_norm.item() ** 2   
        for p in self.model.parameters():
            p.grad = torch.div(p.grad.clone(), Norm**0.5)




    #reset some variables used for temp storing of gradient information    
    def StoringDATASET_GradReset(self):
        """
        Clear the Norm of the gradient and the temp variable where we store it before assign it to the p.grad

        Returns
        -------
        None.

        """
        self.DatasetNorm = np.zeros(self.model.num_classes)
        self.DataGradCopy = [[] for i in range(self.model.num_classes)] 

    def SaveNormalizedGradient(self):
        """
        save the normalized gradient on a second variable to perform the signal projection

        Returns
        -------
        None.

        """
        for index in range(0, self.model.num_classes):
            #compute the total loss function as the sum of all class losses
                   
            TGComp=0
            for obj in self.GradCopy[index]:
                self.DatasetNorm[index] += (torch.norm(obj.cpu().clone()*self.RoundSolveConst).detach().numpy())**2
    
            self.DatasetNorm[index] = (self.DatasetNorm[index]**0.5)/self.RoundSolveConst
            ParCount = 0
            for p in self.model.parameters():
                self.DataGradCopy[index].append(torch.div(self.GradCopy[index][ParCount].clone(), (self.DatasetNorm[index] + 0.000001)).clone())
                ParCount +=1   
      

    def SignalProjectionNorm(self):
        """
        projection normalization:
            -we normalize the batch class vector
            -compute their projection along the signal (normalized vector)
            -we moltiply each class vector for the scalr product of the other class such that the 2 will assume the same value
        Returns
        -------
        None.
        """
        self.Norm = np.zeros(self.model.num_classes) #reset norm
        self.SignalProj = np.zeros(self.model.num_classes)
        for index in range(0, self.model.num_classes):
            #compute the total loss function as the sum of all class losses
            
            #compute batch classes norms to pass from vector to versor
            for obj in self.GradCopy[index]:
                self.Norm[index] += (torch.norm(obj.cpu().clone()*self.RoundSolveConst).detach().numpy())**2        
            self.Norm[index] = (self.Norm[index]**0.5)/self.RoundSolveConst
            
            #compute the scalar product between the dataset vectors and the batchs' ones
            TGComp=0
            
            for obj in self.GradCopy[index]:

                self.SignalProj[index] += np.sum(np.multiply(self.DataGradCopy[index][TGComp].cpu().clone().detach().numpy()*self.RoundSolveConst, obj.cpu().clone().detach().numpy()*self.RoundSolveConst))


                TGComp +=1
            self.SignalProj[index] = self.SignalProj[index]/(self.RoundSolveConst*self.RoundSolveConst*self.Norm[index])       
        
        #we define the factor to put the same projection value to each class
        self.ProjNormFactor = self.SignalProj.prod()/self.SignalProj
        for index in range(0, self.model.num_classes):
            ParCount = 0
            for p in self.model.parameters():
                p.grad += torch.div(self.GradCopy[index][ParCount].clone(), (self.Norm[index] + 0.000001))*self.ProjNormFactor[index]
                ParCount +=1               
            
            
            
            
            
            
            

    def PerClassNormalizedGradient(self, TimeComp):
        """
        vector for the weights' update is given by the sum of class gradient terms, each one normalized 

        Parameters
        ----------
        TimeComp : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.TotGrad = []
        
        

        for index in range(0, self.model.num_classes):
            #compute the total loss function as the sum of all class losses
                   
            TGComp=0
            for obj in self.GradCopy[index]:
                self.Norm[index] += (torch.norm(obj.cpu().clone()*self.RoundSolveConst).detach().numpy())**2
                #filling the total gradient
                if(index==0):
                    self.TotGrad.append(obj.clone())
                else:
                    self.TotGrad[TGComp] += obj.clone()
                    TGComp+=1
    
            self.Norm[index] = (self.Norm[index]**0.5)/self.RoundSolveConst
            ParCount = 0
            for p in self.model.parameters():
                p.grad += torch.div(self.GradCopy[index][ParCount].clone(), (self.Norm[index] + 0.000001))
                ParCount +=1   
        if self.model.training:  # assign the mean gradient norm only in the training phase and not during evaluation cycle   
            self.model.ClassesGradientNorm[TimeComp] += self.Norm #for the PCNGD I have only one element for each epoch, for the PCNSGD I have one for each batch; I sum up all of them (and divide for the number of batches to obtain an average value)                
        #calculating the total gradient norm
        for obj in self.TotGrad:
            self.total_norm += (torch.norm(obj.cpu().clone()).detach().numpy())**2        
        self.TotNormCopy = self.total_norm



    def BisectionGradient(self, TimeComp):
        """
        vector for the weights' update is given by the sum of class gradient terms, each one normalized 

        Parameters
        ----------


        Returns
        -------
        None.

        """
        self.TotGrad = []
        self.PCNorm =0
        

        for index in range(0, self.model.num_classes):
            #compute the total loss function as the sum of all class losses
                   
            TGComp=0
            for obj in self.GradCopy[index]:
                self.Norm[index] += (torch.norm(obj.cpu().clone()*self.RoundSolveConst).detach().numpy())**2
                #filling the total gradient
                if(index==0):
                    self.TotGrad.append(obj.clone())
                else:
                    self.TotGrad[TGComp] += obj.clone()
                    TGComp+=1
    
            self.Norm[index] = (self.Norm[index]**0.5)/self.RoundSolveConst
            ParCount = 0
            for p in self.model.parameters():
                p.grad += torch.div(self.GradCopy[index][ParCount].clone(), (self.Norm[index] + 0.000001))
                ParCount +=1   
        if self.model.training:  # assign the mean gradient norm only in the training phase and not during evaluation cycle   
            self.model.ClassesGradientNorm[TimeComp] += self.Norm #for the PCNGD I have only one element for each epoch, for the PCNSGD I have one for each batch; I sum up all of them (and divide for the number of batches to obtain an average value)                
        #calculating the total gradient norm
        for obj in self.TotGrad:
            self.total_norm += (torch.norm(obj.cpu().clone()).detach().numpy())**2        
        self.TotNormCopy = self.total_norm
        
        #calculating the norm of the per class normalized and renormalize with the total norm 
        for p in self.model.parameters():
            param_norm = p.grad.detach().data.norm(2)
            self.PCNorm += param_norm.item() ** 2
        self.PCNorm =  self.PCNorm** 0.5   
        
        for p in self.model.parameters():
            p.grad = torch.mul(torch.div(p.grad.detach(), (self.PCNorm+ 0.000001)), (self.total_norm**0.5))


    def GradNorm(self, TimeComp):
        """
        Compute the norm of gradient associated to each class

        Parameters
        ----------
        TimeComp : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.Norm = np.zeros(self.model.num_classes) #reset norm
        for index in range(0, self.model.num_classes):
            #compute the total loss function as the sum of all class losses
            for obj in self.GradCopy[index]:
                self.Norm[index] += (torch.norm(obj.cpu().clone()*self.RoundSolveConst).detach().numpy())**2    
            self.Norm[index] = (self.Norm[index]**0.5)/self.RoundSolveConst
            ParCount = 0  
        self.model.ClassesGradientNorm[TimeComp] += self.Norm #for the PCNGD I have only one element for each epoch, for the PCNSGD I have one for each batch; I sum up all of them (and divide for the number of batches to obtain an average value)                
        #calculating the total gradient norm

    def NormGrad1Copy(self):
        """
        copy the normalized grad at each 'relevant' step.
        call this function after the PerClassNormalizedGradient(self, TimeComp) in which classes norms of gradients are computed

        Returns
        -------
        None.

        """
        
        self.NormGrad1 = [[] for i in range(self.model.num_classes)] 
        self.Grad1_Norm = np.zeros(self.model.num_classes)
        
        if(all(self.GradCopy)):  
            #self.ns+=1
            for index in range(0, self.model.num_classes):# we normalize fixing the norm to self.RoundSolveConst to avoid to much lit values (underflow issues during overlap)
                for obj in self.GradCopy[index]:
                    self.Grad1_Norm[index] += (torch.norm(obj.cpu().clone()*self.RoundSolveConst).detach().numpy())**2    
                self.Grad1_Norm[index] = (self.Grad1_Norm[index]**0.5)/self.RoundSolveConst    
                ParCount=0
                for p in self.model.parameters():
                    self.NormGrad1[index].append(torch.div((copy.deepcopy(self.GradCopy[index][ParCount])*self.RoundSolveConst), (self.Grad1_Norm[index]+ 0.00000001)))
                    ParCount+=1
            self.NormGrad1Tot.append(copy.deepcopy(self.NormGrad1))
                
    def NormGrad2Copy(self):
        """
        Copy the normalized gradient after the 'relevant' step. Here I have to compute the norm also because is not stored in 
        during a call of a different method (self.Norm is used for NormGrad1Copy )
        ; and compute the overlap with the previous one
        

        Returns
        -------
        None.

        """
        self.NormGrad2 = [[] for i in range(self.model.num_classes)] 
        #self.NormGradOverlap = np.zeros((self.model.num_classes, self.params['samples_grad']))
        self.NormGradOverlap = np.zeros((self.model.num_classes, len(self.train_loader)))
        self.Grad2_Norm = np.zeros(self.model.num_classes)

        if(all(self.GradCopy) and all(self.NormGrad1)):
            #self.ns+=1
            for index in range(0, self.model.num_classes):
                #compute the total loss function as the sum of all class losses
                for obj in self.GradCopy[index]:
                    self.Grad2_Norm[index] += (torch.norm(obj.cpu().clone()*self.RoundSolveConst).detach().numpy())**2    
                self.Grad2_Norm[index] = (self.Grad2_Norm[index]**0.5)/self.RoundSolveConst
                for obj in self.GradCopy[index]:
                    self.NormGrad2[index].append(torch.div((obj.cpu().clone()*self.RoundSolveConst), (self.Grad2_Norm[index]+ 0.00000001)))
                
                
            self.NormGrad2Tot.append(copy.deepcopy(self.NormGrad2))
                


    def Wandb_Log_Grad_Overlap(self, TimeStep):
        """
        logging on wandb the measure of gradient overlap between classes

        Parameters
        ----------
        TimeStep : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        print("the length of the lists is {} and {}".format(len(self.NormGrad1Tot), len(self.NormGrad2Tot) ))
        
        if(all(self.GradCopy) and all(self.NormGrad1)):
            for i in range(0, len(self.NormGrad2Tot)):
                for index in range(0, self.model.num_classes):               
                
                    #overlap computation
                    ParCount=0
                    for p in self.model.parameters():
                        #RIPARTI DA QUA
                        self.NormGradOverlap[index][i] += np.sum(np.multiply((self.NormGrad1Tot[0][index][ParCount].cpu().detach().numpy()), (self.NormGrad2Tot[i][index][ParCount].cpu().detach().numpy())))
                        ParCount +=1     
                    self.NormGradOverlap[index][i] = self.NormGradOverlap[index][i]/(self.RoundSolveConst*self.RoundSolveConst)

            
            for i in range(0, self.model.num_classes):
                wandb.log({'Performance_measures/Normalized_Gradient_Overlap_Class_{}'.format(i): np.mean(self.NormGradOverlap,1)[i],
                           'Performance_measures/Rescaled_Steps': (TimeStep*self.params['batch_size']/self.params['learning_rate']),
                           'Performance_measures/True_Steps_+_1': TimeStep+1}) 
                
        self.NormGrad1Tot = []
        self.NormGrad2Tot = []     

        
               
                
    def PerClassMeanGradient(self, TimeComp):
        """
        Instead of normalize the per class gradient with the norm (as in PerClassNormalizedGradient) here we divide for the number of elements

        Parameters
        ----------
        TimeComp : TYPE
            DESCRIPTION.
            
        class_members:vector
            store the number of element for each batch (we use them to normalize the per class components).
            To use the algorithm for PCNGD give in input the total number of members in the dataset

        Returns
        -------
        None.

        """
        self.TotGrad = []

        for index in range(0, self.model.num_classes):
            #compute the total loss function as the sum of all class losses
                   
            TGComp=0
            for obj in self.GradCopy[index]:
                self.Norm[index] += (torch.norm(obj.cpu().clone()*self.RoundSolveConst).detach().numpy())**2
                
                #filling the total gradient
                if(index==0):
                    self.TotGrad.append(obj.clone())
                else:
                    self.TotGrad[TGComp] += obj.clone()
                    TGComp+=1
    
            self.Norm[index] = (self.Norm[index]**0.5)/self.RoundSolveConst
            ParCount = 0
            for p in self.model.parameters():
                p.grad += torch.div(self.GradCopy[index][ParCount].clone(), self.TrainClassBS[index])
                ParCount +=1   
        self.model.ClassesGradientNorm[TimeComp] += self.Norm #for the PCNGD I have only one element for each epoch, for the PCNSGD I have one for each batch; I sum up all of them (and divide for the number of batches to obtain an average value)                
        #calculating the total gradient norm
        for obj in self.TotGrad:
            self.total_norm += (torch.norm(obj.cpu().clone()).detach().numpy())**2 
        self.TotNormCopy = self.total_norm


    def PCN_lr_scheduler(self):
        """
        modify the learning rate during the simulation according to LR->LR*(1+cos(a)), where alpha is the angle between the 2 classes gradient

        Returns
        -------
        None.

        """
        
        cos_alpha = self.cos_alpha           


        lr = self.params['learning_rate']*(1+cos_alpha)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr



    def StepSize(self):
        """
        compute the size of forwarding steps (norm of the gradient*learning rate)

        Returns
        -------
        None.

        """
        self.Step=0
        for p in self.model.parameters():
            param_norm = p.grad.detach().data.norm(2)
            self.Step += param_norm.item() ** 2             
        print(self.Step**0.5, flush=True, file = self.params['StepNorm_file_object'])
        
    def GradientAngles(self, TimeComp):
        """
        Calculate the gradient angles between classes and their norm

        Parameters
        ----------
        TimeComp : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        if(all(self.GradCopy)): #I add this condition to be sure there is at least 1 element of each class (for the SGD case)
            AngleIndex=0
            for i,j in((i,j) for i in range(self.model.num_classes) for j in range(i)):
                #calculate the cos(ang) as the scalar product normalized with the l2 norm of the vectors
                ScalProd=0
    
                TGComp=0
                for obj in self.GradCopy[i]:
    
                    #print("GRAD COMP i {}".format(obj[0]))
                    #print("GRAD COMP j {}".format(self.GradCopy[j][TGComp][0]))
                    #print("DOT pROD {} E LA SOMMA {}".format(np.multiply(self.GradCopy[j][TGComp].cpu().clone().detach().numpy(), obj.cpu().clone().detach().numpy())[0], np.sum(np.multiply(self.GradCopy[j][TGComp].cpu().clone().detach().numpy(), obj.cpu().clone().detach().numpy()))))
                    #print("CON I LONG DOUBLE {} E LA SOMMA {}".format(np.multiply(np.longdouble(self.GradCopy[j][TGComp].cpu().clone().detach().numpy()), np.longdouble(obj.cpu().clone().detach().numpy()))[0], np.sum(np.multiply( np.longdouble(self.GradCopy[j][TGComp].cpu().clone().detach().numpy()), np.longdouble(obj.cpu().clone().detach().numpy())))) )
                    ScalProd += np.sum(np.multiply(self.GradCopy[j][TGComp].cpu().clone().detach().numpy()*self.RoundSolveConst, obj.cpu().clone().detach().numpy()*self.RoundSolveConst))

                    """
                    if TGComp==0:
                        print("IL TIPO INCRIMINATO È", obj.type())
                        print("L'ORDINE DI GRANDEZZA", obj, obj.cpu().clone().detach().numpy())
                        print("L'ALTRO", self.GradCopy[j][TGComp], self.GradCopy[j][TGComp].cpu().clone().detach().numpy())
                        print("L'ORDINE DOPO IL DOT PROD", np.multiply(self.GradCopy[j][TGComp].cpu().clone().detach().numpy()*self.RoundSolveConst, obj.cpu().clone().detach().numpy()*self.RoundSolveConst) )
                    print("LA SOMMA", np.sum(np.multiply(self.GradCopy[j][TGComp].cpu().clone().detach().numpy()*self.RoundSolveConst, obj.cpu().clone().detach().numpy()*self.RoundSolveConst)))
                    """
                    TGComp +=1
                ScalProd = ScalProd/(self.RoundSolveConst*self.RoundSolveConst)       
                
                #saving angle between 2 classes in a variable used for the lr_rate schedule
                #TODO: the following works only for the 2 classes problem (only one angle) pay attention in passing to more classes
                self.cos_alpha = ScalProd/((self.Norm[i]*self.Norm[j])+self.Epsilon) #I add a regularizzation in case I get near pi to preventh math error domain due to rounding errors 
                
                #print("The Scalar product between class {} and {} and corresponding norms are: {} {} {}, il prodotto delle 2 norme {}".format(i,j,ScalProd, self.Norm[i], self.Norm[j], self.Norm[i]*self.Norm[j]), flush=True)
                print("The Scalar product between class {} and {} and corresponding norms are: {} {} {}, il prodotto delle 2 norme {}".format(i,j,ScalProd, self.Norm[i], self.Norm[j], self.Norm[i]*self.Norm[j]), file = self.params['DebugFile_file_object'])
                self.model.PCGAngles[AngleIndex][TimeComp+1] = math.acos(ScalProd/((self.Norm[i]*self.Norm[j])+self.Epsilon)) #I add a regularizzation in case I get near pi to preventh math error domain due to rounding errors 
                print("ANGLE IS ", self.model.PCGAngles[AngleIndex][0], file = self.params['EpochValues_file_object'])
                
                #COMPUTATION OF ANGLES FROM NORMS (IT DOESN'T WORKS FOR THE GREAT DIFFERENCE IN THE 2 VECTORS)
                #print('WE HAVE: TOT {}, CLASSES {} {}, SCAL PROD {}'.format(self.TotNormCopy, self.Norm[i]**2, self.Norm[j]**2 ,self.Norm[i]*self.Norm[j] ) )
                #self.model.GradAnglesNormComp[AngleIndex][TimeComp] = (self.TotNormCopy - self.Norm[i]**2 - self.Norm[j]**2)/(2*self.Norm[i]*self.Norm[j])
                #self.model.GradAnglesNormComp[AngleIndex][TimeComp] = math.acos(self.model.GradAnglesNormComp[AngleIndex][TimeComp])
                
                AngleIndex+=1 
            for k in range(0, self.model.num_classes):
                self.model.StepGradientClassNorm[k][TimeComp+1] = self.Norm[k]
                

            


    def LastLayerRepresCompression(self):
        """
        We compress (sum up) the components calculated over the elements of a batch during the step.
        The compression happen as follows:
            - tensors of batch's in MeanRepresClass elements are summed up togheter
            - the resulting summed tensor is charged on MRC or added to it (depending  if MRC is empty or not)
            - MeanRepresClass is cleared to host the next batch's tensor
        Returns
        -------
        None.

        """
        for index in range(0, self.model.num_classes):
            if self.MeanRepresClass[index]:
                
                self.MeanRepresClass[index][0] = torch.sum(self.MeanRepresClass[index][0], dim=0) #sum up all the batch's elements; index 0 is due to the only presence of one batch tensor in the list
                #self.MRC[index]= sum(torch.stack(self.MeanRepresClass[index])).detach().clone()
                if not self.MRC[index]:#if the MRC[index] is empty (i.e. is the first batch with that index) we move the MeanRepres calculated over the batch there
                    self.MRC[index].append(self.MeanRepresClass[index][0].detach().clone())
                elif self.MRC[index]: #if, on the other hand, there is already something inside we add the old tensor to the new one calculated over the last batch forwarded
                    self.MRC[index][0] = torch.add(self.MRC[index][0].detach().clone(), self.MeanRepresClass[index][0].detach().clone())
                self.MeanRepresClass[index].clear() #we clear the MeanRepresClass for the next step
                
                #self.MeanRepresClass[index].append(self.MRC[index].detach().clone()) 
        

    def ReprAngles(self, TimeComp):
        """
        Compute the angles between the last layer representation of the classes (see LastLayerRepr)

        Parameters
        ----------
        TimeComp : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        #first we calculate the mean representation of the various classes    
        print('check dimensions', self.model.num_classes, len(self.MRC), self.SamplesClass.shape)
        for index in range(0, self.model.num_classes):
            self.MRC[index][0] = torch.div(self.MRC[index][0], self.SamplesClass[index]) 
            #we put the normalization of classes loss in the appensing functions
            #self.model.TrainClassesLoss[index][TimeComp+1] = self.model.TrainClassesLoss[index][TimeComp+1]/self.SamplesClass[index]
            #self.model.TestClassesLoss[index][TimeComp+1] = self.model.TestClassesLoss[index][TimeComp+1]/self.TestSamplesClass[index]
            self.model.RepresentationClassesNorm[index][TimeComp+1] = torch.norm(self.MRC[index][0].cpu()*self.RoundSolveConst).detach().numpy()
            self.model.RepresentationClassesNorm[index][TimeComp+1] = self.model.RepresentationClassesNorm[index][TimeComp+1]/self.RoundSolveConst
        
        #you can write the following iteration in more compact way:
        #from itertools import combinations
        #for i, j in combinations(range(N), 2):
        AngleIndex=0
        if(TimeComp==0):
            print("lista delle classi usate per gli angoli", flush=True, file=self.params['info_file_object'])
        for i,j in((i,j) for i in range(self.model.num_classes) for j in range(i)):
            if(TimeComp==0):
                print(i,j, flush=True, file=self.params['info_file_object'])
            

            #calculate the cos(ang) as the scalar product normalized with the l2 norm of the vectors
            self.model.TrainAngles[AngleIndex][TimeComp+1] = math.acos(np.sum(np.multiply(self.MRC[i][0].cpu().detach().numpy()*self.RoundSolveConst, self.MRC[j][0].cpu().detach().numpy()*self.RoundSolveConst))/((self.model.RepresentationClassesNorm[i][TimeComp+1]*self.model.RepresentationClassesNorm[j][TimeComp+1]*self.RoundSolveConst*self.RoundSolveConst)+self.Epsilon))
            AngleIndex+=1 
            print("angles are:", self.model.TrainAngles[:, TimeComp+1], flush=True, file =self.params['DebugFile_file_object'])

            
    def UpdatePerformanceMeasures(self, TimeComp):
        """
        Compute Precision,Recall and F1-measure

        Parameters
        ----------
        TimeComp : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        #computing precision, recall and F-measure for each class
        self.model.Prec[:, TimeComp] = self.model.TP[:, TimeComp]/ (self.model.TP[:, TimeComp]+self.model.FP[:, TimeComp])
        #self.RealPositive[:, TimeComp] = self.model.TP[:, TimeComp] + self.model.FN[:, TimeComp]
        self.model.Recall[:, TimeComp] = self.model.TP[:, TimeComp]/ (self.model.TP[:, TimeComp] + self.model.FN[:, TimeComp])
        #self.model.PR[:, TimeComp] = self.model.Prec[:, TimeComp]*self.model.Recall[:, TimeComp]
        #self.model.PR[:, TimeComp] = 2*self.model.PR[:, TimeComp]
        self.model.FMeasure[:, TimeComp] = 2*(self.model.Prec[:, TimeComp]*self.model.Recall[:, TimeComp])/ (self.model.Prec[:, TimeComp]+self.model.Recall[:, TimeComp])  

        
    def WeightsForCorrelations(self):
        """
        we save the weight configuration associated with the first time of the 2 involved  in the 2 point correlation/overlap 
        

        Returns
        -------
        None.

        """
        for param in self.model.parameters():
            self.TwWeightTemp.append(param.data)
        self.model.TwWeights.append(copy.deepcopy(self.TwWeightTemp))
        self.TwWeightTemp = []
            
        
    def CorrelationsComputation(self, IterationCounter, N, CorrTimes, tw, t):
        """
        COMPUTATION OF BOTH THE 2 TIMES CORRELATIONS: 
            we use as first time vector the one stored in self.model.TwWeights with WeightsForCorrelations
            and as second a
        

        Parameters
        ----------
        IterationCounter : TYPE
            DESCRIPTION.
        N : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        for a in range(0, self.model.Ntw):
            for b in range(0, self.model.Nt):
                if (IterationCounter==CorrTimes[a][b]):
                    
                    
                    #carico i parametri nuovi sulla variabile d'appoggio
                    for param in self.model.parameters():
                        self.TwWeightTemp.append(param.data)
                    for c in range(0, len(self.model.TwWeights[a])):
                        self.model.TwoTimesOverlap[a][b] += np.sum(np.multiply((self.model.TwWeights[a][c].cpu().detach().numpy()*self.RoundSolveConst), (self.TwWeightTemp[c].cpu().detach().numpy()*self.RoundSolveConst)))
                        self.model.TwoTimesDistance[a][b] += np.sum(np.square((np.subtract((self.TwWeightTemp[c].cpu().detach().numpy()*self.RoundSolveConst), (self.model.TwWeights[a][c].cpu().detach().numpy()*self.RoundSolveConst)))))
                    self.model.TwoTimesOverlap[a][b] = self.model.TwoTimesOverlap[a][b] / N
                    self.model.TwoTimesOverlap[a][b] = self.model.TwoTimesOverlap[a][b] /(self.RoundSolveConst*self.RoundSolveConst)
                    self.model.TwoTimesDistance[a][b] = self.model.TwoTimesDistance[a][b] / N     
                    self.model.TwoTimesDistance[a][b]=self.model.TwoTimesDistance[a][b]/(self.RoundSolveConst*self.RoundSolveConst)
                    #saving correlation on the tensorboard summary
                    #self.writer.add_scalar('Mean square distance for tw = {}'.format(tw[a]), self.model.TwoTimesDistance[a][b], global_step =  t[b])
                    #self.writer.add_scalar('Overlap for tw = {}'.format(tw[a]), self.model.TwoTimesOverlap[a][b], global_step =  t[b])
      
                    self.Corrwriter.add_scalar('Mean square distance for tw = {}'.format(tw[a]), self.model.TwoTimesDistance[a][b], global_step =  t[b])
                    self.Corrwriter.add_scalar('Overlap for tw = {}'.format(tw[a]), self.model.TwoTimesOverlap[a][b], global_step =  t[b])
                    
                    
                    #saving correlation for wandb
                    if (a==(self.model.Ntw-1)):
                        for ftc in range(0, self.model.Ntw):
                            wandb.log({'MeanSquare_Distance/tw_{}'.format(tw[ftc]): self.model.TwoTimesDistance[ftc][b],
                                       'Overlap/tw_{}'.format(tw[ftc]): self.model.TwoTimesOverlap[ftc][b],
                                       'MeanSquare_Distance/t' : t[b],
                                       'Overlap/t' : t[b]})

                    
                    
                    #empty TwWeightTempafter the computation (in order to be used for appending the first time vector in the next WeightsForCorrelations call)
                    self.TwWeightTemp = [] #note the indent; we empty the temp variable here because a single time can be a second corr. time for multiple matrix element (multiple choices of a,b)




    def WeightNormComputation(self):
        """
        compute the norm of the weight of the network

        Returns
        -------
        None.

        """
        self.model.WeightNorm.append( ((sum(p.pow(2.0).sum() for p in self.model.parameters() if p.requires_grad)) / (sum(p.numel() for p in self.model.parameters() if p.requires_grad))) )
 
    

    def UpdateFileData(self):
         """
         Rewrite data saved in files with new rupdated measures 

         Returns
         -------
         None.

         """
        #if the spherical constrain is active we add to the loss a term containing a tensor; so the loss become a tensor itself;
        #since tensor contains additive components with respect to numpy array contain ,components related to gradient computation, to consider only the array values we have to use the detach module
        

         #if the simulation starts from 0 we simply call periodically the update of files  rewriting each time the updated vectors

         #to have the same format for each vector we transpose some of the variables
         """
         if(self.params['SphericalConstrainMode']=='ON'):
             self.TempTrainingLoss=[t.detach().numpy() for t in self.model.TrainLoss]
         elif(self.params['SphericalConstrainMode']=='OFF'):
             self.TempTrainingLoss=self.model.TrainLoss
         """
         self.TempTrainingLoss=self.model.TrainLoss
         with open(self.params['FolderPath'] + "/TrainLoss.txt", "w") as f:
             np.savetxt(f, np.array(self.TempTrainingLoss), delimiter = ',')
         """
         if(self.params['SphericalConstrainMode']=='ON'):    
             self.TempTestLoss=[t.detach().numpy() for t in self.model.TestLoss]  
         elif(self.params['SphericalConstrainMode']=='OFF'):
             self.TempTestLoss=self.model.TestLoss
         """
         
         self.TempValidLoss=self.model.ValidLoss
         with open(self.params['FolderPath'] + "/ValidLoss.txt", "w") as f:
             np.savetxt(f, np.array(self.TempValidLoss), delimiter = ',')
         
         with open(self.params['FolderPath'] + "/TrainAcc.txt", "w") as f:
             np.savetxt(f, self.model.TrainAcc, delimiter = ',') 
         print("the train accuracy saved from the first simulation is ", self.model.TrainAcc)
         with open(self.params['FolderPath'] + "/ValidAcc.txt", "w") as f:
             np.savetxt(f, self.model.ValidAcc, delimiter = ',') 
                     
         with open(self.params['FolderPath'] + "/TrainPrecision.txt", "w") as f:
             np.savetxt(f,  np.transpose(self.model.Prec), delimiter = ',')
         with open(self.params['FolderPath'] + "/TrainRecall.txt", "w") as f:
             np.savetxt(f,  np.transpose(self.model.Recall), delimiter = ',')
         with open(self.params['FolderPath'] + "/TrainF_Measure.txt", "w") as f:
             np.savetxt(f,  np.transpose(self.model.FMeasure), delimiter = ',')
         
         
             
         with open(self.params['FolderPath'] + "/TrainClassesLoss.txt", "w") as f:
             np.savetxt(f,  np.transpose(self.model.TrainClassesLoss), delimiter = ',')    
         with open(self.params['FolderPath'] + "/TrainClassesAcc.txt", "w") as f:
             np.savetxt(f,  np.transpose(self.model.TrainClassesAcc) , delimiter = ',')   
         with open(self.params['FolderPath'] + "/ValidClassesLoss.txt", "w") as f:
             np.savetxt(f,  np.transpose(self.model.ValidClassesLoss), delimiter = ',')               
         with open(self.params['FolderPath'] + "/ValidClassesAcc.txt", "w") as f:
             np.savetxt(f,  np.transpose(self.model.ValidClassesAcc) , delimiter = ',')  
             
 
         if self.params['ValidMode']=='Test':        
             self.TempTestLoss=self.model.TestLoss
             with open(self.params['FolderPath'] + "/TestLoss.txt", "w") as f:
                 np.savetxt(f, np.array(self.TempTestLoss), delimiter = ',')
             with open(self.params['FolderPath'] + "/TestAcc.txt", "w") as f:
                 np.savetxt(f, self.model.TestAcc, delimiter = ',')                      
             #per classes accuracy and loss for the test set (for now implemented only for the Gd and PCNGD)    
             with open(self.params['FolderPath'] + "/TestClassesLoss.txt", "w") as f:
                 np.savetxt(f,  np.transpose(self.model.TestClassesLoss), delimiter = ',')       
             with open(self.params['FolderPath'] + "/TestClassesAcc.txt", "w") as f:
                 np.savetxt(f,  np.transpose(self.model.TestClassesAcc) , delimiter = ',')  
          
             
          
         """   
         if(self.params['Dynamic']=='PCNGD'): 
             with open(self.params['FolderPath'] + "/PCGradientAngles.txt", "w") as f:
                 np.savetxt(f, self.model.PCGAngles, delimiter = ',') 
         elif(self.params['Dynamic']=='GD'): 
             with open(self.params['FolderPath'] + "/GDGradientAngles.txt", "w") as f:
                 np.savetxt(f, np.transpose(self.model.PCGAngles), delimiter = ',') 
        """
      
         with open(self.params['FolderPath'] + "/TrainAngles.txt", "w") as f:
             np.savetxt(f,  np.transpose(self.model.TrainAngles), delimiter = ',') 
         with open(self.params['FolderPath'] + "/TestAngles.txt", "w") as f:
             np.savetxt(f,  np.transpose(self.model.TestAngles), delimiter = ',') 
         
         with open(self.params['FolderPath'] + "/RepresentationNorm.txt", "w") as f:
             np.savetxt(f,  np.transpose(self.model.RepresentationClassesNorm) , delimiter = ',')     
             

         """
         with open(self.params['FolderPath'] + "/TotP.txt", "w") as f:
             np.savetxt(f,  np.transpose((self.model.TP+self.model.FP)), delimiter = ',')   
         self.TempWeightNorm=[t.detach().numpy() for t in self.model.WeightNorm]  
         with open(self.params['FolderPath'] + "/WeightSquaredNorm.txt", "w") as f:
             np.savetxt(f, np.array(self.TempWeightNorm), delimiter = ',')
         """
             
         
         
         
         
         """
         #salva in un file i parametri (Numero di campioni, Numero di tempi, Numero di classi)
         #NOTE: CONTROLLA CHE IL NUMERO DI CLASSI SIA CARICATO CORRETTAMENTE; ClassesNumber
         Parameters = []
         Parameters.append(NSteps)
         Parameters.append(ClassesNumber)
         with open('./'+ args.FolderName + "/Parameters.txt", "w") as f:
             np.savetxt(f, Parameters, delimiter = ',')                 
         """
            
            


                    
                    
        
    def SimulationID(self):
        """
        save some useful information about the simulation in a file (that will be loaded also in wandb)

        Returns
        -------
        None.

        """
        #the server associated to one simulation is a useful information (if for example you run multiple project in parallel and xtake track of the output from an external source (like wandb))
        subprocess.run('echo The server that host the run is:  $(whoami)@$(hostname)', shell=True, stdout=self.params['info_file_object']) 
        subprocess.run('echo The path of the run codes, inside the server, is:  $(pwd)', shell=True, stdout=self.params['info_file_object']) 
        print("The PID of the main code is: ", os.getpid())
        print("The CheckMode is set on {}; when set on 'ON' mode the random seeds of the simulation are set to a fixed value to reproduce the same result over different runs".format(self.params['CheckMode']))
        print('The simulation total time is {} epochs, the start mode is set on {}, we took measures at {} logaritmically equispaced steps/epochs'.format(self.params['n_epochs'], self.params['StartMode'], self.params['NSteps']), file = self.params['info_file_object'])
        print("the simulation use {} as algorithm".format(self.params['Dynamic']), file = self.params['info_file_object'])
        print("the simulation run over the device: ", self.params['device'],  file = self.params['info_file_object'])
        print('The batch size of the simulation is {}, the learning rate is {}, dropout dropping probability is {}, the parameter for group norm is {}'.format(self.params['batch_size'], self.params['learning_rate'], self.params['dropout_p'], self.params['group_factor']), file = self.params['info_file_object'])
        print('The spherical regularization parameter (if the constrain is imposed) is: ', self.params['SphericalRegulizParameter'], file = self.params['info_file_object'])
        print("dataset used in the simulation is: ",self.params['Dataset'], file = self.params['info_file_object'])
        print("classes defined for the simulation are {} ".format(self.params['label_map']), file = self.params['info_file_object'])
        print("the oversampling (i.e. if the batch respect the dataset proportion (OFF) or takes an equal number of element from each class) mode is set on {}".format(self.params['OversamplingMode']), file = self.params['info_file_object'])
        print(" Imbalance factor introduced inside the dataset is {}, in particular the disproportion between the mapped classes is {} ".format(self.params['ClassImbalance'], self.params['ImabalnceProportions']), flush = True, file = self.params['info_file_object'])
        print("architecture used in the simulation is: ",self.params['NetMode'],  file = self.params['info_file_object'])
        print("the size of the batch was ", self.params['batch_size'], ", the learning rate value: ", self.params['learning_rate'], flush = True, file = self.params['info_file_object']) #we flush the printing at the the end
        
        
        
    def TrashedBatchesReset(self):
        self.TrashedBatches=0
        
        
    def CheckClassPresence(self, label):
        """
        check the presence of all the classes inside the batch that is about to be forwarded

        Parameters
        ----------
        label : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.TrainClassBS = np.zeros(self.model.num_classes)
        for im in range(0, len(label)):
            self.TrainClassBS[label[im]] +=1
        self.ClassPresence = np.prod(self.TrainClassBS)        
        
    def SummaryWriterCreation(self, path):
        """
        tensorboard writer creation

        Parameters
        ----------
        path : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        #self.writer = SummaryWriter(path)
        
        #SUMMARY DIVISION TO DIVIDE INFO TO UPLOAD TO WANDB
        self.writer = SummaryWriter(path+'/NotWandB')
        #I create a separate folder to save correlations (because I want to learn only these in wandb)
        CorrPath = path + '/Corr'
        self.Corrwriter = SummaryWriter(CorrPath)
        
        
    def SummaryScalarsSaving(self, TimeVec, Comp):
        """
        saving summary (measures) on tensorboard

        Parameters
        ----------
        TimeVec : TYPE
            DESCRIPTION.
        Comp : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if self.params['ValidMode']=='Valid':
            self.writer.add_scalar('Training loss', np.array(self.model.TrainLoss)[-1], global_step =  TimeVec[Comp])
            self.writer.add_scalar('Valid loss', np.array(self.model.ValidLoss)[-1], global_step =  TimeVec[Comp])
            self.writer.add_scalar('Training accuracy', (100*self.TrainCorrect / self.TrainTotal), global_step =  TimeVec[Comp])
            self.writer.add_scalar('Valid accuracy', (100*self.ValCorrect / self.ValTotal), global_step =  TimeVec[Comp])
            self.writer.add_scalar('Balanced Training accuracy', np.sum(self.model.TrainClassesAcc, axis=0)[Comp+1]/self.model.num_classes, global_step =  TimeVec[Comp])
            
            for i in range(0, self.model.num_classes):
                self.writer.add_scalar('Training loss class {}'.format(i), self.model.TrainClassesLoss[i][Comp+1], global_step =  TimeVec[Comp])
                self.writer.add_scalar('Valid loss class {}'.format(i), self.model.ValidClassesLoss[i][Comp+1], global_step =  TimeVec[Comp])                        
                self.writer.add_scalar('Training accuracy class {}'.format(i), self.model.TrainClassesAcc[i][Comp+1], global_step =  TimeVec[Comp])
                self.writer.add_scalar('Valid accuracy class {}'.format(i), self.model.ValidClassesAcc[i][Comp], global_step =  TimeVec[Comp]) 
     
            #NUOVI CHECK
            """
            if(np.shape(self.model.TwoTimesOverlap)[0]<Comp):
                self.writer.add_scalar('2-Time Autocorrelation', (self.model.TwoTimesDistance[Comp][0]), global_step =  TimeVec[Comp])
                self.writer.add_scalar('Overlap', (self.model.TwoTimesOverlap[Comp][0]), global_step =  TimeVec[Comp])
            """
            
            if self.params['Dynamic']=='SGD':
                for name, ciccia in self.model.named_parameters():
                    self.writer.add_scalar('Gradient Norm of the layer {}'.format(name), ((ciccia.pow(2.0).sum())**0.5)*self.params['batch_size'], global_step =  TimeVec[Comp])
                self.writer.add_scalar('Step Norm', (self.Step**0.5)*self.params['batch_size'], global_step =  TimeVec[Comp])
            else:
                AngleIndex=0
                for i,j in((i,j) for i in range(self.model.num_classes) for j in range(i)):
                    self.writer.add_scalar('Gradient Angles between classes {} and {}'.format(i, j), (self.model.PCGAngles[AngleIndex][Comp+1] ), global_step =  TimeVec[Comp])
                    
                    #print("ANGLE IS ", PCGAngles)
                    AngleIndex+=1 
                for name, ciccia in self.model.named_parameters():
                    self.writer.add_scalar('Gradient Norm of the layer {}'.format(name), (ciccia.pow(2.0).sum())**0.5, global_step =  TimeVec[Comp])
                self.writer.add_scalar('Step Norm', (self.Step**0.5), global_step =  TimeVec[Comp])           
                
        elif self.params['ValidMode']=='Test':
            self.writer.add_scalar('Training loss', np.array(self.model.TrainLoss)[-1], global_step =  TimeVec[Comp])
            self.writer.add_scalar('Valid loss', np.array(self.model.ValidLoss)[-1], global_step =  TimeVec[Comp])
            self.writer.add_scalar('Test loss', np.array(self.model.TestLoss)[-1], global_step =  TimeVec[Comp])
            self.writer.add_scalar('Training accuracy', (100*self.TrainCorrect / self.TrainTotal), global_step =  TimeVec[Comp])
            self.writer.add_scalar('Test accuracy', (100*self.TestCorrect / self.TestTotal), global_step =  TimeVec[Comp])
            self.writer.add_scalar('Valid accuracy', (100*self.ValCorrect / self.ValTotal), global_step =  TimeVec[Comp])
            self.writer.add_scalar('Balanced Training accuracy', np.sum(self.model.TrainClassesAcc, axis=0)[Comp+1]/self.model.num_classes, global_step =  TimeVec[Comp])
            
            for i in range(0, self.model.num_classes):
                self.writer.add_scalar('Training loss class {}'.format(i), self.model.TrainClassesLoss[i][Comp+1], global_step =  TimeVec[Comp])
                self.writer.add_scalar('Test loss class {}'.format(i), self.model.TestClassesLoss[i][Comp+1], global_step =  TimeVec[Comp])                        
                self.writer.add_scalar('Valid loss class {}'.format(i), self.model.ValidClassesLoss[i][Comp+1], global_step =  TimeVec[Comp]) 
                self.writer.add_scalar('Training accuracy class {}'.format(i), self.model.TrainClassesAcc[i][Comp+1], global_step =  TimeVec[Comp])
                self.writer.add_scalar('Test accuracy class {}'.format(i), self.model.TestClassesAcc[i][Comp], global_step =  TimeVec[Comp]) 
                self.writer.add_scalar('Valid accuracy class {}'.format(i), self.model.ValidClassesAcc[i][Comp], global_step =  TimeVec[Comp]) 
     
            #NUOVI CHECK
            """
            if(np.shape(self.model.TwoTimesOverlap)[0]<Comp):
                self.writer.add_scalar('2-Time Autocorrelation', (self.model.TwoTimesDistance[Comp][0]), global_step =  TimeVec[Comp])
                self.writer.add_scalar('Overlap', (self.model.TwoTimesOverlap[Comp][0]), global_step =  TimeVec[Comp])
            """
            
            if self.params['Dynamic']=='SGD':
                for name, ciccia in self.model.named_parameters():
                    self.writer.add_scalar('Gradient Norm of the layer {}'.format(name), ((ciccia.pow(2.0).sum())**0.5)*self.params['batch_size'], global_step =  TimeVec[Comp])
                self.writer.add_scalar('Step Norm', (self.Step**0.5)*self.params['batch_size'], global_step =  TimeVec[Comp])
            else:
                AngleIndex=0
                for i,j in((i,j) for i in range(self.model.num_classes) for j in range(i)):
                    self.writer.add_scalar('Gradient Angles between classes {} and {}'.format(i, j), (self.model.PCGAngles[AngleIndex][Comp+1] ), global_step =  TimeVec[Comp])
                    
                    #print("ANGLE IS ", PCGAngles)
                    AngleIndex+=1 
                for name, ciccia in self.model.named_parameters():
                    self.writer.add_scalar('Gradient Norm of the layer {}'.format(name), (ciccia.pow(2.0).sum())**0.5, global_step =  TimeVec[Comp])
                self.writer.add_scalar('Step Norm', (self.Step**0.5), global_step =  TimeVec[Comp])     
            
    def PerClassNormGradDistrSaving(self, TimeVec, Comp):
        """
        save on tensorboard the component distribution of gradient associated to each class

        Parameters
        ----------
        TimeVec : TYPE
            DESCRIPTION.
        Comp : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        Norma = np.zeros(self.model.num_classes)
        for index in range(0, self.model.num_classes):
            #compute the total loss function as the sum of all class losses
                   
            for obj in self.GradCopy[index]:
                Norma[index] += (torch.norm(obj.cpu().clone()*self.RoundSolveConst).detach().numpy())**2

    
            Norma[index] = (Norma[index]**0.5)/self.RoundSolveConst
            ParCount = 0
            for p in self.model.parameters():
                p.grad = torch.div(self.GradCopy[index][ParCount].clone(), (Norma[index] + 0.000001))
                ParCount +=1   
     
            for name, ciccia in self.model.named_parameters():

                self.writer.add_histogram('Gradient layer {} of class {}'.format(name, index), ciccia.grad, global_step=TimeVec[Comp])       
            self.optimizer.zero_grad() #clear gradient before passing to the next class 
        self.optimizer.zero_grad() #clear gradient before passing to the next class 
           
    def SummaryDistrSavings(self, TimeVec, Comp):  
        """
        here we save the distribution of weight and gradient on tensorboard (this is an automized useful feature of tensorboard, not present, as far as I know, in wandb)

        Parameters
        ----------
        TimeVec : TYPE
            DESCRIPTION.
        Comp : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        for name, ciccia in self.model.named_parameters():

            #print(ciccia)
            self.writer.add_histogram('Weights layer {}'.format(name), ciccia, global_step=TimeVec[Comp])
            self.writer.add_histogram('Gradient layer {}'.format(name), ciccia.grad, global_step=TimeVec[Comp])
       
    def SummaryHP_Validation(self, lr, bs):
        """
        here we select the measure we want to store to evaluate the right HP (HP tuning) throught tensorboard

        Parameters
        ----------
        lr : TYPE
            DESCRIPTION.
        bs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if self.params['ValidMode']=='Test':
            self.writer.add_hparams({'lr': lr, 'bsize': bs}, {'test loss': np.sum(self.model.TestClassesLoss, axis=0)[-1], 'test accuracy': self.model.TestAcc[-1]})
        else: 
            self.writer.add_hparams({'lr': lr, 'bsize': bs}, {'valid loss': np.sum(self.model.ValidClassesLoss, axis=0)[-1], 'test accuracy': self.model.ValidAcc[-1]})
            

    def WandB_logs(self, TimeVec, Comp):
        """
        here we log the relevant measures in wandb

        Parameters
        ----------
        TimeVec : TYPE
            DESCRIPTION.
        Comp : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        #TODO: for some reson the plot of classes valid accuracyis shifted forward in the wandb charts; this doesn't seems to happen for the training; fix this logging issue
        if self.params['ValidMode']=='Valid':
            wandb.log({'Performance_measures/Training_Loss': np.array(self.model.TrainLoss)[-1],
                       'Performance_measures/Valid_Loss': np.array(self.model.ValidLoss)[-1],
                       'Performance_measures/Training_Accuracy': (100*self.TrainCorrect / self.TrainTotal),
                       'Performance_measures/Valid_Accuracy': (100*self.ValCorrect / self.ValTotal),
                       'Performance_measures/Balanced_Training_Accuracy': np.sum(self.model.TrainClassesAcc, axis=0)[Comp+1]/self.model.num_classes,
                       'Performance_measures/Rescaled_Steps': (TimeVec[Comp]*self.params['batch_size']/self.params['learning_rate']),
                       'Performance_measures/True_Steps_+_1': TimeVec[Comp]+1})

            Performance_data = []
            Performance_data.append(TimeVec[Comp]+1)
            Performance_data.append(TimeVec[Comp]*self.params['batch_size']/self.params['learning_rate'])
            Grad_data = []
            Grad_data.append(TimeVec[Comp]+1)
            Grad_data.append(TimeVec[Comp]*self.params['batch_size']/self.params['learning_rate'])
            for i in range(0, self.model.num_classes):
                wandb.log({'Performance_measures/Training_Loss_Class_{}'.format(i): self.model.TrainClassesLoss[i][Comp+1],
                           'Performance_measures/Valid_Loss_Class_{}'.format(i): self.model.ValidClassesLoss[i][Comp+1],
                           'Performance_measures/Training_Accuracy_Class_{}'.format(i): self.model.TrainClassesAcc[i][Comp+1],
                           'Performance_measures/Valid_Accuracy_Class_{}'.format(i): self.model.ValidClassesAcc[i][Comp],
                           'Performance_measures/Rescaled_Steps': (TimeVec[Comp]*self.params['batch_size']/self.params['learning_rate']),
                           'Performance_measures/True_Steps_+_1': TimeVec[Comp]+1})    
                
                Performance_data.append(self.model.TrainClassesLoss[i][Comp+1])
                Performance_data.append(self.model.TrainClassesAcc[i][Comp+1])
            #if self.params['Dynamic']=='SGD':
            wandb.log({'Check/Step_Norm': (self.Step**0.5),
                       'Check/Rescaled_Steps': (TimeVec[Comp]*self.params['batch_size']/self.params['learning_rate']),
                       'Performance_measures/True_Steps_+_1': TimeVec[Comp]+1})
    
            if(all(self.GradCopy) or self.params['Dynamic']=='PCNSGD+R'): #if we are in the projection normalization mode we calculate with the whole dataset, so we are granted to have all the component 
                
                for k in range(0, self.model.num_classes):
                    wandb.log({'GradientAngles/Gradient_Single_batch_Norm_of_Classes_{}'.format(k): (self.model.StepGradientClassNorm[k][Comp+1] ),
                               'GradientAngles/Rescaled_Steps': (TimeVec[Comp]*self.params['batch_size']/self.params['learning_rate']),
                               'Performance_measures/True_Steps_+_1': TimeVec[Comp]+1})   
                    Grad_data.append(self.model.StepGradientClassNorm[k][Comp+1])
                
                AngleIndex=0
                for i,j in((i,j) for i in range(self.model.num_classes) for j in range(i)):
                    wandb.log({'GradientAngles/Gradient_Angles_Between_Classes_{}_and_{}'.format(i, j): (self.model.PCGAngles[AngleIndex][Comp+1] ),
                               #'GradientAngles/Gradient_Angles_NormComp_Between_Classes_{}_and_{}'.format(i, j): (self.model.GradAnglesNormComp[AngleIndex][Comp] ),
                               'GradientAngles/Rescaled_Steps': (TimeVec[Comp]*self.params['batch_size']/self.params['learning_rate']),
                               'Performance_measures/True_Steps_+_1': TimeVec[Comp]+1})
                    Grad_data.append(self.model.PCGAngles[AngleIndex][Comp+1])
                    AngleIndex+=1 
                    
            self.Performance_data_table.add_data(*copy.deepcopy(Performance_data))
            self.Grad_data_table.add_data(*copy.deepcopy(Grad_data))
        
        elif self.params['ValidMode']=='Test':
            wandb.log({'Performance_measures/Training_Loss': np.array(self.model.TrainLoss)[-1],
                       'Performance_measures/Valid_Loss': np.array(self.model.ValidLoss)[-1],
                       'Performance_measures/Test_Loss': np.array(self.model.TestLoss)[-1],
                       'Performance_measures/Training_Accuracy': (100*self.TrainCorrect / self.TrainTotal),
                       'Performance_measures/Valid_Accuracy': (100*self.ValCorrect / self.ValTotal),
                       'Performance_measures/Test_Accuracy': (100*self.TestCorrect / self.TestTotal),
                       'Performance_measures/Balanced_Training_Accuracy': np.sum(self.model.TrainClassesAcc, axis=0)[Comp+1]/self.model.num_classes,
                       'Performance_measures/Rescaled_Steps': (TimeVec[Comp]*self.params['batch_size']/self.params['learning_rate']),
                       'Performance_measures/True_Steps_+_1': TimeVec[Comp]+1})
            Performance_data = []
            Performance_data.append(TimeVec[Comp]+1)
            Performance_data.append(TimeVec[Comp]*self.params['batch_size']/self.params['learning_rate'])
            Grad_data = []
            Grad_data.append(TimeVec[Comp]+1)
            Grad_data.append(TimeVec[Comp]*self.params['batch_size']/self.params['learning_rate'])
            for i in range(0, self.model.num_classes):
                wandb.log({'Performance_measures/Training_Loss_Class_{}'.format(i): self.model.TrainClassesLoss[i][Comp+1],
                           'Performance_measures/Test_Loss_Class_{}'.format(i): self.model.TestClassesLoss[i][Comp+1],
                           'Performance_measures/Valid_Loss_Class_{}'.format(i): self.model.ValidClassesLoss[i][Comp+1],
                           'Performance_measures/Training_Accuracy_Class_{}'.format(i): self.model.TrainClassesAcc[i][Comp+1],
                           'Performance_measures/Test_Accuracy_Class_{}'.format(i): self.model.TestClassesAcc[i][Comp],
                           'Performance_measures/Valid_Accuracy_Class_{}'.format(i): self.model.ValidClassesAcc[i][Comp],
                           'Performance_measures/Rescaled_Steps': (TimeVec[Comp]*self.params['batch_size']/self.params['learning_rate']),
                           'Performance_measures/True_Steps_+_1': TimeVec[Comp]+1})    
                
                Performance_data.append(self.model.TrainClassesLoss[i][Comp+1])
                Performance_data.append(self.model.TrainClassesAcc[i][Comp+1])
            #if self.params['Dynamic']=='SGD':
            wandb.log({'Check/Step_Norm': (self.Step**0.5),
                       'Check/Rescaled_Steps': (TimeVec[Comp]*self.params['batch_size']/self.params['learning_rate']),
                       'Performance_measures/True_Steps_+_1': TimeVec[Comp]+1})
    
            if(all(self.GradCopy) or self.params['Dynamic']=='PCNSGD+R'): #if we are in the projection normalization mode we calculate with the whole dataset, so we are granted to have all the component 
                
                for k in range(0, self.model.num_classes):
                    wandb.log({'GradientAngles/Gradient_Single_batch_Norm_of_Classes_{}'.format(k): (self.model.StepGradientClassNorm[k][Comp+1] ),
                               'GradientAngles/Rescaled_Steps': (TimeVec[Comp]*self.params['batch_size']/self.params['learning_rate']),
                               'Performance_measures/True_Steps_+_1': TimeVec[Comp]+1})   
                    Grad_data.append(self.model.StepGradientClassNorm[k][Comp+1])
                
                AngleIndex=0
                for i,j in((i,j) for i in range(self.model.num_classes) for j in range(i)):
                    wandb.log({'GradientAngles/Gradient_Angles_Between_Classes_{}_and_{}'.format(i, j): (self.model.PCGAngles[AngleIndex][Comp+1] ),
                               #'GradientAngles/Gradient_Angles_NormComp_Between_Classes_{}_and_{}'.format(i, j): (self.model.GradAnglesNormComp[AngleIndex][Comp] ),
                               'GradientAngles/Rescaled_Steps': (TimeVec[Comp]*self.params['batch_size']/self.params['learning_rate']),
                               'Performance_measures/True_Steps_+_1': TimeVec[Comp]+1})
                    Grad_data.append(self.model.PCGAngles[AngleIndex][Comp+1])
                    AngleIndex+=1 

                self.Performance_data_table.add_data(*copy.deepcopy(Performance_data))
                self.Grad_data_table.add_data(*copy.deepcopy(Grad_data))

        
    def Gradient_Norms_logs(self, TimeComp):
        """
        here we log the norm of the gradient (total and per-class) in wandb

        Parameters
        ----------
        TimeComp : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        wandb.log({'Check/Gradient_Total_Norm': self.total_norm,
                   'Check/Epoch': self.params['epoch']})        
        for i in range(0, self.model.num_classes):
            wandb.log({'Check/Mean_Gradient_Norm_OfClass_{}'.format(i):  self.model.ClassesGradientNorm[TimeComp][i],
                       'Check/Epoch': self.params['epoch']})       
            

    def Histo_logs(self, n0, n1, n1_n0):
        """
        Here we save, as a sanity check the number of class element in each batch and their ratio.
        for now we collect them all togheter, but it is possible to save the histograms epochs bt epochs to see how does it change

        Parameters
        ----------
        n0 : TYPE
            DESCRIPTION.
        n1 : TYPE
            DESCRIPTION.
        n1_n0 : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        """
        print(n0)
        print('LA RESA DEI CONTI')
        print(np.array(n0))
        print(np.array(n1))
        print(np.array(n1_n0))
        
        hist_n0 = np.histogram(np.array(n0), density=True, bins = 'auto')
        hist_n1 = np.histogram(np.array(n1), density=True, bins = 'auto')            
        hist_n1_n0 = np.histogram(np.array(n1_n0), density=True, bins = 'auto')  
        
        wandb.log({"Histo/n0": wandb.Histogram(np_histogram=hist_n0)})
        wandb.log({'Histo/n1': wandb.Histogram(np_histogram=hist_n1)})
        wandb.log({'Histo/n1_n0': wandb.Histogram(np_histogram=hist_n1_n0)})
        """

        
        n0_table = wandb.Table(data=n0, columns=["n0"])
        n1_table = wandb.Table(data=n1, columns=["n1"])       
        #n1_n0_table = wandb.Table(data=n1_n0, columns=["n1_n0"])    
        
        histogram_n0 = wandb.plot.histogram(n0_table, value='n0', title='Histogram_n0')
        histogram_n1 = wandb.plot.histogram(n1_table, value='n1', title='Histogram_n1')
        #histogram_n1_n0 = wandb.plot.histogram(n1_n0_table, value='n1_n0', title='Ratio_Histogram')
        
        wandb.log({'Classes_Presence_Histo/histogram_n0': histogram_n0, 
                   'Classes_Presence_Histo/histogram_n1': histogram_n1 
                   #,'Classes_Presence_Histo/histogram_n1_n0': histogram_n1_n0
                   })
        
        
        
        #saving matplotlib plot
        #we first convert list into numpy array, we flatten it and finally we plot the histo and log it
        n0_arr = (np.array(n0)).flatten()
        n1_arr = (np.array(n1)).flatten()        
        #n1_n0_arr = (np.array(n1_n0)).flatten()    
        kwargs = dict(histtype='stepfilled', alpha=0.3, bins='auto',density=True)
        
        plt.hist(n0_arr, **kwargs, label = r'$n_0$')
        plt.title(r"$n_0$ Distribution")
        plt.legend(loc='best', fontsize=7)
        wandb.log({"Classes_Presence_Histo/histo_n0": wandb.Image(plt)})
        plt.show()
        plt.clf()
        
        plt.hist(n1_arr, **kwargs, label = r'$n_1$')
        plt.title(r"$n_1$ Distribution")
        plt.legend(loc='best', fontsize=7)
        wandb.log({"Classes_Presence_Histo/histo_n1": wandb.Image(plt)})
        plt.show()
        plt.clf()
        
        """
        plt.hist(n1_n0_arr, **kwargs, label = r'$\frac{n_1}{n_0}$')
        plt.title(r"$\frac{n_1}{n_0}$ Distribution")
        plt.legend(loc='best', fontsize=7)
        wandb.log({"Classes_Presence_Histo/histo_n1_n0": wandb.Image(plt)}) 
        plt.show()  
        plt.clf()            
        """
        
        """
        hist_n0 = np.array(n0)
        hist_n1 = np.array(n1)            
        hist_n1_n0 = np.array(n1_n0)  
        
        
        wandb.log({"Histo/n0": wandb.Histogram(hist_n0)})
        wandb.log({'Histo/n1': wandb.Histogram(hist_n1)})
        wandb.log({'Histo/n1_n0': wandb.Histogram(hist_n1_n0)})
        """


        
    def CustomizedX_Axis(self):
        """
        Set the default x axis to assign it on each group of logged measures

        Returns
        -------
        None.

        """
        wandb.define_metric("MeanSquare_Distance/t")
        # set all other MeanSquare_Distance/ metrics to use this step
        wandb.define_metric("MeanSquare_Distance/*", step_metric="MeanSquare_Distance/t")

        wandb.define_metric("Overlap/t")
        # set all other Overlap/ metrics to use this step
        wandb.define_metric("Overlap/*", step_metric="Overlap/t")        
        
        wandb.define_metric("Performance_measures/True_Steps_+_1")
        # set all other MeanSquare_Distance/ metrics to use this step
        wandb.define_metric("Performance_measures/*", step_metric="Performance_measures/True_Steps_+_1")    
        
        wandb.define_metric("Grad_Overlap/*", step_metric="Grad_Overlap/Steps")
        
        
        wandb.define_metric("Check/Epoch")
        # set all other MeanSquare_Distance/ metrics to use this step
        wandb.define_metric("Check/*", step_metric="Check/Epoch")         
        wandb.define_metric("Check/Step_Norm", step_metric="Check/True_Steps_+_1")  
        wandb.define_metric("GradientAngles/*", step_metric="GradientAngles/True_Steps_+_1")  


    def RAM_check(self,line_number):
        """
        print the RAM IN Gb on a file

        Parameters
        ----------
        line_number : line number where the function is called
            DESCRIPTION.

        Returns
        -------
        None.

        """
        print("the amount of RAM used in line ~~{}~~ of the code MainBlock.py (PID: {}) is:  ".format(line_number, os.getpid()) , psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, flush = True, file = self.params['memory_leak_file_object']) 


    def LineNumber(self):
        """
        return the line number of the code

        Returns
        -------
        None.

        """
        cf = currentframe()
        return cf.f_back.f_lineno        

    def TorchCheckpoint(self):
        """
        Create/update the checkpoint of the status

        Returns
        -------
        None.

        """
        #TODO: VERIFY THAT THE SAVING AND LOAD WORKS PROPERLY (WE MODIFIED THE MODEL, CHARGING VARIABÒES ON IT)
        #WARNING: for some models a snapshot of the last epoch could not be enough; there could be a dependence also from the previous state of the net 
        #saving the model () at the end of the run with
        torch.save({'model_state_dict' : self.model.state_dict(),
                    'optimizer_state_dict' : self.optimizer.state_dict(), #When saving a general checkpoint, you must save more than just the model’s state_dict. It is important to also save the optimizer’s state_dict, as this contains buffers and parameters that are updated as the model trains.
                    'epoch': self.params['epoch'],  #Other items that you may want to save are the epoch at which you stopped
                    'step': self.params['IterationCounter'],  #or number of iteration
                    'OldNSteps': self.params['NSteps'],
                    'TimeComp': self.params['TimesComponentCounter'],  #or number of evaluation block encountered
                    #TODO: AGGIUNGI I TEMPI PER IL CALCOLO DELLE FUNZIONI DI CORRELAZIONE (Tw)
                    'proj_id': self.params['ProjId'],
                    #metrix saved at the latest component updated
                    'OldTP':self.model.TP[:,:self.params['TimesComponentCounter']],
                    'OldTN':self.model.TN[:,:self.params['TimesComponentCounter']],
                    'OldFN':self.model.FN[:,:self.params['TimesComponentCounter']],
                    'OldFP':self.model.FP[:,:self.params['TimesComponentCounter']],
                    'OldTotP':self.model.TotP[:,:self.params['TimesComponentCounter']],
                    'OldTimeVector': self.Times,
                    'OldTrainPrec': self.model.Prec[:,:self.params['TimesComponentCounter']],
                    'OldTrainRecall': self.model.Recall[:,:self.params['TimesComponentCounter']],
                    'OldTrainF_Measure':self.model.FMeasure[:,:self.params['TimesComponentCounter']],
                    'OldTrainAngles':self.model.TrainAngles[:,:self.params['TimesComponentCounter']+1],
                    'OldTestAngles':self.model.TestAngles[:,:self.params['TimesComponentCounter']],
                    'OldRepresentationClassesNorm':self.model.RepresentationClassesNorm[:,:self.params['TimesComponentCounter']+1],
      
                    'OldTrainLoss':self.model.TrainLoss,#unidimensional measure are stored in list over which we append time by time
                    'OldTrainAcc':self.model.TrainAcc,
                    'OldTrainClassesLoss':self.model.TrainClassesLoss[:,:self.params['TimesComponentCounter']+1],
                    'OldTrainClassesAcc':self.model.TrainClassesAcc[:,:self.params['TimesComponentCounter']+1],
                    'OldValidLoss':self.model.ValidLoss,
                    'OldValidAcc':self.model.ValidAcc,
                    'OldValidClassesLoss':self.model.ValidClassesLoss[:,:self.params['TimesComponentCounter']+1],
                    'OldValidClassesAcc':self.model.ValidClassesAcc[:,:self.params['TimesComponentCounter']+1],
                        #we save always also the test measure, then depending if we are in the test or valid mode they could be meaningful or not
                    'OldTestLoss':self.model.TestLoss,
                    'OldTestAcc':self.model.TestAcc,
                    'OldTestClassesLoss':self.model.TestClassesLoss[:,:self.params['TimesComponentCounter']+1],
                    'OldTestClassesAcc':self.model.TestClassesAcc[:,:self.params['TimesComponentCounter']+1],                                   
                    }, self.params['FolderPath'] +'/model.pt')
        #just after the update of pytorch model we update the version stored in wandb
        wandb.save(self.params['FolderPath'] +'/model.pt') #save the model at the end of simulation to restart it from the end point         


    def RecallOldVariables(self, checkpoint):
        """
        This method is called when we restart an old simulation from its checkpoint; we store the variables collected in that simulation from the old files.
        The variables are stored in the model.pt file (checkpoint file) so that you don't need the variables file to recall it.
        
        The 1-d variables (like TrainLoss) are stored in list; we can then just assign the old variable to the new variable and keep appending new measures
        The multidimensional variables (like Classes measures) are stored in numpy array; in this case we can merge the old variables and new initialized vector using numpy.concatenate((a1, a2, ...), axis=0, out=None, dtype=None, casting="same_kind")
        
        NOTE: during checkpoint all vector are stored as they are, without any transposistion or modification in their shape, so you don't have to apport any changes for the merging
        Parameters
        ----------
        checkpoint : dict 
            dict of variables stored from checkpoint of previous simulation
        Returns
        -------
        None.
        """
        
        self.model.TP = np.concatenate((checkpoint['OldTP'],self.model.TP), axis=1)
        self.model.TN = np.concatenate((checkpoint['OldTN'],self.model.TN), axis=1)
        self.model.FP = np.concatenate((checkpoint['OldFP'],self.model.FP), axis=1)
        self.model.FN = np.concatenate((checkpoint['OldFN'],self.model.FN), axis=1)
        self.model.TotP = np.concatenate((checkpoint['OldTotP'],self.model.TotP), axis=1)
        
        self.model.Prec = np.concatenate((checkpoint['OldTrainPrec'],self.model.Prec), axis=1)
        self.model.Recall = np.concatenate((checkpoint['OldTrainRecall'],self.model.Recall), axis=1)
        self.model.FMeasure = np.concatenate((checkpoint['OldTrainF_Measure'],self.model.FMeasure), axis=1)
        self.model.TrainAngles = np.concatenate((checkpoint['OldTrainAngles'],self.model.TrainAngles), axis=1)
        self.model.TestAngles = np.concatenate((checkpoint['OldTestAngles'],self.model.TestAngles), axis=1)
        self.model.RepresentationClassesNorm = np.concatenate((checkpoint['OldRepresentationClassesNorm'],self.model.RepresentationClassesNorm), axis=1)
        self.model.ClassesGradientNorm = np.concatenate((checkpoint['OldClassesGradientNorm'],self.model.ClassesGradientNorm), axis=0) #note that here we concatenate along axis 0 because order fo classes and times are reversed
        self.model.TrainLoss = checkpoint['OldTrainLoss'] #for list variable we don't need to concatenate, since there is no initaliazed vector (we build it concatenating measures time by time)
        self.model.TrainAcc = checkpoint['OldTrainAcc']
        self.model.TrainClassesLoss = np.concatenate((checkpoint['OldTrainClassesLoss'],self.model.TrainClassesLoss), axis=1)
        self.model.TrainClassesAcc = np.concatenate((checkpoint['OldTrainClassesAcc'],self.model.TrainClassesAcc), axis=1)
        self.model.ValidLoss = checkpoint['OldValidLoss']
        self.model.ValidAcc = checkpoint['OldValidAcc']
        self.model.ValidClassesLoss = np.concatenate((checkpoint['OldValidClassesLoss'],self.model.ValidClassesLoss), axis=1)
        self.model.ValidClassesAcc = np.concatenate((checkpoint['OldValidClassesAcc'],self.model.ValidClassesAcc), axis=1)
        self.model.TestLoss = checkpoint['OldTestLoss']
        self.model.TestAcc = checkpoint['OldTestAcc']
        self.model.TestClassesLoss = np.concatenate((checkpoint['OldTestClassesLoss'],self.model.TestClassesLoss), axis=1)
        self.model.TestClassesAcc = np.concatenate((checkpoint['OldTestClassesAcc'],self.model.TestClassesAcc), axis=1)










