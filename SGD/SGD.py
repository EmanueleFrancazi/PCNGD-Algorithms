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
#import tensorboard
from torch.utils.tensorboard import SummaryWriter
#import wandb
import wandb
# for data visualization
import matplotlib.pyplot as plt
#to set network architecture
import torch.nn as nn
import torch.nn.functional as F

import psutil
from inspect import currentframe
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
import CodeBlocks
#torch.set_default_tensor_type(torch.DoubleTensor)
#module to fix the seed (for repeat the experiment)
import random
import sys

torch.set_printoptions(precision=17)

#%% FLAG, MODES VARIABLES AND PARAMETERS




def FixSeed(seed):
    """
    initialize the seed for random generator used over the run : we have to do it for all the libraries that use on random generator (yorch, random and numpy)

    Parameters
    ----------
    seed : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    torch.manual_seed(seed) #fixing the seed of 'torch' module
    random.seed(seed) #fixing the seed of 'random' module
    np.random.seed(seed) #fixing the seed of 'numpy' module   

#HYPERPARAMETERS
CheckMode = 'OFF' #this flag active ('ON') or deactive ('OFF') the checking mode (used for debugging purposes)
#NOTE:Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms. 
#Furthermore, results may not be reproducible between CPU and GPU executions, even when using identical seeds.
#However, there are some steps you can take to limit the number of sources of nondeterministic behavior for a specific platform, device, and PyTorch release.
if CheckMode=='ON':#when we are in the checking mode we want to reproduce the same simulation to check the new modified code reproduce the same behaviour
    seed = 0
    FixSeed(seed)
elif CheckMode=='OFF':
    #creation of seeds for usual simulations
    #WARNING: use the time of the machine as seed you have to be sure that also for short interval between successive interval you get different seeds
    #with the following choice  for very short periods of time, the initial seeds for feeding the pseudo-random generator will be hugely different between two successive calls
    t = int( time.time() * 1000.0 )
    seed = ((t & 0xff000000) >> 24) + ((t & 0x00ff0000) >>  8) + ((t & 0x0000ff00) <<  8) + ((t & 0x000000ff) << 24)   
    
    #if the above syntax should be confusing:
    """
    Here is a hex value, 0x12345678, written as binary, and annotated with some bit positions:
    
    |31           24|23           16|15            8|7         bit 0|
    +---------------+---------------+---------------+---------------+
    |0 0 0 1 0 0 1 0|0 0 1 1 0 1 0 0|0 1 0 1 0 1 1 0|0 1 1 1 1 0 0 0|
    +---------------+---------------+---------------+---------------+
    
    ...and here is 0x000000FF:
    
    +---------------+---------------+---------------+---------------+
    |0 0 0 0 0 0 0 0|0 0 0 0 0 0 0 0|0 0 0 0 0 0 0 0|1 1 1 1 1 1 1 1|
    +---------------+---------------+---------------+---------------+
    
    So a bitwise AND selects just the bottom 8 bits of the original value:
    
    +---------------+---------------+---------------+---------------+
    |0 0 0 0 0 0 0 0|0 0 0 0 0 0 0 0|0 0 0 0 0 0 0 0|0 1 1 1 1 0 0 0|
    +---------------+---------------+---------------+---------------+
    
    ...and shifting it left by 24 bits moves it from the bottom 8 bits to the top:
    
    +---------------+---------------+---------------+---------------+
    |0 1 1 1 1 0 0 0|0 0 0 0 0 0 0 0|0 0 0 0 0 0 0 0|0 0 0 0 0 0 0 0|
    +---------------+---------------+---------------+---------------+
    
    ...which is 0x78000000 in hex.
    
    The other parts work on the remaining 8-bit portions of the input:
    
      0x12345678
    & 0x000000FF
      ----------
      0x00000078 << 24 = 0x78000000       (as shown above)
    
      0x12345678
    & 0x0000FF00
      ----------
      0x00005600 <<  8 = 0x00560000
    
      0x12345678
    & 0x00FF0000
      ----------
      0x00340000 >>  8 = 0x00003400
    
      0x12345678
    & 0x00000000
      ----------
      0x12000000 >> 24 = 0x00000012
    
                       | ----------
                         0x78563412
    
    so the overall effect is to consider the 32-bit value ldata as a sequence of four 8-bit bytes, and reverse their order.
        
    """
    
    
    
    FixSeed(seed)
    

StopFlag='OFF' #this flag active ('ON') or deactive ('OFF') the early stopping (at a time specified by variable 'StopPoint') to simulate an interruption (debug purpose)
StopPoint = 30

#the 2 following variables are only to easly set same times (to compare continuous and interrupted runs) for checkout testing 
n_epochsComp = 10
NStepsComp = 10


#code computation parameters
# number of subprocesses to use for data loading
num_workers = 0 # 0 uses automatically the max number available
#DEVICE CHOICE
# Get cpu or gpu device for training throught the following flag variable (the choice can be either 'CPU' or 'GPU')
Set_Device = 'GPU'
#note: to use gpu you have to specify below the corresponding index in case of multiple choice: e.g. device = "cuda:1" will use the GPU with index "1" in the server

if Set_Device=='CPU':
#CPU
    device = "cpu"
elif Set_Device=='GPU':
#GPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"    #device = "cuda"
print("Using {} device".format(device))

#for large dataset the run times can be very longs; we therefore define the flag variable Cheap_Mode 
#the idea is to avoid the evaluation of training curves since in the valid phase we are interested in the performances of validation set
#if Cheap_Mode=='ON' we don't evaluate the train performances, if instead Cheap_Mode=='OFF' we do it
Cheap_Mode='ON'

# time parameters (run extention, number of points,...)
n_epochs = 10 #8000#3000#400 # number of epochs to train the model
epoch=0 #initialization of value of current epoch
#we want to express the time in steps, not in epoch: Nbatch*Nepoch = Nsteps, with Nbatch = Nsamples/Batchsize (and the batch size is the one used in SGD at each gradient computation)

NSteps = 20

#extention for autocorrelation observables
Ntw = 10
Nt = 15

#hyperparameters for algorithm (setting Learning rate (set by external parameter (from parent bash script)), momentum...)
lr_decay = 0 #1e-6 #parameter for the lr_decay at each step
lr_drop = np.inf #n_epochs*2 #this parameter set the number of epochs after which perform the lr variation (se lr_schedule in CodeBlocks.py)
weight_decay= 0 #0.0005
momentum = 0. #0.9

SphericalRegulizParameter = 0.060 #regularization parameter for the spherical constraint
FreqUpdateDetDir = 1 #we compute the deterministic direction every FreqUpdateDetDir steps to save computation time


#check variables
batches_num =0 #this variable counts the number of non-trashed batches (for example when we deal with PCN alg. we have to trash batches if they doesn't contain at least one element from each class)




#FLAG VARIABLES
#SPERICAL CONSTRAIN FLAG; CAN BE ONLY ON OR OFF
SphericalConstrainMode = 'OFF'  #(WARNING: feature not implemented yet, ignore it for now)
ClassSelectionMode = 'ON' #can be set either 'OFF' or 'ON'. setting this flag 'ON' to modify the default composition of a dataset (excluding some classes) 
ClassImbalance = 'ON' #can be set either 'OFF' or 'ON'. setting this flag 'ON' to set imbalance ratio between classes of the dataset 
MacroMode =  'Cats_&_Dogs' #'CIFAR100' #'CIFAR100SC' #Set the desired configuration for the composition of the modified dataset. The selected classes (and their relative abboundance (in case of imbalance)) can be set by LM and IR dict (see below)
ValidMode = 'Test' #('Valid' or 'Test') #can be valid or test and selsct different part of a separate dataset used only for testing/validating 

IR = {'Cats_&_Dogs': 1., 'ON': 1./60, 'OFF': 1./7, 'MULTI': 1, 'DH': 1./7, 'MultiTest': 1./3, '0_4': 0.6, 'CIFAR100SC':0.85, 'INaturalist':1., 'CIFAR100':0.955, 'C100':1.} #IR = {'ON': 1./60, 'OFF': 1./7, 'MULTI': 0.6, 'DH': 1./7, 'MultiTest': 1./3, '0_4': 0.6} #we define the dictionary IR to automatically associate the right imbalance ratio to the selected MacroMode


Dynamic = 'SGD' #algorithm selection 

FFCV_Mode = 'OFF' #this flag trigger the using of ffcv library to speed up the simulations (WARNING: feature not implemented yet, ignore it for now)

SetFlag = 'Train' #this flag specify which dataset we forward; can be 'Train', 'Test' or 'Valid' (do not change it, is automatically updated over the course of the simulation. the value set in this line constitutes the initial initialization)

IGB_flag = 'ON'

TrainFlag ='OFF'

#we specify throught the following dict wich algorithms perform resampling and which not
OversamplingMode = {'SGD': 'OFF', 'BISECTION': 'OFF', 'PCNSGD': 'OFF', 'PCNSGD+O': 'ON', 'SGD+O': 'ON', 'PCNSGD+R': 'OFF', 'GD':'OFF', 'PCNGD': 'OFF'}
#we specify throught the following dict wich algorithms is stochastic (forward each time only a part of the dataset (batch)) and which not
StochasticMode = {'SGD': 'ON', 'BISECTION': 'ON', 'PCNSGD': 'ON', 'PCNSGD+O': 'ON', 'SGD+O': 'ON', 'PCNSGD+R': 'ON', 'GD':'OFF', 'PCNGD': 'OFF'}

#class selection variables
#TIPS FOR CLASS MAPPINGS (TO MAKE THINGS EASIER AND CLEARER)
# - as a rule we map our class in the list (0,1,2,...) The mapped list is in descending order of elements; i.e. 0 is the majority class, 1 the second and so on

# since to establish the mapping for each input label we have to sequentially (taking each input label value one by one) modify the values of the targets we have to avoid crossing

UnbalanceFactor = IR[MacroMode]
if(ClassSelectionMode=='ON'):
    
    CIFAR100_coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  
                           3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                           6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  
                           0, 11,  1, 10, 12, 14, 16,  9, 11,  5, 
                           5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 
                           16, 4, 17,  4,  2,  0, 17,  4, 18, 17, 
                           10, 3,  2, 12, 12, 16, 12,  1,  9, 19,  
                           2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 
                          16, 19,  2,  4,  6, 19,  5,  5,  8, 19, 
                          18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
    
    CIFAR100_SuperClass_Dict = {}
    for classes in range(0,100):
        CIFAR100_SuperClass_Dict[classes] = CIFAR100_coarse_labels[classes]
    print(CIFAR100_SuperClass_Dict)
    
    C100TrivialDict = {}
    for i in range(0,100):
        C100TrivialDict[i] = i
    print(C100TrivialDict)    
    
    
    LM = {'ON': {0:1, 1: 1, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:1, 9: 1}, 'OFF': {1: 0, 9: 1}, 'MULTI': {0:0, 1: 1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9: 9}, 'DH': {7:0, 4:1}, 'MultiTest' : {0:2, 1:1, 2:0}, '0_4': {0:0, 1: 1, 2:2, 3:3, 4:4}, 'CIFAR100SC':CIFAR100_SuperClass_Dict, 'INaturalist': {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12}, 'Cats_&_Dogs':{3:0, 5:1}, 'C100': C100TrivialDict } #with philum we have in total 13 classes   'INaturalist': {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7}}
    
    label_map = LM[MacroMode]#{1: 0, 9: 1} #{0:1, 1: 1, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:1, 9: 1} #{start_class:mapped_class}
    ClassesList = list(label_map.values()) #sarà utile per l'embedding in tensorboard
    #to derive the number of classes, since we may want to map different classes in a single one, we count the number of different item (mapped classes) in the mapping dict
    #Because sets cannot have multiple occurrences of the same element, it makes sets highly useful to efficiently remove duplicate values from a list or tuple and to perform common math operations like unions and intersections.  
    num_classes = len(set(ClassesList))
    #we finally count the number of occurence of input classes for each mapped one
    MappedClassOcc = np.zeros(num_classes)
    for MC in set(ClassesList):
        MappedClassOcc[MC] = (sum(1 for value in label_map.values() if value == MC))
    
    print("the number of effective classes (after the label mapping) is {}".format(num_classes))
    
elif(ClassSelectionMode=='OFF'):
    #we set preliminarly the following variables to define the params dict and then we update them with the correct values inferred directly from the dataset
    label_map = 'Dummy parameter'
    num_classes = 10
    MappedClassOcc = np.zeros(num_classes)
    
#define now an array of imabalance ratio for the multiclass case
ImabalnceProportions = np.zeros(num_classes)
for i in range (0,num_classes):
    ImabalnceProportions[i] = UnbalanceFactor**i


#in some datset (as INaturalist) there is multi-labelling and it is necessary to specify which label to use
label_type = 'phylum'
#label_type = 'kingdom'



StartMode = 'BEGIN' #this flag variable rule if the simulation start from 0 ('BEGIN') or if it continue a past interrupted run ('RETRIEVE')


#transform the input value token from outside rin a variable
p = argparse.ArgumentParser(description = 'Sample index')
p.add_argument('SampleIndex', help = 'Sample index')
p.add_argument('FolderName', type = str, help = 'Name of the main folder where to put the samples folders')
p.add_argument('Dataset', type = str, help = 'Dataset to use (MNIST or CIFAR10)')
p.add_argument('Architecture', type = str, help = 'Architecture to use for the NN')
p.add_argument('DataFolder', type = str, help = 'Path for the dataset folder')
p.add_argument('LR', type = float, help = 'learning rate used in the run')
p.add_argument('BS', type = int, help = 'batch size used in the run')
p.add_argument('GF', type = float, help = 'parameter for the group norm used in the run')    
p.add_argument('DP', type = float, help = 'Probability parameter used for the dropout')
args = p.parse_args()


print('first parameter (run index) passed from the script is: ', args.SampleIndex)
print('second parameter (Output Folder) passed from the script is: ', args.FolderName)
print('third parameter (dataset) passed from the script is: ', args.Dataset)
print('fourth parameter (architecture) passed from the script is:', args.Architecture)
print('fifth parameter (dataset folder) passed from the script is:', args.DataFolder)
print('sixth parameter (learning rate) passed from the script is: ', args.LR)
print('seventh parameter (batch size) passed from the script is: ', args.BS)
print('eighth parameter (Group norm parameter) passed from the script is : ', args.GF)
print('nineth parameter (dropout prob. parameter)  passed from the script is :', args.DP)               
#we perform all the program SamplesNumber times

#to have the complete printing of long numpy vector/ pytorch tensor on file
np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=np.inf)

learning_rate = args.LR
batch_size = args.BS
group_factor = args.GF
dropout_p =  args.DP

#if you are using a NN architecture that doesn't require this parameters you can just set it with a negative value (which is normally a non valid range for the parameters)
if args.GF < 0:
    group_factor = 'None'
if args.DP < 0:
    dropout_p = 'None'



#we first create the folder associated to the sample and then save inside the folder all the files
#we start by creating the path for the folder to be created
#we first create the parameter folder
FolderPath = './'+ args.FolderName +'/lr_{}_Bs_{}_GF_{}_DP_{}'.format(learning_rate, batch_size, group_factor, dropout_p)
if not os.path.exists(FolderPath):
    os.makedirs(FolderPath, exist_ok=True)         
#then we create the specific sample folder
FolderPath = './'+ args.FolderName +'/lr_{}_Bs_{}_GF_{}_DP_{}'.format(learning_rate, batch_size, group_factor, dropout_p)  + '/Sample' + str(args.SampleIndex)
print('La cartella creata per il sample ha come path: ', FolderPath)
if not os.path.exists(FolderPath):
    os.makedirs(FolderPath, exist_ok=True) 

DebugFolderPath = FolderPath + '/Debug'
if not os.path.exists(DebugFolderPath):
    os.makedirs(DebugFolderPath, exist_ok=True) 


#I create the files where I store the outputs of the various prints
info = open(DebugFolderPath + "/InfoGenerali.txt", "a") 
#file for the epoch values
EpochValues = open(FolderPath + "/PerformancesValues.txt", "a")
#file for the execution time of each epoch
ExecutionTimes = open(DebugFolderPath + "/ExecutionTimes.txt", "a")

#file for the correlation measures
CorrPrint = open(FolderPath + "/CorrelationsPrint.txt", "a")

#file for debug purpose
DebugFile = open(DebugFolderPath + "/DebugChecks.txt", "a")

WarningFile = open(DebugFolderPath + "/Warnings.txt", "a")

StepNorm =  open(FolderPath + "/StepNorm.txt", "a")

memory_leak = open(DebugFolderPath + "/MemoryHistoryLog.txt", "a") 


#fixing initial times to calculate the total time of the cycle
start_TotTime = time.time()
    
#%% CREATION OF THE CLASS INSTANCE REPRESENTING THE NETWORK


#%%% Dict of input parameters for the model
params = {'Dynamic': Dynamic,  'FolderPath': FolderPath,  'info_file_object' : info, 'EpochValues_file_object': EpochValues, 
          'ExecutionTimes_file_object' : ExecutionTimes,  'memory_leak_file_object': memory_leak,
          'CorrPrint_file_object' : CorrPrint, 'DebugFile_file_object' : DebugFile , 'WarningFile' : WarningFile, 
          'StepNorm_file_object' : StepNorm , 
          'NetMode' : args.Architecture, 'ClassImbalance' : ClassImbalance , 'SphericalRegulizParameter' : SphericalRegulizParameter,
          'DataFolder': args.DataFolder ,'ClassSelectionMode' : ClassSelectionMode, 'SphericalConstrainMode' : SphericalConstrainMode, 
          'CheckMode' : CheckMode, 'n_epochsComp':n_epochsComp, 'NStepsComp':NStepsComp,
          'n_out' : num_classes , 'label_map' : label_map , 'NSteps' : NSteps , 'n_epochs' : n_epochs, 
          'UnbalanceFactor' : UnbalanceFactor, 'ImabalnceProportions' : ImabalnceProportions,
          'Dataset' : args.Dataset, 'device' : device, 'SampleIndex': args.SampleIndex,
          'group_factor': group_factor,'learning_rate' : learning_rate, 'dropout_p': dropout_p,
          'lr_drop': lr_drop,'batch_size' : batch_size, 'momentum': momentum, 'weight_decay' : weight_decay, 
          'num_workers' : num_workers,  'epoch' : epoch, 'Ntw' : Ntw, 'Nt' : Nt,
          'CheckMode': CheckMode,'StartMode': StartMode, 'MacroMode': MacroMode, 'ValidMode': ValidMode,
          'OversamplingMode': OversamplingMode[Dynamic],'StochasticMode':StochasticMode[Dynamic] ,'MappedClassOcc': MappedClassOcc,
          'IterationCounter': 0, 'TimesComponentCounter':0,  'TwComponentCounter':0 , 'ProjId': None
          , 'label_type': label_type
          , 'IGB_flag': IGB_flag}


print('the folder root', params['DataFolder'])


if(ClassSelectionMode=='OFF'):
    
            # tomake the code flexible we trat this case as a specific case of the classe selection in which all the classes are selected:
            # we start determining the total number of classes
        
    #we create a dummy instance just to evaluate the total number of classes
    FakeNet = CodeBlocks.DatasetTrial(params)
    FakeNet.DataTempCopy()


    num_classes = len(torch.unique(torch.Tensor(FakeNet.train_data.targets)))
    #print(FakeNet.train_data.targets)
    print("number of classes : ", num_classes)
    label_map = {}
    for i in range(0, num_classes):
        label_map[i]=i
        
    ClassesList = list(label_map.values()) #sarà utile per l'embedding in tensorboard
    MappedClassOcc = np.zeros(num_classes)
    for MC in set(ClassesList):
        MappedClassOcc[MC] = (sum(1 for value in label_map.values() if value == MC))       
           
            
    #define now an array of imabalance ratio for the multiclass case
    ImabalnceProportions = np.zeros(num_classes)
    for i in range (0,num_classes):
        ImabalnceProportions[i] = UnbalanceFactor**i
        
    #now we just update theitem of the previous dict and we can pass to the real network
    params['label_map'] = label_map  
    params['ImabalnceProportions']= ImabalnceProportions
    params['MappedClassOcc']= MappedClassOcc
    params['n_out']= num_classes




#if you include the HP looping inside the code at each "hyper-params iteration" makes a new object and assign it to the variable NetInstance
#The old instance is not referenced anymore, and you cannot access it anymore. So in each loop you have a new fresh instance.
#since the loops is now on the script that calls this code we don't have to think about what said in the above lines

#%%% creation of model instance 

NetInstance = CodeBlocks.Bricks(params) #we create a network by creating an istance of a class contained in CodeBlocks

NetInstance.NetLoad() #load model on device and initialize optimizer and loss function
#here we see if we want to start a new simulation or if we are continuing an old one
#the elements required for retrieve a started simulation are (the model state, the old Proj_id) 

if (StartMode=='BEGIN'):
    StartEpoch = 0
    StartIterationCounter=0
    StartTimesComponentCounter=0
    StartTwComponentCounter=0
    NetInstance.params['ProjId'] = wandb.util.generate_id()
    PreviousTimes=None
elif(StartMode=='RETRIEVE'):
    #here we retrieve the model and state of the old checkpoint; first initialize the model and optimizer, then load the dictionary locally.
    checkpoint = torch.load(FolderPath +'/model.pt') #load the model (substitute with the right pathas argument)
    NetInstance.model.load_state_dict(checkpoint['model_state_dict'])
    NetInstance.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    NetInstance.params['ProjId'] = checkpoint['proj_id']
    StartEpoch = checkpoint['epoch']
    StartIterationCounter = checkpoint['step']  
    StartTimesComponentCounter=checkpoint['TimeComp'] 
    NetInstance.params['NSteps'] = checkpoint['OldNSteps'] #This is the true one; we substituted only for the check phase NSteps+checkpoint['OldNSteps']
    n_epochs = n_epochs + StartEpoch
    #we have also to recall the old variables:
    #we define the new vector 
    NetInstance.RecallOldVariables(checkpoint)
    PreviousTimes=checkpoint['OldTimeVector']
    
    
#printiing the id of the simulation in the info file 
print("the id of the run (wandb) is: {}".format(NetInstance.params['ProjId']), flush=True, file = info)
print("the seed used for the run (pythorc, random and numpy) is: {}".format(seed), flush=True, file = info)

#%%% remote recording init
#LOGging INIT
TB_path = 'TensorBoard'+'/lr_{}_Bs_{}_GF_{}_DP_{}'.format(learning_rate, batch_size, group_factor, dropout_p)  + '/Sample' + str(args.SampleIndex)                

#WANDB INIT
#initialize my run with project (name of the group of experiment the run belongs to), notes (brief commit message) entity (account associated) and reinit option (to init multiple run in the same script)
#note that each run (in case of multiple in the same script ends with run.finish() command)

#wandb.tensorboard.patch(root_logdir=TB_path + '/Corr', pytorch=True, tensorboardX=False)

#DASHBOARD SUBFOLDER
#wandb_dir = '/prova/EpsReg_Alg_{}_ImbRatio_{}_lr_{}_Bs_{}_GF_{}_DP_{}'.format(Dynamic, UnbalanceFactor, learning_rate, batch_size, group_factor, dropout_p)
#compl_wandb_dir = './wandb' + wandb_dir
#if not os.path.isdir(compl_wandb_dir):
#    os.makedirs(compl_wandb_dir)

if IGB_flag=='ON':
    PN = 'Rebuttal_IGB_{}'.format(args.Architecture)
elif IGB_flag=='OFF':
    PN = 'Rebuttal_NO_IGB_{}'.format(args.Architecture)
    
    
ProjName = PN#'Rebuttal NONO_IGB_{}'.format(args.Architecture) #'OPTIM_Net_{}'.format(args.Architecture) #'FINAL_Net_{}'.format(args.Architecture) #'OPTIM_Net_{}'.format(args.Architecture)  #'BALANCED_Test' #'MultiClass_Test'#'FINAL_Net_{}'.format(args.Architecture)#'MultiClass_Test' #'TestRetrieve' #'~~OPTIM_Net_CNN_Alg_PCNSGD+R'#'OPTIM_Net_{}'.format(args.Architecture) #  #~~F_Net_CNN_Alg_GD'  #'TestNewVersion' #'RETRIEVEProva'  #the project refers to all the simulations we would like to compare
GroupName = '/GaussInitAlg_{}_ImbRatio_{}_lr_{}_Bs_{}_GF_{}_DP_{}_MacroMode_{}~'.format( Dynamic, UnbalanceFactor, learning_rate, batch_size, group_factor, dropout_p, MacroMode)#'/~Alg_{}_ImbRatio_{}_lr_{}_Bs_{}_GF_{}_DP_{}_MacroMode_{}~'.format( Dynamic, UnbalanceFactor, learning_rate, batch_size, group_factor, dropout_p, MacroMode) #the group identifies the simulations we would like to average togheter for the representation
RunName = '/Sample' + str(args.SampleIndex)#'/Sample' + str(args.SampleIndex)  #the run name identify the single run belonging to the above broad categories

#we define a list of tags that we can use for group more easly runs on wandb
#we list all the relevant parameter as tag
tags = ["LR_{}".format(learning_rate), "BS_{}".format(batch_size), "GF_{}".format(group_factor), "DP_{}".format(dropout_p), "Alg_{}".format(Dynamic), "ClassesMode_{}".format(MacroMode)]

run = wandb.init(project= ProjName, #PCNSGD_vs_SGD #CNN_PCNSGD_VS_SGD
           group =  GroupName,#with the group parameter you can organize runs divide them into groups
           #job_type="ImbRatio_{}_lr_{}_Bs_{}_GF_{}_DP_{}_Classes_1_9".format(UnbalanceFactor,learning_rate, batch_size, group_factor, dropout_p) , #specify a subsection inside a simulation group
           #dir = compl_wandb_dir,
           tags = tags,
           notes="experiments to figure out why PCNSGD doesn't work with per-class norm division in the unbalance case",
           entity= "emanuele_francazi", #"gpu_runs", #
           name = RunName,
           id = NetInstance.params['ProjId'], #you can use id to resume the corresponding run; to do so you need also to express the resume argument properly
           resume="allow"
           #sync_tensorboard=True,
           ,reinit=True #If you're trying to start multiple runs from one script, add two things to your code: (1) run = wandb.init(reinit=True): Use this setting to allow reinitializing runs
           )


NetInstance.CustomizedX_Axis() #set the customized x-axis in a automatic way for all the exported charts



wandb.config = {
  "Algorithm": Dynamic,
  "Imbalance_ratio":UnbalanceFactor,
  "learning_rate": learning_rate,
  "epochs": n_epochs,
  "batch_size": batch_size,
  "Number_of_channel_grous_(group_norm)": group_factor,
  "Percentage_of_Elements_Dropped_out" : dropout_p,
  "Imbalance_ratio":UnbalanceFactor,
  "architecture": args.Architecture,
  "Dataset":args.Dataset
}

NetInstance.StoringGradVariablesReset() #clear the gradient copy and Norm variable before initial state


start_time = time.time()

#CREATION OF THE DATASET AND STATISTIC STATISTICS BEFORE TRAINING STARTS
NetInstance.DataLoad() #build dataset and load it on the device
if (StartMode=='BEGIN'):
    NetInstance.InitialState() #evaluation before starting training
elif (StartMode=='RETRIEVE'):
    NetInstance.DefineRetrieveVariables() #evaluation before starting training

print("---initialization %d last %s seconds ---" % (NetInstance.params['epoch'] , time.time() - start_time), flush=True, file = ExecutionTimes)


img, lab = next(iter(NetInstance.TrainDL['Class0']))
img = img.double()
img = img.to(device)
lab = lab.to(device)
# we want to convert the learning rate to a value independent from the dimension of the picture (and also from the batch size)
#usually you would set criterion = 'mean'; this would normalize the loss (and so the gradient) by the shape of the input tensor
#since with the new version of the code the classes are divided in different dataloader (and each of them could in principle have a different batch size),
#we want to avoid the use of 'mean' reduction and proceed by a first rescaling of the learning rate according to the image shape
#finally just before the step (so when many batches from different dataloader) have been forwarded and collected togheter, we normalize by the total number of element (sum of batches' size) involved in the step
#the image have all the same shape so:
    
    
#LEARNING RATE RESCALING
"""
np.prod(list(img[0].shape))
print("the image input shape is {}, we rescale the learning rate according to its product ({}) passing from {} to {}".format(img[0].shape, np.prod(list(img[0].shape)), learning_rate, learning_rate/np.prod(list(img[0].shape))), flush=True, file = info)
#rescale learning rate
for g in NetInstance.optimizer.param_groups:
    g['lr'] = g['lr']/np.prod(list(img[0].shape))
NetInstance.params['learning_rate'] = NetInstance.params['learning_rate']/np.prod(list(img[0].shape))
"""

#save the model representation on wandb
if NetInstance.params['NetMode']=='VGG_Custom_Dropout':
    torch.onnx.export(NetInstance.model, (img,0), 'model.onnx')
else:
    torch.onnx.export(NetInstance.model, img, 'model.onnx')
wandb.save('model.onnx')

#OPENING RESEVOIR FOR TENSOR BOARD SUMMARY
NetInstance.SummaryWriterCreation(TB_path) #we specify the subfolder path where to save all the files that will be used by tensorboard    

"""
DummyInput,_  = next(iter(NetInstance.train_loader))

writer.add_graph(NetInstance.model, DummyInput)    #write the net structure on the tensor board file

"""

N = sum(p.numel() for p in NetInstance.model.parameters() if p.requires_grad)
print(N, flush=True)


#%%  TRAIN THE NETWORK: PRELIMINARY SETTINGS

# initialize tracker for minimum validation loss, 
valid_loss_min = np.Inf  # set initial "min" to infinity
NetInstance.params['TimesComponentCounter'] = StartTimesComponentCounter
NetInstance.params['TwComponentCounter'] = 0
NetInstance.params['IterationCounter'] = StartIterationCounter


#number of learnable parameters (weights and, eventually, bias)
WeightsNumber = sum(p.numel() for p in NetInstance.model.parameters() if p.requires_grad)




if(SphericalConstrainMode=='ON'):
    NetInstance.CorrelationTempVariablesInit() #initialize variable to store temp state for correlation computation

NetInstance.cos_alpha=0
cos_alpha = NetInstance.cos_alpha
#SET THE LR SCHEDULE
#add a lr schedule to decrease the learning rate value during the training
#we use here the lambda function (or lambda expression )syntax 

lr_rule = lambda epoch: (0.1 ** (int(1+math.log10((epoch+1)/8)))) 
#lr_rule = lambda epoch: (0.5 ** (epoch // NetInstance.params['lr_drop'])) #here we make lr decay with the number of epochs

#lr_rule = lambda cos_alpha: (1+cos_alpha)

lr_schedule = torch.optim.lr_scheduler.LambdaLR(NetInstance.optimizer, lr_lambda = lr_rule)


"""
#DEBUG CHECK 
data, label = next(iter(NetInstance.train_loader)) #select only the first batch of the training loader for the debug (and verify if the net is able to overfit it)
dataval, labelval = data, label 
"""



#before starting the training we PRINT THE PID of the process (this can be useful if you have more process in parallel on the same machine to identify which one is which)
print("The PID of the main code is: ", os.getpid(), flush=True)
#saving the static (ones that keep constant during simulation) files on wandb dashboard
wandb.save('CodeBlocks.py')
wandb.save('MainBlock.py')
wandb.save('PythonRunManager.sh')



#BRANCHING POINT FOR DIFFERENT DYNAMICS        

#%% SGD
if (Dynamic=='SGD'):
    #%%% Times setting
    Times = CodeBlocks.Define(params, n_epochs, NSteps, len(NetInstance.TrainDL['Class0']), StartIterationCounter, PreviousTimes).StocasticTimes()
    print("the checks times (were the measures are collected) are: ", Times, flush=True, file = info)
    MaxStep = n_epochs*len(NetInstance.TrainDL['Class0']) #we calculate the number of steps equivalent to epochs by multiplying the epochs by the number of batches in the majority class dataloader
    
    NetInstance.Times = Times
    
    #save the real interesting times on file (log equispaced); but store variables of all times
    with open(NetInstance.params['FolderPath'] + "/time.txt", "a") as f:
        np.savetxt(f, Times, delimiter = ',')
    
    #PREPARING VRIABLES FOR THE CORRELATION ANALYSIS
    if(SphericalConstrainMode=='ON'):

        tw = CodeBlocks.Define(params, n_epochs, NSteps,len(NetInstance.TrainDL['Class0']), StartEpoch, PreviousTimes).FirstCorrTimes(Ntw, MaxStep)
        t = CodeBlocks.Define(params, n_epochs, NSteps,len(NetInstance.TrainDL['Class0']), StartEpoch, PreviousTimes).SecondCorrTimes(Ntw, Nt, tw, MaxStep, spacing_mode= 'linear')
        """
        #TODO: remove the below lines that set different times for a specific case
        tw = Times[:-1]
        t = Times[1:] - Times[:-1]
        """
        
        
        CorrTimes = CodeBlocks.Define(params, n_epochs, NSteps,len(NetInstance.TrainDL['Class0']), StartEpoch, PreviousTimes).CorrTimes(Ntw, Nt, tw, t)

        print('i tw are: ', tw, file = info)
        print('i t sono: ', t, file = info)
        print('Correlation times matrix is: ', CorrTimes, file = info)
        
        

    #%%% Saving Simulation ID
    
    NetInstance.SimulationID()
    
    NumberOfTrainBatches = len(NetInstance.TrainDL['Class0'])
    print('we have to check the number of batches in each class dataloader; if the oversampling is set to OFF we expect the same number of batches for each class')
    print("the oversampling mode is set on {}".format(params['OversamplingMode']), file = DebugFile)
    print("the number of batches in the dataset are: ", NumberOfTrainBatches, file = DebugFile)


    #%%%  computing initial bias
    OutMeanValue=torch.zeros(num_classes)

    InitGuess = np.zeros(num_classes)

    for EvalKey in NetInstance.TrainDL:
        SetFlag = 'Train' 
        
        
        for dataval,labelval in NetInstance.TrainDL[EvalKey]:
            #DatasetSize+=torch.numel(labelval)
    
            Mask_Flag = 1
            
            dataval = dataval.double() 
            dataval = dataval.to(device)
            labelval = labelval.to(device) 

            if NetInstance.params['NetMode']=='VGG_Custom_Dropout':
                
                NetInstance.DropoutBatchForward(dataval, Mask_Flag)
                Mask_Flag = 0
            else:
                NetInstance.BatchForward(dataval)


            NetInstance.pred=NetInstance.OutDict['pred'].clone()
            NetInstance.TrainPred = NetInstance.pred
            OutMeanValue+=torch.sum(NetInstance.OutDict['out'].clone(),0).detach().cpu().numpy()
            #print("PREDICTIONS: {}".format(NetInstance.TrainPred), flush=True)
            
            for i in range(0, num_classes):
                #print('Init Guess of class {} BEFORE update : {}'.format(i, InitGuess[i]), flush=True)
                InitGuess[i] += ((NetInstance.TrainPred==i).int()).sum().item()
                #print('Init Guess of class {} AFTER update : {}'.format(i, InitGuess[i]), flush=True)
    InitClassFraction = InitGuess/(NetInstance.TrainTotal)
    print('sum of fractions: ', np.sum(InitClassFraction))
    InitClassFx = OutMeanValue/(NetInstance.TrainTotal)


    #saving statistics

    with open("./GuessImbalance.txt", "a") as f:
        f.write('{}\n'.format(InitClassFraction[0]))
   
    with open('./ClassesGI.txt', "a") as f:
        np.savetxt(f, [InitClassFraction], delimiter = ',')
        
    with open("./Outup_Value.txt", "a") as f:
        np.savetxt(f, [InitClassFx.numpy()], delimiter = ',')
        
    #we save also fractions and fx of the ordered output list 
    
    PreSortedIdx = np.argsort(InitClassFraction)[::-1]  #first we sort the vector of classes fractions
    SortedIdx= PreSortedIdx.copy()

    
    SortedInitClassFraction = InitClassFraction[SortedIdx]
        
    SortedInitClassFx = InitClassFx[SortedIdx]
 
    with open('./OrderedClassesGI.txt', "a") as f:
        np.savetxt(f, [SortedInitClassFraction], delimiter = ',')
        
    with open("./Ordered_Outup_Value.txt", "a") as f:
        np.savetxt(f, [SortedInitClassFx.numpy()], delimiter = ',')
   



    #%%% Training Start  
    
    if TrainFlag=='ON':
    
        for NetInstance.params['epoch'] in range (StartEpoch,n_epochs):
            
            
            #open the file at each epoch to overwrite the content
            True_FalseFile = open(DebugFolderPath + "/True_False.txt", "w")
        
            #fixing initial times to calculate the time of each epoch
            start_time = time.time()
            ###################
            # train the model #
            ###################
            NetInstance.model.train() # prep model for training    
    
    
            batches_num =0
       
            
    
    
            for data,label in NetInstance.TrainDL['Class0']: #we start taking, at each step the part of the batch of class 0 (since this is the only class that we necessary have(the label mapping follow growing order starting from 0))
                
                
                batches_num +=1
                
                Mask_Flag = 1 #we update dropout mask at the beginning of each step
                
                data = data.double()
                #load data on device
                data = data.to(device)
                label = label.to(device)                  
                
                NetInstance.StoringGradVariablesReset() # clear the storing variables for gradients
                
                NetInstance.optimizer.zero_grad() # clear the gradients of all optimized variables
                
    
                """
                #note: you can use either NetInstance.optimizer.param_groups[0]['lr'] or NetInstance.optimizer.state_dict()['param_groups'][0]['lr'] to access the real learning rate value
                #but to modify it only NetInstance.optimizer.param_groups is valid
                for g in NetInstance.optimizer.param_groups: #FINE LR STEP-TUNING
                    g['lr'] = g['lr']*(1./(1.+(NetInstance.params['IterationCounter']*lr_decay)))
                #print("lr reale step", NetInstance.optimizer.state_dict()['param_groups'][0]['lr'], NetInstance.optimizer.param_groups[0]['lr']) 
                """
                
                NetInstance.params['IterationCounter'] +=1
    
     
                if NetInstance.params['NetMode']=='VGG_Custom_Dropout':
                    
                    NetInstance.DropoutBatchForward(data, Mask_Flag)
                    Mask_Flag = 0 #the dropout stay fixed for the rest of the batch 
                else:
                    NetInstance.BatchForward(data)
     
                NetInstance.BatchEvalLossComputation(label) #computation of the loss function and the gradient (with backward call)
         
                NetInstance.loss.sum().backward()   # backward pass: compute gradient of the loss with respect to model parameters
    
    
    
    
    
    
                #SAVING THE WEIGHT AT tw (FOR THE CORRELATION COMPUTATION) this block we don't need to put it in the evaluation block because is about the weights, not the gradient (weights are not modified during evaluation procedures)
                if(SphericalConstrainMode=='ON'):
                    if(NetInstance.params['TwComponentCounter']<Ntw):
                        if(NetInstance.params['IterationCounter']==tw[NetInstance.params['TwComponentCounter']]):      
                            NetInstance.WeightsForCorrelations()
                            NetInstance.params['TwComponentCounter']+=1
                    
                    NetInstance.CorrelationsComputation(NetInstance.params['IterationCounter'], N, CorrTimes, tw, t)     
                        
                        
                #we end GD dynamic calculating the per-class norm, normalizing PC gradient, sum them and update the weight following the direction of the obtained vector
                #also the norm gradient on the whole dataset is calculated (summing gradient' vectors of all classes and calculating the corresponding norm)   
                NetInstance.train_loss=0
      
    
                NetInstance.StepSize() #compute the step size associated to the batch
    
    
                NetInstance.optimizer.step() # perform a single optimization step (parameter update)  
    
                #NetInstance.PCN_lr_scheduler()
                
                    
                    
              
                    
                #%%% data capture block 
                #Computations on individual steps (at logarithmically equispaced times): this make sense only for the SGD because in the other case (GD) we update at the end of the epoch   
    
                if(NetInstance.params['TimesComponentCounter']<NetInstance.params['NSteps']):    
                    if ((NetInstance.params['IterationCounter']) == Times[NetInstance.params['TimesComponentCounter']]): 
                        
                        NetInstance.StoringGradVariablesReset() # clear the storing variables for gradients
                        NetInstance.optimizer.zero_grad() # clear the gradients of all optimized variables
                        NetInstance.EvaluationVariablesReset() #reset the total norm, repres. vectors, correct guess vectors  
                        #WEIGHT NORM 
                        NetInstance.WeightNormComputation()
        
                        NetInstance.model.eval()  # prep model for evaluation
                        
    
                        #HERE WE EVALUATE THE DETERMINISTIC VECTOR
                        #we evaluate the training and test set at the times corresponding to the logarithmic equispaced steps
                        #we start from the train set; in the eval mode we are not inerested in the weights' updates
                        for EvalKey in NetInstance.TrainDL:
                            SetFlag = 'Train' 
                            for dataval,labelval in NetInstance.TrainDL[EvalKey]:
                        
                                Mask_Flag = 1
                                
                                dataval = dataval.double() 
                                dataval = dataval.to(device)
                                labelval = labelval.to(device) 
            
                                if NetInstance.params['NetMode']=='VGG_Custom_Dropout':
                                    
                                    NetInstance.DropoutBatchForward(dataval, Mask_Flag)
                                    Mask_Flag = 0
                                else:
                                    NetInstance.BatchForward(dataval)
                                    
                                NetInstance.BatchEvalLossComputation(labelval, NetInstance.params['TimesComponentCounter']+1, SetFlag) #computation of the loss function and the gradient (with backward call)
    
                                #computation of quantity useful for precision accuracy,... measures
                                NetInstance.CorrectBatchGuesses(labelval, NetInstance.params['TimesComponentCounter']+1, SetFlag)
                                #Store the last layer mean representation and per classes loss function
    
                                #NOTE: TO CALCULATE LAST HIDDEN LAYER REPR. WE DIDN'T CALL AGAIN THE COMMAND .model(...). THIS WOULD HAVE PROPAGATED FOR A SECOND TIME THE SAME INPUT ACROSS THE NETWORK; DOING SO WE WASTE COMPUTATIONAL TIME AND WE INCREMENT FOR A SECOND TIME THE SAME GRADIENT VECTOR 
                                #TO AVOID SO WE RECALL DIRECTLY THE DICT CREATED FROM THE FIRST FORWARDING (OutDict)
                                #the computation of last layer representation has been temporary removed because you cannot use a generic label since now classes are mixed inside the batches
                                #NetInstance.MeanRepresClass[labelval[0]].append(NetInstance.OutDict['l2'].clone().double()) 
                                NetInstance.loss.sum().backward()   # backward pass: compute gradient of the loss with respect to model parameters
    
                                NetInstance.optimizer.zero_grad()
                                                            
                            NetInstance.optimizer.zero_grad()#putting gradient to 0 before filling it with the per class normalized sum
                            
                            #NetInstance.LastLayerRepresCompression()
    
                        
                        
                        """
                        for index in range(0, NetInstance.model.num_classes):    
                            ParCount = 0
                            for p in NetInstance.model.parameters():                                                             
                                NetInstance.GradCopy[index].append(copy.deepcopy(NetInstance.NormGrad1Tot[0][index][ParCount]))
                       
                                NetInstance.Norm[index] += (torch.norm(copy.deepcopy(NetInstance.NormGrad1Tot[0][index][ParCount]).cpu()*NetInstance.RoundSolveConst).detach().numpy())**2    
                                ParCount+=1  
                            NetInstance.Norm[index] = (NetInstance.Norm[index]**0.5)/NetInstance.RoundSolveConst
                            
                        NetInstance.Wandb_Log_Grad_Overlap(Times[NetInstance.params['TimesComponentCounter']-1])                    
                        """
    
     
                        NetInstance.LossAccAppend(NetInstance.params['TimesComponentCounter']+1, SetFlag)                   
    
                        for EvalKey in NetInstance.ValidDL:
                            SetFlag = 'Valid' 
                            for dataval,labelval in NetInstance.ValidDL[EvalKey]:
                        
                                Mask_Flag = 1
                                
                                dataval = dataval.double() 
                                dataval = dataval.to(device)
                                labelval = labelval.to(device) 
            
                                if NetInstance.params['NetMode']=='VGG_Custom_Dropout':
                                    
                                    NetInstance.DropoutBatchForward(dataval, Mask_Flag)
                                    Mask_Flag = 0
                                else:
                                    NetInstance.BatchForward(dataval)
                                    
                                NetInstance.BatchEvalLossComputation(labelval, NetInstance.params['TimesComponentCounter']+1, SetFlag) #computation of the loss function and the gradient (with backward call)
    
                                #computation of quantity useful for precision accuracy,... measures
                                NetInstance.CorrectBatchGuesses(labelval, NetInstance.params['TimesComponentCounter']+1, SetFlag)
    
    
                        NetInstance.LossAccAppend(NetInstance.params['TimesComponentCounter']+1, SetFlag)                   
                               
                        NetInstance.model.eval()  # prep model for evaluation
    
                        if params['ValidMode']=='Test': #if we are in the testing mode (i.e. we are running over a set of optimal hyper parameters, then we collect also measures from the test set)
                            for EvalKey in NetInstance.TestDL:
                                SetFlag = 'Test' 
                                for dataval,labelval in NetInstance.TestDL[EvalKey]:
                            
                                    Mask_Flag = 1
                                    
                                    dataval = dataval.double() 
                                    dataval = dataval.to(device)
                                    labelval = labelval.to(device) 
                
                                    if NetInstance.params['NetMode']=='VGG_Custom_Dropout':
                                        
                                        NetInstance.DropoutBatchForward(dataval, Mask_Flag)
                                        Mask_Flag = 0
                                    else:
                                        NetInstance.BatchForward(dataval)
                                        
                                    NetInstance.BatchEvalLossComputation(labelval, NetInstance.params['TimesComponentCounter']+1, SetFlag) #computation of the loss function and the gradient (with backward call)
    
                                    #computation of quantity useful for precision accuracy,... measures
                                    NetInstance.CorrectBatchGuesses(labelval, NetInstance.params['TimesComponentCounter']+1, SetFlag)
                                    #Store the last layer mean representation and per classes loss function
                                    #Warning: don't call more than once  .model: doing this every time input forward, accumulating gradient; instead, just forward once and save tho output (NOT JUST ['OUT'] BUT THE ENTIRE LIST AND RECALLING WHAT YOU NEED FROM TIME TO TIME)
    
             
                            NetInstance.LossAccAppend(NetInstance.params['TimesComponentCounter']+1, SetFlag)      
                        
                        NetInstance.UpdatePerformanceMeasures(NetInstance.params['TimesComponentCounter']) #ipdating of precision recall F1-measures
    
                        #NetInstance.ReprAngles(NetInstance.params['TimesComponentCounter']) #upd. the norm associated to the repres. vector (last layer state)
                        
                        #TENSORBOARD SUMMARY SAVING
                        NetInstance.SummaryScalarsSaving(Times, NetInstance.params['TimesComponentCounter'])
                        #add also the distribution of weight for each layer (we iterate over each layer saving its weight)
                        NetInstance.SummaryDistrSavings(Times, NetInstance.params['TimesComponentCounter'])     
                        
                        #WandB LOGS SAVINGS
                        NetInstance.WandB_logs(Times, NetInstance.params['TimesComponentCounter'])  #wandb logs                                        
                        
                        NetInstance.params['TimesComponentCounter']+=1
                        
                        NetInstance.UpdateFileData()
    
                        #SAVE A CHECKPOINT OF THE MODEL AT EVERY EVALUATION BLOCK
                        NetInstance.TorchCheckpoint()
                        
                        NetInstance.model.train() # prep model for training    (for the next training step after evaluation)
      
    
            
            print("EPOCH: ", NetInstance.params['epoch'])
         
    
    
            #print("lr reale", NetInstance.optimizer.state_dict()['param_groups'][0]['lr'])        
    
            #lr_schedule.step() #perform the lr_schedule after each epoch
            #print("lr reale", NetInstance.optimizer.state_dict()['param_groups'][0]['lr'])
    
            print("---epoch %d last %s seconds ---" % (NetInstance.params['epoch'] , time.time() - start_time), flush=True, file = ExecutionTimes)        
            if args.SampleIndex==str(1):
                print('epoch %d over %d performed' %(NetInstance.params['epoch'], n_epochs), flush=True)
                
            print("MEASURE TP FP FN", NetInstance.model.TP, NetInstance.model.FP, NetInstance.model.FN, flush=True, file = True_FalseFile)   
            True_FalseFile.close()
        
        NetInstance.UpdateFileData()
        
        #TENSORBOARD HP VALIDATION
        NetInstance.SummaryHP_Validation(learning_rate, batch_size) #at the end of the epochs we save the last test accuracy and test loss for the validation check
  
    print("---total cycle last %s seconds ---" % (time.time() - start_TotTime), flush=True, file = ExecutionTimes)
        



       
#SAVE THE FINAL CHECKPOINT BEFORE CLOSING THE SIMULATION
NetInstance.TorchCheckpoint()




wandb.save(DebugFolderPath+"/InfoGenerali.txt")
wandb.save(FolderPath +'/model.pt') #save the model at the end of simulation to restart it from the end point 
wandb.save(FolderPath+'/*.txt')

run.finish()  #If you're trying to start multiple runs from one script, add two things to your code: (2) run.finish(): Use this at the end of your run to finish logging for that run
        
        
        
