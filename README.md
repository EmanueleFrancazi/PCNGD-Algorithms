#Introduction
##Content (in a nutshell)
The goal of the repo is to provide codes needed to perform simulations on DNNs of some descent-based algorithms.
More specifically you can find, implemented within it, some variants of (S)GD, and, in particular:
* PCNGD
* PCNSGD
* PCNSGD+O
* SGD+O
* PCNSGD+R\
In addition, vanilla versions of (S)GD are present.\
The code is written in python and the reference library, used internally, to implement the networks and work with them is Pytorch.

##Structure

In order to ensure clearer readability the code is divided into 2 scripts: \
* MainBlock.py: code on which the simulation runs; it defines the framework of the program and the flow of executed commands.
* CodeBlocks.py: a secondary library, called from MainBlock.py that implements through methods within it tulle functions that MainBlock.py needs.\
The program is launched by a ,command within the bash script "PythonRunManager.sh". Within the script you can select some parameters/flags that will be given as input to "MainBlock.py". After properly setting the initial parameters (see later sections) the simulations can then be started with the command (from terminal):\
`./PythonRunManager.sh i1 i2` .\
`i1` and `i2` are 2 integer values such that `i1`<`i2`. With this command we begin the serial execution of `i2`-`i2`+1 replicas of the desired simulation. Specifically, each replica is identified by a numeric index between `i1` and `i2`.
The data for each replica, associated with a given set of parameters, are loaded, during the course of the simulation, into a folder that has the index of the replica in its name.

###Inteaction between MainBlock.py and CodeBlocks.py

The methods in "CodeBlocks.py", called by "MainBlock.py" during the simulation, are enclosed in classes.
The simulation starts by setting and defining some essential variables. That done, the DNN is generated as an instance of a class, through the command:\
`NetInstance = CodeBlocks.Bricks(params)`.\
The `Bricks(params)` class has to be interpred as the bricks that will used to made up the main code;
it contains all the blocks of code that performs simple tasks. The general structure structure is as follow:\
inside class Bricks we instantiate one of the Net classes. So, in this case, we will not use class inheritance but only class composition: I don't use the previous class as super class but simply call them creating an istance inside the class itself. Each of the Net classes inherit the class NetVariables (where important measures are stored)
_________________________________________
**Notes for newcomers in python:**
                
Inheritance is used where a class wants to derive the nature of parent class and then modify or extend the functionality of it. 
Inheritance will extend the functionality with extra features allows overriding of methods, but in the case of Composition, we can only use that class we can not modify or extend the functionality of it. It will not provide extra features.\
**Warning:** you cannot define a method that explicitly take as input one of the instance variables (variables defined in the class); it will not modify the variable value. 
Instead if you perform a class composition as done for NetVariables you can give the variable there defined as input and effectively modify them                                 
_________________________________________


#Running Pipeline
##Bash script

As mentioned, the simulation is started through a bash script ("PythonRunManager.sh"). Within that script some parameters are set. Specifically:
* **FolderName** : is the name of the folder that will contain all the results of the execution.
* **Dataset** : parameter that identifies the dataset to be used; at present the code accepts only CIFAR10 as dataset; to include other datasets (e.g. MNIST) some small changes are necessary because of the different data format.
* **Architecture** : parameter that identifies a the network to be used for the simulation: some option already available (see "DEFINE NN ARCHITECTURE" in CodeBlocks.py). including an arbitrary architecture is very simple; just define the corresponding class and a name that identifies it as a parameter, following the example of the networks already present.
* **LR** : learning rate that will be used. It can be a single value or a set of values (which will be given one after the other)
* **BS** : batch size that will be used. Can be a single value or a set of values (which will be given one after another)
* **GF** : This parameter sets the block size, for groupings operated in group norm layers. It can be a single value or a set of values (which will be given one after the other)
* **DP** : Dropout probability. This parameter sets the probability of zeroing entries across dropout layers. It can be a single value or a set of values (which will be given one after the other)\

For each of the above parameters, it is also possible to select more than one value. In this case, `i2`-`i2`+1 runs will be performed sequentially for each combination of the chosen parameters. For each run, the simulation is started, from the bash script, through the command:\
`python3 MainBlock.py $i $FolderName $Dataset $Architecture $LR $BS $GF $DP` \
The `MainBlock.py` script is thus called.

##MainBlock.py
The code starts with an initial block, where some general parameters are defined (number of epochs, any changes on dataset composition, algorithm to be used, seed initialization, ..). To facilitate the connection with CodeBlocks.py we define a `params` dict where we save all the parameters that we want to be able to access also from "CodeBlocks.py". The network instance is then defined, as explained above, and the program is then started. 
###Logging on server
to more easily monitor the runs and their results the code automatically saves logs of relevant metrics on some server which can then be accessed at any time to check the status of the simulation.
Specifically, simulation results will be available in:\
* Tensorboard: no logging is required for such a server. for more information on using tensorboard see [How to use TensorBoard with PyTorch](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html) 
* Wandb: you can access the server by creating a new account or through accounts from other portals (github, google,...). For more details see, for example 
[W&B-Getting Started with PyTorch ](https://docs.wandb.ai/guides/integrations/pytorch) [Intro to Pytorch with W&B ](https://wandb.ai/site/articles/intro-to-pytorch-with-wandb) 

### Per class DataLoader.
In our experiments, it is essential to monitor individual class performance. Not only that; some of the algorithms also require calculation of the gradient associated with each class. To facilitate this division we define a dataloader associated with each class. We associate with the latter a batchsize parameter set by the imbalance ratio (and by the oversampling strategy, if any, operated by the specific algorithm). To clarify this point, let us consider a practical example. \
Let us imagine simulating PCNSGD (one of the available algorithms) with a binary dataset of 10000 images and an imblance ratio of 9:1 (so 9000 images belonging to the majority class and 1000 to the minority class). Let us now assume that the **BS** parameter is equal to 100, for example. In this case I will then proceed by defining two dataloaders (one for each class) and defining in each of them a batch size parameter following the imbalance ratio. I will thus obtain 100 batches for the two dataloaders with 90 and 10 elements in each batch, respectively.
In this way within each batch of a given dataloader we will find only images belonging to the same class. We can at this point, at each step, propagate one batch for each class, save the gradient to an auxiliary variable, reset it, and finally proceed to update the network weights after all classes have propagated one of their batches. 

