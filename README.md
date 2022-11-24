# Introduction

## Notation

We introduce here the notation that will be employed in the following sections:
* $|\cdot|$ : Cardinality of a set, i. e. the number of elements that make up the set
* $|\cdot|_2$ : L2 Norm
* $N_e$ : total number of simulation epochs
* $N_c$ : number of classes
* $\mathcal{D} = (\xi_i, y_i)_{i=1}^n$ : dataset
* $\xi_i \in \mathbb{R}^d$ : input vector
* $y_i \in  [0, \dots N_c - 1]$ : label ; by convention, label " $0$ " identifies the majority class of the dataset
* $C_l =$ { $i \mid y_i = l$ } : subgroup of elements belonging to class $l$
* $n$ : dataset size,i.e. the number of elements that makes up the dataset
* $\gamma_t$: batch selected at step $t$
* $|\gamma_t|$ : batch size at step $t$; note that for full-batch algorithms (e.g. GD) $|\gamma_t|=n$ 
* $\\{ \gamma_t$ $\\}_e$: set of batches defined for the epoch $e$
* $\eta_t$ : learning rate
* $\boldsymbol{x}_t$ : set of network parameters at time $t$
* $f(\boldsymbol{x}_t) \equiv \frac{1}{|\gamma_t|} \sum\_{i \in \gamma_t} f_i(\boldsymbol{x}_t)$ : Average loss function calculated over all elements in the batch.\
To emphasize the difference between full-batch and mini-batch cases, we introduce the symbol $f\_{FB}(\boldsymbol{x}_t)$ for the full-batch case, i. e. when $\gamma_t = \mathcal{D}$

* $f^{(l)}(\boldsymbol{x}_t) = \frac{1}{|\gamma_t|} \sum\_{i \in C_l, i \in \gamma_t} f_i(\boldsymbol{x}_t)$ : contribution to $f(\boldsymbol{x}_t)$ from class $l$
* $\\%$ : Modulo operator
* $\boldsymbol{v} \cdot \boldsymbol{w}$ : dot product 


## Content (in a nutshell)
The goal of the repo is to provide codes needed to perform simulations on DNNs of some descent-based algorithms.
More specifically you can find, implemented within it, some variants of (S)GD, and, in particular:
* **PCNGD** \
The algorithm is as follows:

    + Initialize $\boldsymbol{x}_0$
    + for epoch $e \in [1, \dots, N_e]$
        - divide the examples in $\mathcal{D}$ into subgroups $\\{ C_l \\}$ according to their class
        - Calculates the gradient associated with each class $l$ , $\nabla f\_{FB}^{(l)}(\boldsymbol{x}_t)$ , and its norm , $|\nabla f\_{FB}^{(l)}(\boldsymbol{x}_t) |_2$ 
        - $\boldsymbol{x}\_{t+1} = \boldsymbol{x}_t -\eta_t \left( \sum_l \frac{\nabla f\_{FB}^{(l)}(\boldsymbol{x}_t)}{|\nabla f\_{FB}^{(l)}(\boldsymbol{x}_t) |_2} \right)$ 

        
        

* **PCNSGD** \
The algorithm is as follows:

    + Initialize $\boldsymbol{x}_0$
    + for epoch $e \in [1, \dots, N_e]$
        - shuffle $\mathcal{D}$
        - group the dataset into the set of batches $\\{ \gamma_t$ $\\}_e$
        - for batch $\gamma_t \in \\{ \gamma_t$ $\\}_e$
            * divide the examples in $\gamma_t$ according to their class
            * Calculates the gradient associated with each class $l$ , $\nabla f^{(l)}(\boldsymbol{x}_t)$ , and its norm , $|\nabla f^{(l)}(\boldsymbol{x}_t) |_2$ 
            * $\boldsymbol{x}\_{t+1} = \boldsymbol{x}_t -\eta_t \left( \sum_l \frac{\nabla f^{(l)}(\boldsymbol{x}_t)}{|\nabla f^{(l)}(\boldsymbol{x}_t) |_2} \right)$ 

* **PCNSGD+O** \
The algorithm is as follows:

    + Initialize $\boldsymbol{x}_0$
    + divide the examples in the dataset into subgroups $\\{ C_l \\}$ according to their class
    + for epoch $e \in [1, \dots, N_e]$
        - shuffle each subgroups $C_l$
        - $\forall l$ group $C_l$ into the set of batches $\\{ \gamma_t^{(l)} \\}$ , using the same batch size, $|\gamma_t^{(l)}|= |\gamma_t| \forall l$ \
        as different classes have a different number of elements, $|C_l|$, we will get a different number of batches for each of them: $|\\{ \gamma_t^{(l)} \\}| = N_b^{(l)}$, with $N_b^{(0)} = \max_l N_b^{(l)} $
        - for $i \in [1, \dots, N_b^{(0)}]$
            * for $l \in [0, \dots, N_c - 1]$
                * if $i \\% N_b^{(l)} = 0$
                    * shuffle $C_l$
                    * group $C_l$ into the set of batches $\\{ \gamma_t^{(l)} \\}$ , using again $|\gamma_t|$ as batch size 
                * extract a batch without replacement from $\\{ \gamma_t^{(l)} \\}$
                * Calculates the gradient associated to the selected batch , $\nabla f^{(l)}(\boldsymbol{x}_t)$ , and its norm , $|\nabla f^{(l)}(\boldsymbol{x}_t) |_2$ 
            * $\boldsymbol{x}\_{t+1} = \boldsymbol{x}_t -\eta_t \left( \sum_l \frac{\nabla f^{(l)}(\boldsymbol{x}_t)}{|\nabla f^{(l)}(\boldsymbol{x}_t) |_2} \right)$ 

* **SGD+O**\
The algorithm is as follows:

    + Initialize $\boldsymbol{x}_0$
    + divide the examples in the dataset into subgroups $\\{ C_l \\}$ according to their class
    + for epoch $e \in [1, \dots, N_e]$
        - shuffle each subgroups $C_l$
        - $\forall l$ group $C_l$ into the set of batches $\\{ \gamma_t^{(l)} \\}$ , using the same batch size, $|\gamma_t^{(l)}|= |\gamma_t| \forall l$ \
        as different classes have a different number of elements, $|C_l|$, we will get a different number of batches for each of them: $|\\{ \gamma_t^{(l)} \\}| = N_b^{(l)}$, with $N_b^{(0)} = \max_l N_b^{(l)} $
        - for $i \in [1, \dots, N_b^{(0)}]$
            * for $l \in [0, \dots, N_c - 1]$
                * if $i \\% N_b^{(l)} = 0$
                    * shuffle $C_l$
                    * group $C_l$ into the set of batches $\\{ \gamma_t^{(l)} \\}$ , using again $|\gamma_t|$ as batch size 
                * extract a batch without replacement from $\\{ \gamma_t^{(l)} \\}$
                * Calculates the gradient associated to the selected batch , $\nabla f^{(l)}(\boldsymbol{x}_t)$ 
            * $\boldsymbol{x}\_{t+1} = \boldsymbol{x}_t -\eta_t \left( \sum_l \nabla f^{(l)}(\boldsymbol{x}_t) \right)$ 

* **PCNSGD+R**\
The algorithm is as follows:

    + Initialize $\boldsymbol{x}_0$
    + for epoch $e \in [1, \dots, N_e]$
        - shuffle $\mathcal{D}$
        - group the dataset into the set of batches $\\{ \gamma_t$ $\\}_e$
        - for batch $\gamma_t \in \\{ \gamma_t$ $\\}_e$
            * divide the examples in $\gamma_t$ according to their class
            * Calculates the gradient associated with each class $l$ , $\nabla f^{(l)}(\boldsymbol{x}_t)$ , and its norm , $|\nabla f^{(l)}(\boldsymbol{x}_t) |_2$ 
            * calculates the gradient per class associated with the entire dataset, $\nabla f\_{FB}^{(l)}(\boldsymbol{x}_t)$, and the corresponding norm $|\nabla f\_{FB}^{(l)}(\boldsymbol{x}_t) |_2$ 
            * for each class $l$ compute $p_l = \left( \frac{\nabla f^{(l)}(\boldsymbol{x}_t)}{|\nabla f^{(l)}(\boldsymbol{x}_t) |_2} \right) \cdot \left( \frac{\nabla f\_{FB}^{(l)}(\boldsymbol{x}_t)}{|\nabla f\_{FB}^{(l)}(\boldsymbol{x}_t) |_2} \right)$
            
            * $\boldsymbol{x}\_{t+1} = \boldsymbol{x}_t -\eta_t \left( \sum_l \frac{\nabla f^{(l)}(\boldsymbol{x}_t)}{|\nabla f^{(l)}(\boldsymbol{x}_t) |_2 p_l} \right)$ 


In addition, vanilla versions of (S)GD are present.\
The code is written in python and the reference library, used internally, to implement the networks and work with them is Pytorch.

the above pseudo-codes are intended to give a general idea of the operation of the various algorithms discussed. For the sake of clarity and compactness, some details in the implementation are not covered in this summary description; for example, for the **PCNSGD**, **PCNSGD+R** and **PCNSGD+O** algorithms we want to ensure that all classes are represented by at least one element in the batch. This can be ensured in several ways: for more details see **Per class DataLoader**.



## Structure

In order to ensure clearer readability the code is divided into 2 scripts: \
* MainBlock.py: code on which the simulation runs; it defines the framework of the program and the flow of executed commands.
* CodeBlocks.py: a secondary library, called from MainBlock.py that implements through methods within it tulle functions that MainBlock.py needs.\
The program is launched by a ,command within the bash script "PythonRunManager.sh". Within the script you can select some parameters/flags that will be given as input to "MainBlock.py". After properly setting the initial parameters (see later sections) the simulations can then be started with the command (from terminal):\
`./PythonRunManager.sh i1 i2` .\
`i1` and `i2` are 2 integer values such that `i1`<`i2`. With this command we begin the serial execution of `i2`-`i2`+1 replicas of the desired simulation. Specifically, each replica is identified by a numeric index between `i1` and `i2`.
The data for each replica, associated with a given set of parameters, are loaded, during the course of the simulation, into a folder that has the index of the replica in its name.

### Inteaction between MainBlock.py and CodeBlocks.py
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


# Running Pipeline
## Bash script

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

## MainBlock.py
The code starts with an initial block, where some general parameters are defined (number of epochs, any changes on dataset composition, algorithm to be used, seed initialization, ..). To facilitate the connection with CodeBlocks.py we define a `params` dict where we save all the parameters that we want to be able to access also from "CodeBlocks.py". The network instance is then defined, as explained above, and the program is then started. 

### Reproducibility and Initialization: Random seed
Immediately after importing the modules into MainBlock.py 
we proceed to initialize the random seeds. Note that initialization must be performed on all libraries that use pseudo-random number generators (in our case numpy, random, torch). 
The operation of fixing the seed for a given simulation is a delicate operation since a wrong choice could create an undesirable correlation between random variables generated in independent simulations. 
The following two lines fix the seed: 

```
    t = int( time.time() * 1000.0 )
    seed = ((t & 0xff0000) >> 24) + ((t & 0x00ff0000) >> 8) + ((t & 0x0000ff00) << 8) + ((t & 0x0000ff) << 24)   
```
Python time method `time()` returns the time as a floating point number expressed in seconds since the epoch, in UTC. This value is then amplified. Finally, the bit order is reversed so as to reduce the dependence on the least significant bits, further increasing the distance between similar values (more details are given directly in the code, as a comment, immediately after initialization).
The resulting value is then used as a seed for initialization.
The seed is then saved within a file and printed out, so that the simulation can be easily reproduced if required.



### Logging on server
to more easily monitor the runs and their results the code automatically saves logs of relevant metrics on some server which can then be accessed at any time to check the status of the simulation.
Specifically, simulation results will be available in:\
* Tensorboard: no logging is required for such a server. for more information on using tensorboard see [How to use TensorBoard with PyTorch](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html) 
* Wandb: you can access the server by creating a new account or through accounts from other portals (github, google,...). For more details see, for example 
[W&B-Getting Started with PyTorch ](https://docs.wandb.ai/guides/integrations/pytorch) [Intro to Pytorch with W&B ](https://wandb.ai/site/articles/intro-to-pytorch-with-wandb) 

### Per class DataLoader
In our experiments, it is essential to monitor individual class performance. Not only that; some of the algorithms also require calculation of the gradient associated with each class. To facilitate this division we define a dataloader associated with each class. We associate with the latter a batchsize parameter set by the imbalance ratio (and by the oversampling strategy, if any, operated by the specific algorithm). To clarify this point, let us consider a practical example. \
Let us imagine simulating PCNSGD (one of the available algorithms) with a binary dataset of 10000 images and an imblance ratio of 9:1 (so 9000 images belonging to the majority class and 1000 to the minority class). Let us now assume that the **BS** parameter is equal to 100, for example. In this case I will then proceed by defining two dataloaders (one for each class) and defining in each of them a batch size parameter following the imbalance ratio. I will thus obtain 100 batches for the two dataloaders with 90 and 10 elements in each batch, respectively.
In this way within each batch of a given dataloader we will find only images belonging to the same class. We can at this point, at each step, propagate one batch for each class, save the gradient to an auxiliary variable, reset it, and finally proceed to update the network weights after all classes have propagated one of their batches. 
Note that this approach also automatically ensures that each class is represented within the batch.
