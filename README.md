# Content (in a nutshell)
**NOTE**: **Anonymous Github does not format text on markdown following GFM (GitHub Flavored Markdown)**. Therefore, there are minor differences from the formatting in the original repo (mainly the mathematical expressions are not shown properly but through the source code expression, not formatted). **The file can be displayed properly by copying the source code on github.
The final version on github will then not show the same problem.**

The repo contains codes to reproduce the results of the article [title and link of the paper available after publication]. \
The necessary modules to be installed with their versions can be found in the `requirements.txt` file.

## Notation

We introduce here the notation that will be employed in the following sections:

* $|\cdot|$ : Cardinality of a set, i. e. the number of elements that make up the set
* $\lVert \cdot \rVert$ : L2 Norm
* $N_e$ : total number of simulation epochs
* $N_c$ : number of classes
* $\mathcal{D} = (\boldsymbol{xi}_i, y_i)_{i=1}^n$ : dataset
* $\boldsymbol{\xi}_i \in \mathbb{R}^d$ : input vector
* $y_i \in  [0, \dots, N_c - 1]$ : label ; by convention, label " $0$ " identifies the majority class of the dataset
* $C_l = \\{ i \mid y_i = l \\}$  : subgroup of indices belonging to class $l$
* $\mathcal{D}_l = (\boldsymbol{\xi}_i, y_i)\_{i \in C_l}$ : Subgroup of $\mathcal{D}$ elements belonging to class $l$
* $n$ : dataset size, i.e. the number of elements that makes up the dataset
* $n_l$ : Number of examples in the dataset belonging to class $l$
* $\tilde n_l$ : Number of examples in the batch belonging to class $l$
* $\gamma_t$: batch selected at step $t$
* $\tilde n = |\gamma_t|$ : batch size at step $t$; note that for full-batch algorithms (e.g. GD) $|\gamma_t|=n$ 
* $\{ \gamma_t \}_e$: set of batches defined for the epoch $e$
* $\eta_t$ : learning rate
* $\boldsymbol{x}_t$ : set of network parameters at time $t$
* $f(\boldsymbol{x}_t) = \frac{1}{|\mathcal{D}|} \sum\_{i \in \mathcal{D}} f_i(\boldsymbol{x}_t)$ : Average loss function calculated over all elements in the dataset.
* $f^{(l)}(\boldsymbol{x}_t) = \frac{1}{|\mathcal{D}|} \sum\_{ i \in C_l} f_i(\boldsymbol{x}_t)$ : contribution to $f(\boldsymbol{x}_t)$ from class $l$
* $\nabla\_{\tilde n} f^{(l)}(\boldsymbol{x}_t)$ : is the gradient over a random batch of size $\tilde n$, for examples belonging to the class $l$
* $\nabla f\_{\tilde n_l}^{(l)} (\boldsymbol{x}\_t) = \frac{1}{\tilde n_l} \sum\_{\substack{i \in C_l \\ i \in \gamma_t }} \nabla f_i (\boldsymbol{x}\_t)$ : Average gradient computed on the elements in the batch belonging to class $l$. Note that $\nabla f_{\tilde n_l}^{(l)}(\boldsymbol{x}_t)$ and $\nabla\_{\tilde n} f^{(l)}(\boldsymbol{x}_t)$ differ by only one factor in particular $\nabla f\_{\tilde n_l}^{(l)}(\boldsymbol{x}_t) = \frac{\tilde n}{\tilde n_l} \nabla\_{\tilde n} f^{(l)}(\boldsymbol{x}_t)$
* $\%$ : Modulo operator
* $\boldsymbol{v} \cdot \boldsymbol{w}$ : dot product 





## Description of the main Algorithms
The goal of the repo is to provide codes needed to perform simulations on DNNs of some descent-based algorithms.
You can find, implemented within it, some variants of (S)GD; for each of them below is pseudo-code to briefly illustrate how the algorithm works.
**Some of the steps might seem cumbersome at first glance**, e.g. dividing the dataset by classes and then composing the batches. **However, this method was chosen to efficiently** ( thus avoiding micro-batching approaches, i.e., propagating samples one at a time) **isolate the gradient associated with individual classes**. This enables us to normalize the contributions of each class and combine them together according to the update rule.
Recently, libraries ( [Opacus](https://openreview.net/pdf?id=EopKEYBoI-) and [functorch](https://pytorch.org/functorch/stable/notebooks/per_sample_grads.html) ) have been introduced that would allow a more streamlined implementation of the code while still avoiding micro-batching approaches.

### PCNGD


---
 **Algorithm: PCNGD** 
---
1. Initialize $\boldsymbol{x}_0$
2.  Split $\mathcal{D}$ into subgroups $\\{ \mathcal{D}_l \\}$
3. For epoch $e \in [1, \dots, N_e]$
   1. Calculate  $\nabla f^{(l)}(\boldsymbol{x}_t)$ and  $\lVert \nabla f^{(l)}(\boldsymbol{x}_t) \rVert$ 
   2. $\boldsymbol{x}\_{t+1} = \boldsymbol{x}_t -\eta_t \left( \sum_l \frac{\nabla f^{(l)}(\boldsymbol{x}_t)}{\lVert \nabla f^{(l)}(\boldsymbol{x}_t) \rVert} \right)$ 
---

After initializing the network weights, $\boldsymbol{x}_0$ , (line 1 in **Algorithm: PCNGD**) we split the examples of the dataset according to their
class (line 2 in **Algorithm: PCNGD**). For each epoch we calculate the gradient associated with the individual classes and the
corresponding norm (line 3.i in **Algorithm: PCNGD**). Finally, we use the calculated quantities to perform the update rule (line 3.ii in
**Algorithm: PCNGD**).      
        
        
### PCNSGD 

---
 **Algorithm: PCNSGD** 
---
1. Initialize $\boldsymbol{x}_0$
2. Split $\mathcal{D}$ into subgroups $\\{ \mathcal{D}_l \\}$ 
3. For epoch $e \in [1, \dots, N_e]$
   1. Shuffle $\\{ \mathcal{D}_l \\}$
   2. Group $\\{ \mathcal{D}_l \\}$ into $\\{ \gamma_t^{(l)}$ $\\}_e$.  
   3. For $i \in [1, \dots, N_b]$ (Iterate over the batch index)
      1. For $l \in [0, \dots, N_c - 1]$ 
         1. Select   $\gamma_t^{(l)}$
         2. Calculate  $\nabla\_{\tilde n} f^{(l)}(\boldsymbol{x}_t)$ and $\lVert \nabla\_{\tilde n} f^{(l)}(\boldsymbol{x}_t) \rVert$ 
      2. $\boldsymbol{x}\_{t+1} = \boldsymbol{x}_t -\eta_t \left( \sum_l \frac{\nabla\_{\tilde n} f^{(l)}(\boldsymbol{x}_t)}{\lVert \nabla\_{\tilde n} f^{(l)}(\boldsymbol{x}_t) \rVert} \right)$ 
---

After initializing the network weights, $\boldsymbol{x}_0$ , (line 1 in **Algorithm: PCNSGD**) we split the examples of the dataset according to their class (line 2 in **Algorithm: PCNSGD**). At the beginning of each epoch we shuffle the elements of each subgroup (line 3.ii in **Algorithm: PCNSGD**) and group them into per-class batches (line 3.ii in **Algorithm: PCNSGD**). Note that the per-class batch sizes are set by the imbalance ratio; consequently the number of per-class batches is the same $\forall l$, i.e. $|\{ \gamma_t^{(l)} \}|=N_b^{(l)}= N_b$. \newline
At each step we then select a per-class batch (line 3.iii.a.a in **Algorithm: PCNSGD**) and calculate the gradient associated with it and its norm (line 3.iii.a.b in  **Algorithm: PCNSGD**). Finally, we use the calculated quantities to apply the update rule on the network weights (line 3.iii.b in  **Algorithm: PCNSGD**).



---
 **Algorithm: PCNSGD+O** 
---
1. Initialize $\boldsymbol{x}_0$
2. Split $\mathcal{D}$ into subgroups $\\{ \mathcal{D}_l \\}$
3. For epoch $e \in [1, \dots, N_e]$
   1. Shuffle $\\{ \mathcal{D}_l \\}$
   2. Group $\\{ \mathcal{D}_l \\}$ into $\\{ \gamma_t^{(l)}$ $\\}_e$  
   3. For $i \in [1, \dots, N_b^{(0)}]$ (Iterate over the majority class batch index)
      1. For $l \in [0, \dots, N_c - 1]$
         1. If $i \\% N_b^{(l)} = 0$ (This indicates that we have iterated over all batches of the class $l$)
            1. Regroup $\mathcal{D}_l$ into per-class batches
         2. Select $\gamma_t^{(l)}$
         3. Calculate $\nabla\_{\tilde n} f^{(l)}(\boldsymbol{x}_t)$ and $\lVert \nabla\_{\tilde n} f^{(l)}(\boldsymbol{x}_t) \rVert$ 
      2. $\boldsymbol{x}\_{t+1} = \boldsymbol{x}_t -\eta_t \left( \sum_l \frac{\nabla\_{\tilde n} f^{(l)}(\boldsymbol{x}_t)}{\lVert \nabla\_{\tilde n} f^{(l)}(\boldsymbol{x}_t) \rVert} \right)$ 
---

After initializing the network weights, $\boldsymbol{x}_0$ , (line 1 in  **Algorithm: PCNSGD+O**) we split the examples of the dataset according to their class (line 2 in **Algorithm: PCNSGD+O**). At the beginning of each epoch we shuffle the elements of each subgroup (line 3.i in **Algorithm: PCNSGD+O**) and group them into per-class batches using the same per-class batch size, $|\gamma_t^{(l)}|= |\gamma_t|$  $\forall l$ (line 3.ii in **Algorithm: PCNSGD+O**). Note that since different classes have a different number of elements, $| \mathcal{D}_l|$, we will get a different number of batches for each of them: $|\{ \gamma_t^{(l)} \}| = N_b^{(l)}$, with $N_b^{(0)} = \max_l N_b^{(l)}$ (" $0$ " is the label of the majority class). \newline 
At each step we iterate along the classes.
For each of them we check that per-classes batches are still available to use as input; if the per-class batches associated with a class $l$ are finished we shuffle the elements into the subgroup $\mathcal{D}_l$ and define a new set of per-class-batches $\{ \gamma_l \}$ (line 3.iii.a.a.a in **Algorithm: PCNSGD+O**) as done (for each class) at the beginning of the epoch. After this check we then select a per-class batch $\gamma_l$ (line 3.iii.a.b in **Algorithm: PCNSGD+O**).
We calculate the gradient associated with it and its norm (line 3.iii.a.c in **Algorithm: PCNSGD+O**). Finally, we use the calculated quantities to apply the update rule on the network weights (line 3.iii.b in **Algorithm: PCNSGD+O**).



---
 **Algorithm: SGD+O**
---
1. Initialize $\boldsymbol{x}_0$
2. Split $\mathcal{D}$ into subgroups $\\{ \mathcal{D}_l \\}$
3. For epoch $e \in [1, \dots, N_e]$
   1. Shuffle $\\{ \mathcal{D}_l \\}$
   2. Group $\\{ \mathcal{D}_l \\}$ into $\\{ \gamma_t^{(l)}$ $\\}_e$  
   3. For $i \in [1, \dots, N_b^{(0)}]$ (Iterate over the majority class batch index)
      1. For $l \in [0, \dots, N_c - 1]$
         1. If $i \\% N_b^{(l)} = 0$ (This indicates that we have iterated over all batches of the class $l$)
            1. Regroup $\mathcal{D}_l$ into per-class batches
         2. Select $\gamma_t^{(l)}$
         3. Calculate $\nabla\_{\tilde n} f^{(l)}(\boldsymbol{x}_t)$ 
      2. $\boldsymbol{x}\_{t+1} = \boldsymbol{x}_t -\eta_t \left( \sum_l \nabla\_{\tilde n} f^{(l)}(\boldsymbol{x}_t) \right)$ 
---
After initializing the network weights, $\boldsymbol{x}_0$ , (line 1 in Algorithm **Algorithm: SGD+O**) we split the examples of the dataset according to their class (line 2 in **Algorithm: SGD+O**). At the beginning of each epoch we shuffle the elements of each subgroup (line 3.i in **Algorithm: SGD+O**) and group them into per-class batches using the same per-class batch size, $|\gamma_t^{(l)}|= |\gamma_t|$  $\forall l$ (line 3.ii in **Algorithm: SGD+O**). Note that since different classes have a different number of elements, $| \mathcal{D}_l|$, we will get a different number of batches for each of them: $|\{ \gamma_t^{(l)} \}| = N_b^{(l)}$, with $N_b^{(0)} = \max_l N_b^{(l)}$ (" $0$ " is the label of the majority class). \newline 
At each step we iterate along the classes.
 For each of them we check that per-classes batches are still available to use as input; if the per-class batches associated with a class $l$ are finished we shuffle the elements into the subgroup $\mathcal{D}_l$ and define a new set of per-class-batches $\{ \gamma_l \}$ (line 3.iii.a.a.a in **Algorithm: SGD+O**) as done (for each class) at the beginning of the epoch. After this check we then select a per-class batch $\gamma_l$ (line 3.iii.a.b in **Algorithm: SGD+O**).
We calculate the gradient associated with it (line 3.iii.a.c in **Algorithm: SGD+O**). Finally, we use the calculated quantities to apply the update rule on the network weights (line 3.iii.b in **Algorithm: SGD+O**).

---
 **Algorithm: PCNSGD+R**
---
1. Initialize $\boldsymbol{x}_0$
2. Split $\mathcal{D}$ into subgroups $\\{ \mathcal{D}_l \\}$
3. For epoch $e \in [1, \dots, N_e]$
   1. Shuffle $\\{ \mathcal{D}_l \\}$
   2. Group $\\{ \mathcal{D}_l \\}$ into $\\{ \gamma_t^{(l)}$ $\\}_e$ 
   3. For $i \in [1, \dots, N_b]$ (Iterate over the batch index)
      1. For $l \in [0, \dots, N_c - 1]$ 
         1. Select $\gamma_t^{(l)}$
         2. Calculate $\nabla f^{(l)}(\boldsymbol{x}_t)$ and $\lVert \nabla f^{(l)}(\boldsymbol{x}_t) \rVert$ 
         3. Calculate $\nabla f^{(l)}(\boldsymbol{x}_t)$ and $\lVert \nabla f^{(l)}(\boldsymbol{x}_t) \rVert$ 
         4. Compute $p_l = \left( \frac{\nabla\_{\tilde n} f^{(l)}(\boldsymbol{x}_t)}{\lVert \nabla\_{\tilde n} f^{(l)}(\boldsymbol{x}_t) \rVert} \right) \cdot \left( \frac{\nabla f^{(l)}(\boldsymbol{x}_t)}{\lVert \nabla f^{(l)}(\boldsymbol{x}_t) \rVert} \right)$
      2. $\boldsymbol{x}\_{t+1} = \boldsymbol{x}_t -\eta_t \left( \sum_l \frac{\nabla\_{\tilde n} f^{(l)}(\boldsymbol{x}_t)}{p_l \lVert \nabla\_{\tilde n} f^{(l)}(\boldsymbol{x}_t) \rVert } \right)$ 
---
After initializing the network weights, $\boldsymbol{x}_0$ , (line 1 in **Algorithm: PCNSGD+R**) we split the examples of the dataset according to their class (line 2 in **Algorithm: PCNSGD+R**). At the beginning of each epoch we shuffle the elements of each subgroup (line 3.i in **Algorithm: PCNSGD+R**) and group them into per-class batches (line 3.ii in **Algorithm: PCNSGD+R**). Note that Per-class batch sizes are set by the imbalance ratio; consequently the number of per-class batches $| \{ \gamma_t^{(l)} \}|=N_b^{(l)}= N_b$  is the same $\forall l$. \newline
At each step, for each class, we then select a per-class batch (line 3.iii.a.a in **Algorithm: PCNSGD+R**) and calculate the gradient associated with it and its norm (line 3.iii.a.b in **Algorithm: PCNSGD+R**).
Next, we calculate the per-class gradient associated with the entire dataset and its norm (line 3.iii.a.c in **Algorithm: PCNSGD+R**). We then calculate the projections of the normalized mini-batch gradients along the corresponding full-batch directions (line 3.iii.a.d in **Algorithm: PCNSGD+R**). 
Finally, we use the calculated quantities to apply the update rule on the network weights (line 3.iii.b in **Algorithm: PCNSGD+R**).



In addition, vanilla versions of (S)GD are present.\
The code is written in python and the reference library, used internally, to implement the networks and work with them is Pytorch.




## Structure

In order to ensure clearer readability the code is divided into 2 scripts: 
* `MainBlock.py` : code on which the simulation runs; it defines the framework of the program and the flow of executed commands.
* `CodeBlocks.py` : a secondary library, called from `MainBlock.py` that implements through methods within it all the functions that `MainBlock.py` needs.\
The program is launched by a ,command within the bash script `PythonRunManager.sh`. Within the script you can select some parameters/flags that will be given as input to `MainBlock.py`. After properly setting the initial parameters (see later sections) the simulations can then be started with the command (from terminal):\
`./PythonRunManager.sh i1 i2` .\
`i1` and `i2` are two integer values such that `i1`<`i2`. With this command we begin the serial execution of `i2`-`i2`+1 replicas of the desired simulation. Specifically, each replica is identified by a numeric index between `i1` and `i2`.
The data for each replica, associated with a given set of parameters, are loaded, during the course of the simulation, into a folder that has the index of the replica in its name.

### Interaction between `MainBlock.py` and `CodeBlocks.py`
The methods in `CodeBlocks.py`, called by `MainBlock.py` during the simulation, are enclosed in classes.
The simulation starts by setting and defining some essential variables. That done, the DNN is generated as an instance of a class, through the command:\
`NetInstance = CodeBlocks.Bricks(params)`.\
The `Bricks(params)` class has to be interpreted as the bricks that will be used to make up the main code; it contains all the blocks of code that performs simple tasks. The general structure is as follows:\
Inside class `Bricks` we instantiate one of the Net classes. So, in this case, we will not use class inheritance but only class composition: we don't use the previous class as super class but simply call them creating an instance inside the class itself. Each of the Net classes inherits the class NetVariables (where important measures are stored)
_________________________________________
**Notes for newcomers in python:**
                
Inheritance is used where a class wants to derive the nature of parent class and then modify or extend the functionality of it. 
Inheritance will extend the functionality with extra features allows overriding of methods, but in the case of Composition, we can only use that class we can not modify or extend the functionality of it. It will not provide extra features.\
**Warning:** you cannot define a method that explicitly take as input one of the instance variables (variables defined in the class); it will not modify the variable value. 
Instead if you perform a class composition as done for NetVariables you can give the variable there defined as input and effectively modify them.                                
_________________________________________


# Running Pipeline
## Bash script

As mentioned, the simulation is started through a bash script (`PythonRunManager.sh`). Within that script, some parameters are set. Specifically:
* **FolderName** : is the name of the folder that will contain all the results of the execution.
* **Dataset** : parameter that identifies the dataset to be used; at present the code accepts either CIFAR10 or CIFAR100 as dataset; to include other datasets (e.g. MNIST) some small changes are necessary because of the different data format.
* **Architecture** : parameter that identifies the network to be used for the simulation: some option already available (see "DEFINE NN ARCHITECTURE" in CodeBlocks.py). Including an arbitrary architecture is very simple; just define the corresponding class and a name that identifies it as a parameter, following the example of the networks already present.
* **DataFolder** : Path to the folder that contains the dataset to be used in the simulation
* **LR** : the learning rate that will be used. It can be a single value or a set of values (which will be given one after the other)
* **BS** : the batch size that will be used. Can be a single value or a set of values (which will be given one after another)
* **GF** : This parameter sets the block size, for groupings operated in group norm layers. It can be a single value or a set of values (which will be given one after the other)
* **DP** : Dropout probability. This parameter sets the probability of zeroing entries across dropout layers. It can be a single value or a set of values (which will be given one after the other) 

For each of the above parameters, it is also possible to select more than one value. In this case, `i2`-`i2`+1 runs will be performed sequentially for each combination of the chosen parameters. For each run, the simulation is started, from the bash script, through the command: \
`python3 MainBlock.py $i $FolderName $Dataset $Architecture $DataFolder $LR $BS $GF $DP` \
The `MainBlock.py` script is thus called.

## `MainBlock.py`
The code starts with an initial block, where some general parameters are defined (number of epochs, any changes in dataset composition, the algorithm to be used, seed initialization, ..). To facilitate the connection with CodeBlocks.py we define a `params` dict where we save all the parameters that we want to be able to access also from "CodeBlocks.py". The network instance is then defined, as explained above, and the program is then started. 

### Reproducibility and Initialization: Random seed
Immediately after importing the modules into `MainBlock.py`
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
Specifically, simulation results will be available in: 
* Tensorboard: no logging is required for such a server. for more information on using tensorboard see [How to use TensorBoard with PyTorch](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html) 
* Wandb: you can access the server by creating a new account or through accounts from other portals (github, google,...). For more details see, for example 
[W&B-Getting Started with PyTorch ](https://docs.wandb.ai/guides/integrations/pytorch) [Intro to Pytorch with W&B ](https://wandb.ai/site/articles/intro-to-pytorch-with-wandb) 

### Per class DataLoader
In our experiments, it is essential to monitor individual class performance. Not only that; some of the algorithms also require the calculation of the gradient associated with each class. To facilitate this division we define a data loader associated with each class. We associate with the latter a batch size parameter set by the imbalance ratio (and by the oversampling strategy, if any, operated by the specific algorithm). To clarify this point, let us consider a practical example. \
Let us imagine simulating PCNSGD (one of the available algorithms) with a binary dataset of 10000 images and an imbalance ratio of 9:1 (so 9000 images belonging to the majority class and 1000 to the minority class). Let us now assume that the **BS** parameter is equal to 100, for example. In this case, I will then proceed by defining two data loaders (one for each class) and defining in each of them a batch size parameter following the imbalance ratio. I will thus obtain 100 batches for the two data loaders with 90 and 10 elements in each batch, respectively.
In this way, within each batch of a given data loader we will find only images belonging to the same class. We can at this point, at each step, propagate one batch for each class, save the gradient to an auxiliary variable, reset it, and finally proceed to update the network weights after all classes have propagated one of their batches. \
Note that this approach also automatically ensures that each class is represented within the batch. This on the other hand places a limit on the possible choice of batch sizes; in the previous example, for instance, we cannot choose a batch size smaller than 10 since we would end up with a different number of per-class batches for the two classes. In the case of a problematic choice of batch size, the program displays a message warning us of the problem.
