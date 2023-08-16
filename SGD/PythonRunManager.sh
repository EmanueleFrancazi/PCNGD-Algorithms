#!/usr/bin/env bash

#WARNING:  the program takes as input only one parameter (number of replications). More specifically, the minimum and maximum index of the samples

#INPUT PARAMETERS: the following are the set of parameters used to fix the model for the simulation

FolderName='SimulationResult'


#Dataset='CIFAR100'
Dataset='CIFAR10'
#Dataset='MNIST'
#Dataset='Imagenet-LT'
#Dataset='Places365'
#Dataset='CIFAR100'
#Dataset='INATURALIST'

#DataFolder="data_nobackup/INaturalist"
#DataFolder="data_nobackup/CIFAR10"
#DataFolder="$TMPDIR/data_nobackup/INaturalist"
#DataFolder="$TMPDIR/data_nobackup/CIFAR10"
#DataFolder="$SCRATCH/snx3000/efrancaz/data_nobackup/CIFAR100"
DataFolder="/home/EAWAG/francaem/restored/data/CIFAR10"
#DataFolder="/home/EAWAG/francaem/restored/data/CIFAR100/"


#Architecture='VGG_Custom_Dropout'
#Architecture='VGG16'
#Architecture='CNN'
#Architecture='MultiPerceptron'
#Architecture='ResNet18'
Architecture='ResNet34'


mkdir $FolderName 


for ((i=$1; i<=$2; i++))
do
	#perform a nested loop with all the hyperparamater we want to perform, in this way:
	#-if for a certain combination the simulation fails only that one is deprecated

	for GF in -1 #4 #4 16 # 64 1 #16 #4 1
	do
		for DP in -1 #0.2 0.4 # 0.2 0.4 #0.4 #0.2 0.4
		do
			for BS in 25 #500 250 125  
			do
				for LR in 0.1 #0.00000001 0.0000001 0.000001 0.00001  #0.001       #1 0.1 0.01   
				do
				        #if [$i -lt $1]; then


	 		

    					python3 SGD.py $i $FolderName $Dataset $Architecture $DataFolder  $LR $BS $GF $DP
    	
    					#with the & option (as below) scripts run in parallel
					#python3 TestNN.py $i $FolderName $Dataset  &


		
	#else
	#	python3 TestNN.py $i $FolderName
	#fi 
	#fi conclude l'if statement
				done
			done
		done
	done	
done







