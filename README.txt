Project Description:

This project uses the randomized optimization algorithms of Random Hill Climbing, Simulated Annealing, Genetic Algorithm and MIMIC on the optimization problems of Continuous Peaks problems, Flip-flop problem and Travelling Salesman problem and to determine weights for a neural network built using backpropagation.

This file contains the detailed instructions on the contents of the code and how to run the code for this project.

Environment:

The code was developed and exceuted in PyCharm using Jython 2.7 as interpreter and Windows 10 machine. The ABAGAIL.jar is built from the source java files and the python files are written in Jython 2.7.0.

File structure:
Inside Code_files folder, Project_code folder is stored.
Project_code  is the parent folder with the following sub-folders:
	a.ABAGAIL folder having the jar file, ant build file and the java source files
	b.Python files
	c.Log files
The contents of Python files and Log files are described later in this document. 

Code files under Project_code folder:
	
The following python files find the optimal weights of neural network built in Assignment-1 for Letter Recognition dataset.
1.NN0.py uses backpropagation technique
2.NN1.py uses Random Hill Climbing
3.NN2.py uses Simulated Annealing
4.NN3.py uses Genetic Algorithm
The following python files uses the randomized optimization algorithms of Random Hill Climbing, Simulated Annealing, Genetic Algorithm and MIMIC on the optimization problems.
5.continuouspeaks.py uses all 4 randomized optimization algorithms to solve Continuous Peaks problem.
6.flipflop.py uses all 4 randomized optimization algorithms to solve the FlipFlop problem.
7.tsp.py uses all 4 randomized optimization algorithms to solve the Traveling Salesman problem.

Log files under CS-7641-assignment-2 folder:

1.Final_NN folder has the log files of NN0.py, NN1.py, NN2.py and NN3.py
2.FinalCont folder has the log files of continuouspeaks.py
3.FinalFF folder has the log files of flipflop.py
4.FinalTSP folder has the log files of tsp.py
Within these folders, the log files containing the plotting data are csv files and the file names correspond to:
<Algorithm_used><N>_<T if applicable>_<parameters set>_LOG_<Trail_number>.csv
N and T(if applicable) are defined for each optimization problem inside the corresponding .py file.
Some files have <Iteration_count> added at the end of the filename and this was for experimenting individual algorithmic convergence.

Datasets under Project_code folder:

The parsed and preprocessed dataset is stored in the following csv files under Project_code folder:
1.m_train.csv has the training data
2.m_val.csv has the validation data
3.m_test.csv has the testing data.
The steps of preprocessing are detailed later in this README.

How to run the code:

1. The python files under Project_code can be run using Jython 2.7.0 interpreter.
2. Running each of the 7 above mentioned .py files populate the log files in corresponding folders.
 
Notes:

ABAGAIL folder has the source files written using JAVA 1.8.0_121 and the ABAGAIL.jar. The path to ABAGAIL.jar is already included in the .py code files. 

MS Excel was used to plot the charts. The data in log files were copied to MS Excel files and pivot tables were used to average the values of different runs of each parameter combination for each algorithm.

Preprocessing the data using Weka 3.8 GUI:
The preprocessing of data was done in Weka. The steps are:
1. Download dataset from UCI repository at https://archive.ics.uci.edu/ml/datasets/letter+recognition and in Weka GUI, load dataset in Preprocess -> Open File.
2. In Preprocess -> Edit, the correct attribute was set as the class uisng "Attribute as Class" option. 
3. Use Filter -> Unsupervised -> Instance -> RemoveDuplicates filter to remove the duplicated rows from the dataset.
4. Use Filter -> Unsupervised -> Instance -> Randomize filter to randomize the data using default seed value. 
5. Using Filter -> Unsupervised -> Instance -> Resample, with no replacement, 30% of dataset is sampled and stored as testing dataset. This filter produces a random subsample of a dataset using sampling without replacement. (Used 324 as seed for Letter Recognition dataset)
6. Setting the Invertselection parameter of Resample filter to true gives the remaining 70% of the dataset, which is stored as training dataset.
7. From the training dataset, 10% of data is set as validation set and the rest 90% is training dataset.
8.This 90%-10% split is done using weka-> filters -> supervised -> instance -> StratifiedRemoveFolds with fold value set as 1, numFolds as 10 and seed as 0.

The preproceesed train, validation and test sets are included inside Project_code and are also available at:
1. Letter Recognition train dataset 
https://drive.google.com/open?id=1DTpmHsguCmWPmmez9fBFFdD8PFkbYYpv 
2. Letter Recognition validation dataset
https://drive.google.com/open?id=12XAmX4E5o3CASka9CWTgADI8mr2-OcLP
3. Letter Recognition test dataset https://drive.google.com/open?id=1oi1drTqzJNdacoreVv_pkvPpcBpLFb8N 


