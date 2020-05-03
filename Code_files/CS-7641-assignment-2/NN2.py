
import os
import csv
import time
import sys
sys.path.append(os.path.join(sys.path[0],'ABAGAIL','ABAGAIL.jar'))
from func.nn.backprop import BackPropagationNetworkFactory
from shared import SumOfSquaresError, DataSet, Instance
from opt.example import NeuralNetworkOptimizationProblem
from func.nn.backprop import RPROPUpdateRule, BatchBackPropagationTrainer
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
from func.nn.activation import RELU

# Network parameters found "optimal" in Assignment 1
INPUT_LAYER = 16 #attribute-count
HIDDEN_LAYER1 = 42
OUTPUT_LAYER = 26
TRAINING_ITERATIONS = 100001
OUTFILE = './FINAL_NN/XXX_LOG.csv'


def initialize_instances(infile):
    """Read the m_trg.csv CSV data into a list of instances."""
    instances = []

    # Read in the CSV file
    with open(infile, "r") as dat:
        reader = csv.reader(dat)

        for row in reader:
            instance = Instance([float(value) for value in row[:-1]])
            m = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            if row[-1] == 'A':
                # print("A here")
                m = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            elif row[-1] == 'B':
                m = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            elif row[-1] == 'C':
                m = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            elif row[-1] == 'D':
                m = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            elif row[-1] == 'E':
                m = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            elif row[-1] == 'F':
                m = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            elif row[-1] == 'G':
                m = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            elif row[-1] == 'H':
                m = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            elif row[-1] == 'I':
                m = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            elif row[-1] == 'J':
                m = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            elif row[-1] == 'K':
                m = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            elif row[-1] == 'L':
                m = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            elif row[-1] == 'M':
                m = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            elif row[-1] == 'N':
                m = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            elif row[-1] == 'O':
                m = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            elif row[-1] == 'P':
                m = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            elif row[-1] == 'Q':
                m = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            elif row[-1] == 'R':
                m = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
            elif row[-1] == 'S':
                m = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
            elif row[-1] == 'T':
                m = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
            elif row[-1] == 'U':
                m = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
            elif row[-1] == 'V':
                m = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
            elif row[-1] == 'W':
                m = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
            elif row[-1] == 'X':
                m = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
            elif row[-1] == 'Y':
                m = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
            elif row[-1] == 'Z':
                m = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
            instance.setLabel(Instance(m))
            instances.append(instance)

    return instances
	

def errorOnDataSet(network,ds,measure):
    N = len(ds)
    error = 0.
    correct = 0
    incorrect = 0
    for instance in ds:
        network.setInputValues(instance.getData())
        network.run()
        error1 = 0
        actual = instance.getLabel()
        predicted = network.getOutputValues()
        #print("ACTUAL", actual)
        #print("PREDICTED", predicted)
        leng = actual.size()
        actual_list = []
        predicted_list = []
        for i in range(leng):
            diff = abs(predicted.get(i) - actual.getContinuous(i))
            actual_list.append(actual.getContinuous(i))
            predicted_list.append(predicted.get(i))
            error1 += diff * diff
        error += error1
        max_actual = max(actual_list)
        max_predict = max(predicted_list)
        actual_one = actual_list.index(max(actual_list))
        predicted_one = predicted_list.index(max(predicted_list))
        if actual_one == predicted_one:
            correct += 1
        else:
            incorrect += 1
    MSE = error / float(N)
    acc = (correct / float(correct + incorrect))*100
    return MSE, acc
	
	
def train(oa, network, oaName, training_ints, validation_ints, testing_ints, measure):
    """Train a given network on a set of instances.
    """
    print "\nError results for %s\n---------------------------" % (oaName,)
    times = [0]
    for iteration in xrange(TRAINING_ITERATIONS):
        start = time.clock()
        oa.train()
        elapsed = time.clock()-start
    	times.append(times[-1]+elapsed)
        if iteration % 10==0:
    	    MSE_trg, acc_trg = errorOnDataSet(network,training_ints,measure)
            MSE_val, acc_val = errorOnDataSet(network,validation_ints,measure)
            MSE_tst, acc_tst = errorOnDataSet(network,testing_ints,measure)
            txt = '{},{},{},{},{},{},{},{}\n'.format(iteration, MSE_trg, MSE_val, MSE_tst, acc_trg, acc_val, acc_tst,
                                                     times[-1]);
            print txt
            with open(OUTFILE.replace('XXX',oaName),'a+') as f:
                f.write(txt)

def main(CE):
    """Run this experiment"""
    training_ints = initialize_instances('m_trg.csv')
    testing_ints = initialize_instances('m_test.csv')
    validation_ints = initialize_instances('m_val.csv')
    factory = BackPropagationNetworkFactory()
    measure = SumOfSquaresError()
    data_set = DataSet(training_ints)
    relu = RELU()
    rule = RPROPUpdateRule()
    oa_name = "SA{}".format(CE)
    with open(OUTFILE.replace('XXX',oa_name),'w') as f:
        f.write('{},{},{},{},{},{},{},{}\n'.format('iteration', 'MSE_trg', 'MSE_val', 'MSE_tst', 'acc_trg', 'acc_val',
                                                   'acc_tst', 'elapsed'))
    classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER1,OUTPUT_LAYER],relu)
    nnop = NeuralNetworkOptimizationProblem(data_set, classification_network, measure)
    oa = SimulatedAnnealing(1E10, CE, nnop)
    train(oa, classification_network, oa_name, training_ints,validation_ints,testing_ints, measure)


if __name__ == "__main__":
    #[0.15,0.35,0.55,0.70,0.95]
    for CE in [0.15,0.3,0.45,0.6,0.75,0.9]:
        main(CE)

