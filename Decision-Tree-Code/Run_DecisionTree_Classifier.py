import DecisionTreeID3
import sys
import re
import pandas as pd
import numpy as np
import math
#from collections import deque
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.metrics import accuracy_score
from argparse import ArgumentParser


if __name__ == '__main__':

    parser = ArgumentParser()
	
    parser.add_argument("train_data", help="Give data to be trained as input")
						
    parser.add_argument("valid_data", help="Give data to be validated as input")
						
    parser.add_argument("test_data", help="Give data to be tested as input")
	
    parser.add_argument("-c", "--criterion", choices = ('entropy','variance'), default = 'entropy', dest="Impurity_Heuristic",help="Mention Impurity Heuristic. Default is Entropy")

    parser.add_argument("-p", "--prune", choices = ('depth_based','reduced_error'),dest="Pruning_Type",help="Mention Pruning Method")

    args = parser.parse_args()
	
    #print(args)
    
    path1 = str(args.train_data)
    path2 = str(args.valid_data)
    path3 = str(args.test_data)
    data_train = pd.read_csv(path1)
    data_valid = pd.read_csv(path2)
    data_test = pd.read_csv(path3)

    data_arr = np.array(data_train)
    data_valid_arr = np.array(data_valid)
    data_test_arr = np.array(data_test)

    #print(data_test_arr)

    length = len(list(data_train.columns))
    header = [i for i in range(1,length)]
    header.append('labels')
    data_test.columns = header
    data_valid.columns = header
    data_test.columns = header	

    DecisionTreeID3.max_label = DecisionTreeID3.dataclassify(data_arr)

    visited = [0]*(length-1)

    if(args.Impurity_Heuristic == 'entropy'):
        #id3-entropy
        root_without_prune = DecisionTreeID3.ID3(data_arr,visited)
        #print(root1.val)
        print('Accuracy obtained for the given Dataset with Decision Tree Classifier with entropy is:',DecisionTreeID3.GetAccuracy(root_without_prune,data_test_arr))
 
    if(args.Impurity_Heuristic == 'variance'):
        #id3-variance
        root_without_prune = DecisionTreeID3.ID3_V(data_arr,visited)
        print('Accuracy obtained for the given Dataset with Decision Tree Classifier with variance is:',DecisionTreeID3.GetAccuracy(root_without_prune,data_test_arr))    

    if(args.Pruning_Type == 'depth_based'):
        #id3-Depth Prune
        root_with_prune = DecisionTreeID3.DepthBasedPruning(root_without_prune,data_arr,data_valid_arr)
        print('Accuracy obtained for the given Dataset with Decision Tree Classifier with depth based pruning is:',DecisionTreeID3.GetAccuracy(root_with_prune,data_test_arr))

    if(args.Pruning_Type == 'reduced_error'):
        #id3-Reduced_error
        root_with_prune =DecisionTreeID3.BuildTree_Reduced_Error(data_arr, data_valid, root_without_prune)	
        print('Accuracy obtained for the given Dataset with Decision Tree Classifier with reduced error pruning is:',DecisionTreeID3.GetAccuracy(root_with_prune,data_test_arr))
