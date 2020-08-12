import numpy as np
import pandas as pd
import sys
import re
from argparse import ArgumentParser
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score 

def RandomForest(t_data,v_data,test_data):
    length = len(list(t_data.columns))
    header = [i for i in range(1,length)]
    header.append('labels')
    t_data.columns = header
    sk_data_t = t_data.drop(columns = ['labels'],axis = 1)
    sk_target_t = t_data['labels']
    length = len(list(v_data.columns))
    header = [i for i in range(1,length)]
    header.append('labels')
    v_data.columns = header
    sk_data_v = v_data.drop(columns = ['labels'],axis = 1)
    sk_target_v = v_data['labels']
    sk_data=sk_data_t.append(sk_data_v)
    sk_target=sk_target_t.append(sk_target_v)
    RandomForest=RandomForestClassifier(criterion='entropy')
    RandomForest.fit(sk_data,sk_target)
    length = len(list(test_data.columns))
    header = [i for i in range(1,length)]
    header.append('labels')
    test_data.columns = header
    sk_data_test = test_data.drop(columns = ['labels'],axis = 1)
    sk_target_test = test_data['labels']
    sk_target_predict = RandomForest.predict(sk_data_test)
    X=accuracy_score(np.array(sk_target_test), sk_target_predict)
    return X*100


if __name__ == '__main__':

    parser = ArgumentParser()
	
    parser.add_argument("train_data", help="Give data to be trained as input")
						
    parser.add_argument("valid_data", help="Give data to be validated as input")
						
    parser.add_argument("test_data", help="Give data to be tested as input") 

     
    args = parser.parse_args()
       
    path1 = str(args.train_data)
    path2 = str(args.valid_data)
    path3 = str(args.test_data)
    data_train = pd.read_csv(path1)
    data_valid = pd.read_csv(path2)
    data_test = pd.read_csv(path3)

    print('Accuracy obtained for the given Dataset with Random Forest Classifier is:',RandomForest(data_train,data_valid,data_test))
