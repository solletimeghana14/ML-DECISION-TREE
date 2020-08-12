Run Command for all the 6 Algorithms

python Run_DecisionTree_Classifier.py [-h] [-c {entropy, variance}] [-p {depth_based,reduced_error}] train_data valid_data test_data

positional arguments:
train_data give the path of train dataset
valid_data give the path of valid dataset
test_data give the path of test dataset

optional arguments:
-h, --help   shows help message and exit
-c {entropy, variance}, --criterion {entropy,variance}
Mention Impurity Heuristic . Default is Entropy
-p {depth_based,reduced_error}, --prune {depth_based,reduced_error}
Mention Pruning Method
