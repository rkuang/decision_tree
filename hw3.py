# Starter code for CS 165B HW3
import numpy as np
from sklearn.tree import DecisionTreeClassifier

def run_train_test(training_file, testing_file):
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition. 

    Inputs:
        training_file: file object returned by open('traininig.txt', 'r')
        testing_file: file object returned by open('test1/2/3.txt', 'r')

    Output:
        Dictionary of result values 

        IMPORTANT: YOU MUST USE THE SAME DICTIONARY KEYS SPECIFIED
        
        Example:
            return {
    			"gini":{
    				'True positives':0, 
    				'True negatives':0, 
    				'False positives':0, 
    				'False negatives':0, 
    				'Error rate':0.00
    				},
    			"entropy":{
    				'True positives':0, 
    				'True negatives':0, 
    				'False positives':0, 
    				'False negatives':0, 
    				'Error rate':0.00}
    				}
    """

    training = parse_file(training_file)
    training = np.array(training)

    X_train = training[:,:4]
    Y_train = training[:,4]

    testing = parse_file(testing_file)
    testing = np.array(testing)

    X_test = testing[:,:4]
    Y_test = testing[:,4]

    gini_clf = DecisionTreeClassifier(random_state=0)
    gini_clf.fit(X_train, Y_train)
    gini_Y_hat = gini_clf.predict(X_test)
    gini_tp, gini_tn, gini_fp, gini_fn, gini_err = eval_results(Y_test, gini_Y_hat)

    entropy_clf = DecisionTreeClassifier(criterion="entropy", random_state=0)
    entropy_clf.fit(X_train, Y_train)
    entropy_Y_hat = entropy_clf.predict(X_test)
    entropy_tp, entropy_tn, entropy_fp, entropy_fn, entropy_err = eval_results(Y_test, entropy_Y_hat)

    return {
        "gini":{
            'True positives': gini_tp,
            'True negatives': gini_tn,
            'False positives': gini_fp,
            'False negatives': gini_fn,
            'Error rate': gini_err
        },
        "entropy":{
            'True positives': entropy_tp,
            'True negatives': entropy_tn,
            'False positives': entropy_fp,
            'False negatives': entropy_fn,
            'Error rate': entropy_err
        }
    }


def parse_file(file):
    next(file)
    data = [[int(y) for y in x.strip().split()[1:]] for x in file]
    return data


def eval_results(Y_test, Y_hat):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(Y_test)):
        if Y_test[i] == Y_hat[i]:
            if Y_hat[i] == 1:
                tp += 1
            elif Y_hat[i] == 0:
                tn += 1
        else:
            if Y_hat[i] == 1:
                fp += 1
            elif Y_hat[i] == 0:
                fn += 1
    err = (fp+fn)/float(len(Y_test))
    return tp, tn, fp, fn, err
        

#######
# The following functions are provided for you to test your classifier.
#######

if __name__ == "__main__":
    """
    You can use this to test your code.
    python hw3.py [training file path] [testing file path]
    """
    import sys

    training_file = open(sys.argv[1], "r")
    testing_file = open(sys.argv[2], "r")

    print run_train_test(training_file, testing_file)

    training_file.close()
    testing_file.close()

