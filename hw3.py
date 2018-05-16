# Starter code for CS 165B HW3
import numpy as np
from sklearn.tree import DecisionTreeClassifier

BUDGET = 0
GENRE = 1
FAMOUS_ACTORS = 2
DIRECTOR = 3
GOOD_MOVIE = 4

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
    print training

    X_train = training[:,:4]
    Y_train = training[:,4]

    gini = DecisionTreeClassifier(random_state=0)
    gini.fit(X_train, Y_train)

    # return {
    #     "gini":{
    #         'True positives':0, 
    #         'True negatives':0, 
    #         'False positives':0, 
    #         'False negatives':0, 
    #         'Error rate':0.00
    #         },
    #     "entropy":{
    #         'True positives':0, 
    #         'True negatives':0, 
    #         'False positives':0, 
    #         'False negatives':0, 
    #         'Error rate':0.00
    #         }
    # }
    pass


def parse_file(file):
    next(file)
    data = [[int(y) for y in x.strip().split()[1:]] for x in file]
    return data


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

    run_train_test(training_file, testing_file)

    training_file.close()
    testing_file.close()

