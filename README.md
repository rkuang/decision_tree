# decision_tree

Using scikit-learn, I created a Decision Tree Classifier for a dummy problem on my Machine Learning class at UCSB.

I had to parse the training data from a .txt file, and reshape my numpy arrays into inputs compatible with the scikit-learn DecisionTreeClassifier.

This program requires ```scikit-learn``` and ```numpy```.

To run, do ```python hw3.py training.txt test1.txt```

The program fits a decision tree on the provided training data. Then, it predicts the examples in the provided test data. The program outputs various metrics on the accuracy of the model