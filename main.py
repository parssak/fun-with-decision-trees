'''
Each headline appears as a single line in the data file. Words in the headline are separated
by spaces, so just use str.split() in Python to split the headlines into words.

You will build a decision tree to classify real vs. fake news headlines. Instead of coding
the decision trees yourself, you will do what we normally do in practice — use an existing
implementation. You should use the DecisionTreeClassifier included in sklearn. Note
that figuring out how to use this implementation is a part of the assignment.
'''

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report


seed = 42


def load_data():
    # Load the data
    with open('./clean_fake.txt', 'r') as f:
        fake = [l.strip() for l in f.readlines()]
    with open('./clean_real.txt', 'r') as f:
        real = [l.strip() for l in f.readlines()]

    # Each element is a string, corresponding to a headline
    data = np.array(real + fake)
    labels = np.array([0]*len(real) + [1]*len(fake))

    # Splitting 15% of the data for testing
    train_X, test_X, train_Y, test_Y = train_test_split(
        data, labels, test_size=0.15, random_state=seed, stratify=labels)

    # Splitting 15% of the training data for validation
    train_X, val_X, train_Y, val_Y = train_test_split(
        train_X, train_Y, test_size=0.1765, random_state=seed)

    # Preprocess the data
    vectorizer = CountVectorizer()
    train_X = vectorizer.fit_transform(train_X)
    val_X = vectorizer.transform(val_X)
    test_X = vectorizer.transform(test_X)

    return train_X, val_X, test_X, train_Y, val_Y, test_Y


'''
    Train the decision tree classifier using at least 5 different values of max_depth,
    as well as two different split criteria (information_gain & gini_coefficient).

    Evaluate the accuracy of the classifier on the validation set.

    Print the results of the evaluation.
'''
def select_model(train_X, train_Y, val_X, val_Y):
    max_depth = [1, 2, 3, 4, 5]
    split_criteria = ['gini', 'entropy']
    
    best_model = None
    best_accuracy = 0
    
    for d in max_depth:
        for s in split_criteria:
            clf = DecisionTreeClassifier(
                max_depth=d, criterion=s, random_state=seed)
            clf.fit(train_X, train_Y)
            pred = clf.predict(val_X)
            print(pred)
            accuracy = accuracy_score(val_Y, pred)
            if accuracy > best_accuracy:
                best_model = clf
                best_accuracy = accuracy
            print('Accuracy of max-depth: {}, criterion: {}: {}'.format(d, s, accuracy))
    print('Best model:', best_model, best_model.criterion)
    return best_model

    
def main():
    train_X, val_X, test_X, train_Y, val_Y, test_Y = load_data()
    model = select_model(train_X, train_Y, val_X, val_Y)

if __name__ == '__main__':
    main()
