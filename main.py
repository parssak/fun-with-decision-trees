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


'''
    Computes the information gain of a split on the training data
    Compute I(Y;X) for each attribute value x_i
    Compute H(Y) = -sum(p(y)*log2(p(y)))
    Compute H(Y|X) = H(Y) - sum(p(y|x)*I(Y;X))
    Return the information gain of the split
'''
'''

=======

def compute_information_gain(y, x):
    y_prob = np.bincount(y) / len(y)
    x_prob = np.bincount(x) / len(x)
    y_x_prob = np.multiply(y_prob, x_prob)
    y_x_prob = y_x_prob / np.sum(y_x_prob)

    h_y = -np.sum(y_prob * np.log2(y_prob))
    h_y_x = h_y - np.sum(y_x_prob * np.log2(y_x_prob))
    return h_y_x

=======

def compute_information_gain(Y, X):
    H_Y = 0
    H_Y_X = 0
    for y in Y:
        p_y = sum(y) / len(Y)
        if p_y == 0:
            continue
        H_Y -= p_y * np.log2(p_y)
    for y in Y:
        p_y_x = sum(y[X == 1]) / sum(y)
        if p_y_x == 0:
            continue
        H_Y_X -= p_y_x * np.log2(p_y_x)
    return H_Y - H_Y_X

=======

def compute_information_gain(X, Y):
    H = 0
    for i in range(X.shape[1]):
        X_i = X[:, i]
        H_i = 0
        for x_i, y_i in zip(X_i, Y):
            p_y_i = sum(Y)/len(Y)
            p_y_i_x_i = sum(X_i == x_i)/len(X_i)
            p_y_i_x_i_y_i = p_y_i_x_i * p_y_i
            H_i += p_y_i_x_i_y_i * np.log2(p_y_i_x_i_y_i)
        H += H_i
    return H - sum(Y)/len(Y) * sum(X.sum(axis=0) * np.log2(X.sum(axis=0)))

=======

def compute_information_gain(X, Y):
    H = -sum(Y*np.log2(Y))
    H_Y_X = []
    for i in range(len(X[0])):
        H_Y_X.append(0)
        for j in range(len(X)):
            if X[j][i] == 1:
                H_Y_X[i] -= sum(Y[j]*np.log2(Y[j]))
            else:
                H_Y_X[i] -= sum(1-Y[j]*np.log2(1-Y[j]))
    return H - sum(H_Y_X)

=======

def compute_information_gain(train_Y, train_X, split_criteria):
    n = len(train_Y)
    H = -sum(train_Y * np.log2(train_Y))
    for i in range(train_X.shape[1]):
        x_i = train_X[:, i]
        p_x_i = np.count_nonzero(x_i) / n
        if split_criteria == 'gini':
            H -= p_x_i * (1 - p_x_i)
        elif split_criteria == 'entropy':
            H -= p_x_i * np.log2(p_x_i)
    return H

=======

def compute_information_gain(X, Y, attribute_index, threshold):
    # Compute the entropy of Y
    H = -np.sum(np.log2(Y + 1e-10) * Y + 1e-10)

    # Compute the entropy of Y|X
    H_X = 0
    for i in range(len(X)):
        # Compute the information gain of the split
        if X[i][attribute_index] < threshold:
            H_X += Y[i] * np.log2(Y[i] / (Y[i] + 1e-10))
        else:
            H_X += (1 - Y[i]) * np.log2((1 - Y[i]) / ((1 - Y[i]) + 1e-10))

    return H - H_X

=======

def compute_information_gain(X, Y, split_criteria):

=======

def compute_information_gain(y, x):

=======

def compute_information_gain(X, y, attribute_value, attribute_index):
    attribute_values = X[:, attribute_index]
    y_values = y

    # Compute the entropy of the entire set
    H = 0
    n = len(y_values)
    for y_value in np.unique(y_values):
        p_y = np.sum(y_values == y_value) / n
        H -= p_y * np.log2(p_y)

    # Compute the entropy for the subset of the data
    H_X = 0
    for attribute_value in np.unique(attribute_values):
        p_x = np.sum(attribute_values == attribute_value) / n
        y_values_X = y_values[attribute_values == attribute_value]
        H_X -= p_x * compute_information_gain(
            X, y_values_X, attribute_value, attribute_index)

    return H - H_X

'''
    
def main():
    train_X, val_X, test_X, train_Y, val_Y, test_Y = load_data()
    model = select_model(train_X, train_Y, val_X, val_Y)

if __name__ == '__main__':
    main()
