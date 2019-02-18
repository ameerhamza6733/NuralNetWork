import pandas as pd
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier


def process(_data):
    x = _data.iloc[:, 1:17]
    y = _data.iloc[:, 17]

    return x, y


def process_test_data(test_data):
    test_X = test_data.iloc[0:1, 1:17]

    return test_X


data = pd.read_csv("suffled_zoo.csv")
test_data = pd.read_csv("test_zoo_data_set.csv")
data = shuffle(data)
print("before shufflle")
print(data)

all_X, all_y = process(data)
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
clf.fit(all_X, all_y)

print("test data ")
print(process_test_data(test_data))
print(clf.predict(process_test_data(test_data)))

