from keras import Sequential, optimizers
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import shuffle
from theano.tensor.nnet import opt
import numpy as np


def process_data(data):
    x = data[:, 0:42]
    y = data[:, 42:43]

    return x, y

def process_pands_data(data):
    x = data.iloc[:, 0:42]
    y = data.iloc[:, 42:43]

    return x, y


def to_one_hot_encoding(date_frame, column_index):
    one_hot_encoder = OneHotEncoder(categorical_features=[column_index])
    return one_hot_encoder.fit_transform(date_frame).toarray()


orignal_data = pd.read_csv("cancer.csv")

encoded_data = orignal_data.apply(LabelEncoder().fit_transform)


encoded_data = to_one_hot_encoding(encoded_data, 0)
encoded_data = to_one_hot_encoding(encoded_data, 2)
encoded_data = to_one_hot_encoding(encoded_data, 8)
encoded_data = to_one_hot_encoding(encoded_data, 11)
encoded_data = to_one_hot_encoding(encoded_data, 22)
encoded_data = to_one_hot_encoding(encoded_data, 29)
encoded_data = to_one_hot_encoding(encoded_data, 32)
encoded_data = to_one_hot_encoding(encoded_data, 35)
encoded_data = to_one_hot_encoding(encoded_data, 37)

encoded_data = shuffle(pd.DataFrame(encoded_data))

full_x, full_y = process_pands_data(encoded_data)

x_train, x_test, y_train, y_test = train_test_split(full_x, full_y, test_size=0.10, random_state=4)

pd.DataFrame(x_train).join(y_train).to_csv("encoding_cancer_traning_data.csv",index=False,header=False)
pd.DataFrame(x_test).join(y_test).to_csv("encoding_cancer_test_data.csv",index=False,header=False)

train_data_set = np.loadtxt("encoding_cancer_traning_data.csv", delimiter=",")
test_data_set = np.loadtxt("encoding_cancer_test_data.csv", delimiter=",")

x_train, y_train = process_data(train_data_set)
x_test, y_test = process_data(test_data_set)

model = Sequential()
model.add(Dense(50, input_dim=42, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
# Fit the model
av = model.evaluate(x_test, y_test)

model.fit(x_train,y_train, epochs=1000, batch_size=10)
# evaluate the model

predictions = model.predict_classes(x_test)
# round predictions
print(predictions)

print(x_test)
print(y_test)
# clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(2,), random_state=1)
# clf.fit(x_train, pd.DataFrame(y_train).values.ravel())
#
# print("test data ")
# print(pd.DataFrame(x_test).join(y_test))
# print("\n")
# print(clf.predict(x_test))
