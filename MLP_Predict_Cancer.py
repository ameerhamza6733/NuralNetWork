from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import shuffle


def process_data(data):
    x = data.iloc[1:, 0:42]
    y = data.iloc[1:, 42:43]
    print(data)
    return x, y


def to_one_hot_encoding(date_frame, column_index):
    one_hot_encoder = OneHotEncoder(categorical_features=[column_index])
    return one_hot_encoder.fit_transform(date_frame).toarray()


pd.set_option('display.max_columns', 50)

orignal_data = pd.read_csv("cancer.csv")

encoded_data = orignal_data.apply(LabelEncoder().fit_transform)

print(pd.DataFrame(encoded_data))

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

full_x, full_y = process_data(encoded_data)
x_train, x_test, y_train, y_test = train_test_split(full_x, full_y, test_size=0.20, random_state=4)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(2,), random_state=1)
clf.fit(x_train, pd.DataFrame(y_train).values.ravel())

print("test data ")
print(pd.DataFrame(x_test).join(y_test))
print("\n")
print(clf.predict(x_test))

