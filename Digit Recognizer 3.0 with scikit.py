from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
import csv
def load_data():
    train_set = pd.read_csv('train.csv').to_numpy()
    test_set_x = pd.read_csv('test.csv').to_numpy()
    train_set_y = train_set[:,0]
    train_set_x = train_set[:,1:]
    return train_set_x[:, :42000], train_set_y, test_set_x
train_set_x, train_set_y, test_set_x = load_data()
train_set_x=train_set_x/255
test_set_x=test_set_x/255
train_set_x, X_test, train_set_y, y_test = train_test_split(train_set_x, train_set_y, test_size=0.33, random_state=42)
print('train set shapes: ', train_set_x.shape, train_set_y.shape)
model = MLPClassifier(hidden_layer_sizes = (800, 800), max_iter=200, alpha=0.0001,
                    solver='sgd', verbose=10, random_state=1,
                    learning_rate_init=.01)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning,
                            module="sklearn")
    model.fit(train_set_x, train_set_y)

print("Training set score: %f" % model.score(train_set_x, train_set_y))
print("Test set score: %f" % model.score(X_test, y_test))
predict = model.predict(test_set_x)
with open('test_result3.0.csv',  newline="", mode='w') as test_file:
    test_writer = csv.writer(test_file, delimiter=',',skipinitialspace=True, quoting=csv.QUOTE_NONE)
    test_writer.writerow(['id','label'])
    for i, elem in enumerate(predict):
        test_writer.writerow([i+1, str(np.argmax(elem))])
