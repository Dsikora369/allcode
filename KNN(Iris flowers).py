import numpy as np
import matplotlib.pyplot as plt
#loading data
def load_data():
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    
    iris = datasets.load_iris()
    
    train_set_x, test_set_x, train_set_y, test_set_y = train_test_split(iris.data, iris.target, test_size=0.33, random_state=42)
    
    return train_set_x, test_set_x, train_set_y, test_set_y, iris

train_set_x, test_set_x, train_set_y, test_set_y, visualization_set = load_data()
#getting shapes to avoid mistakes
### START CODE HERE ### (≈ 2 lines of code)
m_train = train_set_x.shape[0]
m_test = test_set_x.shape[0]
### END CODE HERE ###

print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))

print ("train_set_x shape: " + str(train_set_x.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
#plotting amount of figures in each class
plt.figure(figsize=(4, 3))
plt.hist(visualization_set.target)
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.pause(2)
#plotting distribution
for index, feature_name in enumerate(visualization_set.feature_names):
    plt.figure(figsize=(4, 3))
    plt.scatter(visualization_set.data[:, index], visualization_set.target)
    plt.ylabel("Class", size=15)
    plt.xlabel(feature_name, size=15)
    plt.tight_layout()
    plt.pause(2)
#function for distance
def euclidian_dist(x_known,x_unknown):
    """
    This function calculates euclidian distance between each pairs of known and unknown points
    
    Argument:
    x_known -- array of training data with shape (num_examples, num_features)
    x_unknown -- array of test data with shape (num_examples, num_features)
    
    Returns:
    dists -- array of euclidian distances between each pairs of known and unknown points, 
    initialized as np.array of zeros with shape of (num_pred,num_data)
    
    """
    num_pred = x_unknown.shape[0]
    num_data = x_known.shape[0]
    
    
    ### START CODE HERE ### (≈ 1 line of code)
    dists = np.zeros(shape=(num_pred, num_data))
    ### END CODE HERE ###

    for i in range(num_pred):
        for j in range(num_data):
            # calculate euclidian distance here
            ### START CODE HERE ### (≈ 1-2 lines of code)
            dists[i,j] = np.sqrt(np.sum(np.power(x_known[j]-x_unknown[i], 2)))
            ### END CODE HERE ###
            
    return dists
x1 = np.array([[1,1], [3,3], [4, 4]])
x2 = np.array([[2,2],[3,3], [5, 5]])
d = euclidian_dist(x1, x2)
print(d)
#function to find k nearest
def k_nearest_labels(dists, y_known, k):
    """
    This function returns labels of k-nearest neighbours to each sample for unknown data.
    
    Argument:
    dists -- array of euclidian distances between each pairs of known and unknown points
    with shape (num_test_examples, num_train_examples)
    y_known -- array of train data labels
    k -- scalar, which means number of nearest neighbours
    
    Returns:
    knearest_labels -- array with shape (num_samples, k) which contains labels of k nearest neighbours for each sample 
    
    """
        
    num_pred = dists.shape[0]
    n_nearest = []
    
    for j in range(num_pred):
        closest_y = []
        dst = dists[j]
        # count k closest points 
        ### START CODE HERE ### (≈ 1-2 lines of code)
        for i in np.argpartition(dst, k-1)[:k]:
            closest_y.append(y_known[i])
        ### END CODE HERE ###
        
        n_nearest.append(closest_y)
    return np.asarray(n_nearest)
y = np.array([2, 3, 1])
knl = k_nearest_labels(d, y, 2)
print(knl)
#final model
class KNearest_Neighbours(object):
    """
    Parameters:
    -----------
    k: int
        The number of nearest neighbours
    """
    def __init__(self, k):
        
        self.k = k
        self.test_set_x = None
        self.train_set_x = None
        self.train_set_y = None

        
    def fit(self, train_set_x, train_set_y):
        
        ### START CODE HERE ### 
        self.train_set_x = train_set_x
        self.train_set_y = train_set_y

        
        ### END CODE HERE ###
        
    def predict(self, test_set_x):
        
        # Returns list of predicted labels for test set; type(prediction) -> list, len(prediction) = len(test_set_y)
        ### START CODE HERE ### 
        dists = euclidian_dist(self.train_set_x, test_set_x)
        k_nearest = k_nearest_labels(dists, self.train_set_y, self.k)
        y_predict = []
        for elem in k_nearest:
            max = 0
            name = ''
            for c in elem:
                if list(elem).count(c)>max:
                    max = list(elem).count(c)
                    name = c
            y_predict.append(name)
        return y_predict
        
        ### END CODE HERE ###
k = 4
model = KNearest_Neighbours(k)
model.fit(train_set_x, train_set_y)
#checking our model
y_predictions = model.predict(test_set_x)
actual = list(test_set_y)
accuracy = (y_predictions == test_set_y).mean()
print(accuracy)
#plotting our predictions
for index, feature_name in enumerate(visualization_set.feature_names):
    plt.figure(figsize=(4, 3))
    plt.scatter(test_set_x[:, index], test_set_y) # real labels
    plt.scatter(test_set_x[:, index], y_predictions) # predicted labels
    plt.ylabel("Class", size=15)
    plt.xlabel(feature_name, size=15)
    plt.tight_layout()
    plt.pause(2)
