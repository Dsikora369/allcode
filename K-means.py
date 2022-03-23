import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from JSAnimation import IPython_display
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
#making random set
np.random.seed(37)
X = np.vstack(((np.random.randn(150, 2)  + np.array([3, 0])),
               (np.random.randn(100, 2)  + np.array([-3.5, 0.5])),
               (np.random.randn(100, 2) + np.array([-0.5, -2])),
               (np.random.randn(150, 2) + np.array([-2, -2.5])),
               (np.random.randn(150, 2) + np.array([-5.5, -3]))))
print('First five examples: ', X[:5])
print('X.shape:', X.shape)
#plotting all examples
'''plt.scatter(X[:, 0], X[:, 1], s=30)
ax = plt.gca()
plt.pause(2)'''
#final class
# GRADED CLASS: KMeans

class KMeans(object):
    """
    Parameters:
    -----------
    X -- np.array
        Matrix of input features
    k -- int
        Number of clusters
    """
    
    def __init__(self, X, k):
        self.X = X
        self.k = k
        
    def initialize_centroids(self):
        """ 
        Returns:
        
        Array of shape (k, n_features), 
            containing k centroids from the initial points
        """
        
        ### START CODE HERE ###
        # use shuffle with random state = 512, and pick first k points
        X = np.copy(self.X)
        np.random.RandomState(512).shuffle(X)
        return X[:self.k]
        ### END CODE HERE ###
             
    def closest_centroid(self, centroids):
        """
        Returns:
        
        Array of shape (n_examples, ), 
            containing index of the nearest centroid for each point
        """
        min = np.zeros(shape = (self.X.shape[0], ))
        ### START CODE HERE ###
        for i,elem in enumerate(self.X):
            ranges = np.power(centroids - elem, 2)
            ranges = ranges[:,0]+ranges[:,1]
            min[i] = int(np.argmin(ranges))
            
        return min
        ### END CODE HERE ###
    
    def move_centroids(self, centroids):
        """
        Returns:
        
        Array of shape (n_clusters, n_features),
        containing the new centroids assigned from the points closest to them
        """
        
        ### START CODE HERE ###
        c = np.zeros(shape=centroids.shape)
        dict = {}
        list = self.closest_centroid(centroids)
        for i,elem in enumerate(list):
            c[int(elem)]+=self.X[i]
            if elem not in dict:
                dict[elem]=1
            else:
                dict[elem]+=1
        
        for i in range(self.k):
            c[i]/=dict[i]
        return c
            
        ### END CODE HERE ###
        

    def final_centroids(self):
        """
        Returns:
        
        clusters -- list of arrays, containing points of each cluster
        centroids -- array of shape (n_clusters, n_features),
            containing final centroids 
        
        """
        
        ### START CODE HERE ###
        centroids = self.initialize_centroids()
        new_centroids = self.move_centroids(centroids)
        mask = (centroids==new_centroids).mean()
        while mask!=1:
            centroids, new_centroids = new_centroids, self.move_centroids(new_centroids)
            mask = (centroids==new_centroids).mean()
        points = self.closest_centroid(centroids)
        dict = {}
        for i in range(self.k):
            dict[i] = []
        for i, elem in enumerate(points):
                dict[elem].append(self.X[i])
        clusters = []
        for elem in dict:
            clusters.append(np.array(dict[elem]))
        
        ### END CODE HERE ###

        return np.array(clusters), centroids
#checking all functions
model = KMeans(X, 3)

centroids = model.initialize_centroids()
print('Random centroids:', centroids)

plt.scatter(X[:, 0], X[:, 1], s=30)
plt.scatter(centroids[:,0], centroids[:,1], s=600, marker='*', c='r')
ax = plt.gca()
plt.pause(2)

closest = model.closest_centroid(centroids)
print('Closest centroids:', closest[:10])

plt.scatter(X[:, 0], X[:, 1], s=30, c=closest)
plt.scatter(centroids[:,0], centroids[:,1], s=600, marker='*', c='r')
ax = plt.gca()
plt.pause(2)

next_centroids = model.move_centroids(centroids)
print('Next centroids:', next_centroids)

clusters, final_centrs = model.final_centroids()
print('Final centroids:', final_centrs)
print('Clusters points:', clusters[0][0], clusters[1][0], clusters[2][0])

from JSAnimation import IPython_display
from matplotlib import animation
fig1 = plt.figure()
ax = plt.axes(xlim=(-4, 4), ylim=(-4, 4))
centroids = model.initialize_centroids()

line1, = ax.plot([], [], 'o')
line2, = ax.plot([], [], 'o')

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    return line1,line2

def animate(i):
    global centroids
    closest = model.closest_centroid(centroids)
    centroids = model.move_centroids(centroids)
    ax.cla()
    ax.scatter(X[:, 0], X[:, 1], c=closest)
    ax.scatter(centroids[:, 0], centroids[:, 1],  marker='*', c='r', s=600)
    
    line1.set_data(X[:, 0] , X[:, 1])
    line2.set_data(centroids[:, 0] ,centroids[:, 1])
    return line1, line2

animation.FuncAnimation(fig1, animate, init_func=init,
                        frames=15, interval=150, blit=True)
#mean distances to choose k
# GRADED FUNCTION: mean_distances

def mean_distances(k, X):
    """
    Arguments:
    
    k -- int, number of clusters
    X -- np.array, matrix of input features
    
    Returns:
    
    Array of shape (k, ), containing mean of sum distances 
        from centroid to each point in the cluster for k clusters
    """
    
    ### START CODE HERE ###
    list1 = []
    for i in range(1, k+1):
        model = KMeans(X, i)
        clusters, centroids = model.final_centroids()
        sum = 0 
        for j in range(i):
            sum+=np.sum(np.power(clusters[j]-centroids[j], 2))
        list1.append(sum/i)
    return list1
    ### END CODE HERE ###
print('Mean distances: ', mean_distances(10, X))
k_clusters = range(1, 11)
distances = mean_distances(10, X)
plt.plot(k_clusters, distances)
plt.xlabel('k')
plt.ylabel('Mean distance')
plt.title('The Elbow Method showing the optimal k')
plt.show()
plt.pause(2)
model_new = KMeans(X, 4)
fig2 = plt.figure()
ax = plt.axes(xlim=(-4, 4), ylim=(-4, 4))
centroids = model_new.initialize_centroids()

line1, = ax.plot([], [], 'o')
line2, = ax.plot([], [], 'o')

def init():
    
    line1.set_data([], [])
    line2.set_data([], [])
    return line1,line2

def animate(i):
    global centroids
    closest = model_new.closest_centroid(centroids)
    centroids = model_new.move_centroids(centroids)
    ax.cla()
    ax.scatter(X[:, 0], X[:, 1], c=closest)
    ax.scatter(centroids[:, 0], centroids[:, 1],  marker='*', c='r', s=600)
    line1.set_data(X[:, 0] , X[:, 1])
    line2.set_data(centroids[:, 0] ,centroids[:, 1])
    return line1, line2

animation.FuncAnimation(fig2, animate, init_func=init,
                        frames=30, interval=200, blit=True)

print("End")
