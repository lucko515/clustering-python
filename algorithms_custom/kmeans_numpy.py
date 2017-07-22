import numpy as np


class KMeans(object):
    #INPUT: n_clusters - number of clusters that algortihm will try to find
    #       tolerance -  number when difference between prev cluster and new cluster is less then this number we will stop algo
    #       max_iter - how many times cetroids will move
    def __init__(self, n_clusters=2, tolerance = 0.001, max_iter = 300):
        self.k_clusters = n_clusters
        self.tol = tolerance
        self.max_iter = max_iter
    
    #TRAIN/FIT function, used to find the best positions for our clusters
    #
    #INPUT: X - fetures of dataset in which we are trying to find clusters
    def fit(self, X):
        #Starting clusters will be random members from X set
        self.centroids = []
        for i in range(self.k_clusters):
            #this index is used to acces random element from input set
            index = random.randint(1, len(X)-1)
            self.centroids.append(X[index])
        
        for i in range(self.max_iter):  
            #storing previous values of centroids
            prev_centroids = self.centroids
            #Centroids values for this iteration
            cen_temp = []
            
            for centroid in range(len(self.centroids)):
                #creating empty list of elements for current cluster/centroid
                list_of_closer_samples = []
                
                for k in range(len(X)):
                    helper_list = []
                    for j in range(self.k_clusters):
                        #caluclating euclidian distance between current X value and all centroids in our list
                        helper_list.append(self.euclidian_distance(self.centroids[j], X[k]))
                    
                    #if minimal distance between curent value and centroid that we are currently interested in
                    #store value to the list of examples for that centroid
                    if np.argmin(helper_list) == centroid:
                        list_of_closer_samples.append(X[k])   
                
                #New position of each cluster is calculated by mean of all examples closest to it
                cen_temp.append(np.average(list_of_closer_samples, axis=0))
           
            #check if it is optimized
            optimized = True
            for c in range(len(self.centroids)):
                original_centroid = prev_centroids[c]
                current_centroid = cen_temp[c]
                #checking if % of change between old position and new position is less then tolerance (0.001 by default)
                if np.sum((current_centroid - original_centroid)/original_centroid*100.0) > self.tol:
                    optimized = False
                    self.centroids = cen_temp
                    
            if optimized:
                break
        #return final clusters values [optional, it is only used for graph]
        c = self.centroids
        return c
    
    #euclidian distance between points
    def euclidian_distance(self, centroid, x_current):
        return np.sqrt(np.sum(np.power((centroid - x_current), 2)))
    
    #INPUT: X_test set - set of data to test our clusters on
    def predict(self, Xtest):
        pred = []
        for i in range(len(Xtest)):
            help_list = []
            for j in range(len(self.centroids)):
                help_list.append(self.euclidian_distance(self.centroids[j], Xtest[i]))
            pred.append(np.argmin(help_list))
        return pred