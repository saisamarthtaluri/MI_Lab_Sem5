from hashlib import new
from operator import ne
from random import randint
import numpy as np
import math
from decimal import Decimal

class KNN:
    """
    K Nearest Neighbours model
    Args:
        k_neigh: Number of neighbours to take for prediction
        weighted: Boolean flag to indicate if the nieghbours contribution
                  is weighted as an inverse of the distance measure
        p: Parameter of Minkowski distance
    """

    def __init__(self, k_neigh, weighted=False, p=2):

        self.weighted = weighted
        self.k_neigh = k_neigh
        self.p = p

    def fit(self, data, target):
        """
        Fit the model to the training dataset.
        Args:
            data: M x D Matrix( M data points with D attributes each)(float)
            target: Vector of length M (Target class for all the data points as int)
        Returns:
            The object itself
        """

        self.data = data
        self.target = target.astype(np.int64)

        return self

    def helper1(self, value, root):
        my_root_value = 1 / float(root)
        final_value = Decimal(value) ** Decimal(my_root_value)
        return round (final_value, 3)
        
    def helper2(self, x, y, p_value):
        summation = sum(pow(abs(m-n), p_value) for m, n in zip(x, y))
        return float(self.helper1(summation , p_value))
    def find_distance(self, x):
        """
        Find the Minkowski distance to all the points in the train dataset x
        Args:            x: N x D Matrix (N inputs with D attributes each)(float)
        Returns:
            Distance between each input to every data point in the train dataset
            (N x M) Matrix (N Number of inputs, M number of samples in the train dataset)(float)
        """
        #print(self.p)
        if(self.p==2):
            new=[]
            for i in x:
                new1=[]
                for j in self.data:
                    new1.append(math.sqrt(((math.pow((j[1]-i[1]),2)+math.pow((j[0]-i[0]),2)))))
                new.append(new1)
            return new
        else:
            r = []

            for i in range(x.shape[0]):
                m = x[i]
                val = []
                for j in range(self.data.shape[0]): 
                    n = self.data[j]
                    val.append(self.helper2(m, n, self.p))
                r.append(val)

            return r
        pass
    def k_neighbours(self, x):
        """
        Find K nearest neighbours of each point in train dataset x
        Note that the point itself is not to be included in the set of k Nearest Neighbours
        Args:
            x: N x D Matrix( N inputs with D attributes each)(float)
        Returns:
            k nearest neighbours as a list of (neigh_dists, idx_of_neigh)
            neigh_dists -> N x k Matrix(float) - Dist of all input points to its k closest neighbours.
            idx_of_neigh -> N x k Matrix(int) - The (row index in the dataset) of the k closest neighbours of each input

            Note that each row of both neigh_dists and idx_of_neigh must be SORTED in increasing order of distance
        """
        ans=self.find_distance(x)
        #print(ans)
        ne=self.k_neigh
        #print(ne)
        new=[]
        ind=[]
        act=[]
        for i in ans:
            new1=[]
            new2=[]
            for k in range(ne):
                #print(self.k_neigh)
                x=min(i)
                new1.append(x)
                z=i.index(x)
                new2.append(z)
                i[z]=10000000
                new1.sort()
            new.append(new1)
            ind.append(new2)
        new=np.array(new)
        ind=np.array(ind)
        act.append(new)
        act.append(ind)
        return act

    def predict(self, x):
        """
        Predict the target value of the inputs.
        Args:
            x: N x D Matrix( N inputs with D attributes each)(float)
        Returns:
            pred: Vector of length N (Predicted target value for each input)(int)
        """
        # TODO
        ans=self.k_neighbours(x)
        ind=ans[1]
        r=[]
        for i in range(len(ind)):
            f={}
            for j in range(len(ind[i])):
                if self.target[ind[i][j]] in f:
                    f[self.target[ind[i][j]]]=f[self.target[ind[i][j]]]+1
                else:
                    f[self.target[ind[i][j]]]=1
            ma=0
            maxi=None
            for i in range(min(f),max(f)+1):
                if f[i]>ma:
                    ma=f[i]
                    maxi=i
            r.append(maxi)
        return r
        pass

    def evaluate(self, x, y):
        """
        Evaluate Model on test data using 
            classification: accuracy metric
        Args:
            x: Test data (N x D) matrix(float)
            y: True target of test data(int)
        Returns:
            accuracy : (float.)
        """
        # TODO
        pred=self.predict(x)
        count=0
        for i in range(len(pred)):
            if(pred[i]==y[i]):
                count=count+1
        return(round((count/len(pred))*100,2))
        pass
