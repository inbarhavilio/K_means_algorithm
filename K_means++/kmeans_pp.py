# imports
import sys
import argparse
import numpy as np
from numpy import matrix
import pandas as pd
import scipy
import mykmeanssp  # C implementation import


####################################
# Reading arguments from cmd/user  #
####################################

parser = argparse.ArgumentParser()

parser.add_argument("K", type=int)
parser.add_argument('max_iter', type=int, default=300, nargs="?")

parser.add_argument('file_name_1')
parser.add_argument('file_name_2')

args = parser.parse_args()

# variables from user input
K = args.K
max_iter = args.max_iter

file_1 = pd.read_csv(args.file_name_1, header=None)
file_2 = pd.read_csv(args.file_name_2, header=None)


# boundary conditions on user input
if args.K < 1:
    print('Error: K argument is smaller than 1!')
    exit()

if args.max_iter < 1:
    print('Error: Max iterations argument is smaller than 1!')
    exit()

if type(args.K) != int or type(args.max_iter) != int:
    print('Error: Arguments need to be integers!')
    exit()

if type(args.file_name_1) != str or type(args.file_name_2) != str:
    print('Error: Files need to be correct paths')
    exit()


##############################
# Inner Join using pandas    #
##############################

# use merge to create inner join of the two files, on the index column
merged_file = pd.merge(file_1, file_2, on=0)

sorted_merged_file = merged_file.sort_values(by=0)  # sort by the index given
sorted_merged_file.drop(columns=sorted_merged_file.columns[0], axis=1, inplace=True)  # drop the first column of indices of points
np_merged_file = sorted_merged_file.to_numpy()  # create a numpy array from pandas data frame to use in kmeans++ algorithm



# check dimensions of different points
def check_dimension(points, d):
    for point in points:
        if len(point) != d:
            print('Error: Dimensions of points are different!')
            exit()


n_row, n_col = np_merged_file.shape  # Number of points and dimension
N = n_row
d = n_col

if K >= N:
    print("Error: Number of clusters cannot be larger or equal to the number of points!")
    exit()

if N < 1:
    print('Error: No points in file!')
    exit()


check_dimension(np_merged_file, d)


#######################################################
# K - means ++ algorithm  implementation              #
# Choose initial centroids for the K means algorithm  #
# Implementation with numpy module                    #
#######################################################


def kmeans_pp(mat, K):
    # random seed
    np.random.seed(0)

    # randomly choose the first centroid
    N, d = mat.shape

    centroids = np.zeros((K, d))
    init_index = np.random.choice(N)
    centroids[0, :] = mat[init_index, :]

    # initialize indices
    indices = np.zeros(K)
    indices[0] = init_index

    minInd = np.array([0 for m in range(N)])
    dist = np.array([0.0 for i in range(N)])
    distSum = 0
    # loop for all centroids
    z = 0
    while z < K - 1:

        for i in range(N):
            if z == 0:
                newVal = norm(mat[i], centroids[z])
                minInd[i] = z
                distSum += newVal
                dist[i] = newVal
            else:
                newVal = norm(mat[i], centroids[z])
                if newVal < dist[i]:
                    minInd[i] = z
                    distSum = distSum - dist[i] + newVal
                    dist[i] = newVal
        z += 1
        # calculate probability
        prob = dist / distSum
        # choose index using the probability
        rand_index = np.random.choice(N, size=1, p=prob)
        # update centroids
        centroids[z, :] = mat[rand_index, :]
        # update indices
        indices[z] = rand_index
        # update closest

    np.sort(indices)

    return centroids, indices


def norm(A, B):
    Nsum = 0
    for i in range(len(A)):
        Nsum = Nsum + (A[i] - B[i]) ** 2
    return Nsum

# printing functions for output
def print_centroids(centroids):
    rounded_centroids = matrix.round(np.array(centroids), 4)
    for i in range(len(rounded_centroids)):
        print(*rounded_centroids[i], sep=',')
    return 0

def print_indices(indices):
    int_indices = np.array(indices, int)
    print(*int_indices, sep=',')
    return 0


#################################
# Interfacing with C extension  #
#################################

# find the initial centroids and indices from the kmeans_pp implementation
initial_centroids, indices_init_centroids = kmeans_pp(np_merged_file, K)

# run the k means algorithm from the C implementation
# returns the clusters from the K means algorithm
updated_centroids = mykmeanssp.kmeans_pp(np_merged_file, initial_centroids, N, K, d, max_iter)



###########################################
# Output of K means algorithm as expected #
###########################################

print_indices(indices_init_centroids)
print_centroids(updated_centroids)

