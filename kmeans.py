import sys
import argparse


#####################################################################################
# Should we check the running time? for input_3 and large numbers its not so good   #
#####################################################################################


# prints the centroids as expected in the exercise
def print_centroids(centroids):
    for i in range(len(centroids)):
        print(*centroids[i], sep=',')


# check dimensions of different points
def check_dimension(points, d):
    for point in points:
        if len(point) != d:
            print('Error: Dimensions of points are different!')
            exit()


###############################################
#              k means algorithm              #
###############################################

def Kmeans(K, filename, max_iter=200):
    dataP = []
    while True:
        point = sys.stdin.readline().rstrip('\n')
        if point:
            dataP.append(point)
        else:
            break
    points = [dataP[i].split(',') for i in range(len(dataP))]
    N = len(points)
    D = len(points[0])

    # check conditions:
    check_dimension(points, D)

    if N < 1:
        print('Error: No points in file!')
        exit()

    if K >= N:
        print("Error: Number of clusters cannot be larger or equal to the number of points!")
        exit()

    points = [[float(point) for point in points[i]] for i in range(N)]
    centroids = [points[i] for i in range(K)]
    centCopy = []
    i = 0
    while centroids != centCopy and i < max_iter:
        centCopy = centroids.copy()
        d = [[sum((points[j][l] - centroids[k][l]) ** 2 for l in range(D)) for k in range(K)] for j in range(N)]
        indices = [d[j].index(min(d[j])) for j in range(N)]
        ind = [[] for k in range(K)]
        for j in range(N):
            ind[indices[j]].append(points[j])
        for k in range(K):
            centroids[k] = [sum(point[m] for point in ind[k]) / (len(ind[k])) for m in range(D)]
        i = i + 1

    # note --> is rounding necessary?? should we trunc?
    centroids = [[round(centroids[m][n], 4) for n in range(D)] for m in range(K)]

    # note --> should ensure how the output needs to look
    print_centroids(centroids)

    # print(centroids)
    return centroids


# read input from user as arguments from cmd
parser = argparse.ArgumentParser()
parser.add_argument("K", type=int)
parser.add_argument('max_iter', type=int, default=200, nargs="?")
args = parser.parse_args()

# conditions
if args.K < 1:
    print('Error: K argument is smaller than 1!')
    exit()

if type(args.K) != int:
    print('Error: K argument needs to be an integer!')
    exit()

if args.max_iter < 1:
    print('Error: Max iterations argument is smaller than 1!')
    exit()

# run the Kmeans algorithm with the arguments
Kmeans(args.K, argparse.FileType('r'), int(args.max_iter))
