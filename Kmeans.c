#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#define _CRT_SECURE_NO_WARNINGS

void kmeans(double**, double**, int N, int K, int d, int max_iter);
double norm(double*, double*, int d);
int findMinPoint(double*, double**, int K, int d);
void copy(double**, double**, int K, int d);
void updateCentroids(double** sumOfPoints, int* indices, double** points, double** centroids, int N, int K, int d);
void assignPointsToCluster(int*, double**, double**, int, int, int); /* assign point to cluster*/
int checkConvergence(double**, double**, int K, int d);
double round(double);
void printCentroids(double**, int, int); /* print centroids to cmd*/
int isInteger(double val);



/* check if convergence condition is true*/
int checkConvergence(double** centroids, double** copy, int K, int d) {
	int i, j;
	for (i = 0; i < K; i++)
		for (j = 0; j < d; j++) {
			if (centroids[i][j] != copy[i][j]) {
				return 1;
			}
		}
	return 0;
}

/* copies the clusters array to a new array */
void copy(double** centroids, double** centCopy, int K, int d) {
	int i, j;
	for (i = 0; i < K; i++) {
		for (j = 0; j < d; j++) {
			centCopy[i][j] = centroids[i][j];
		}
	}
}

/* calculates the distance of a point from a cluster*/
double norm(double* point, double* cluster, int d) {
	double distance = 0;
	double temp = 0;
	int i;
	for (i = 0; i < d; i++) {
		temp = point[i] - cluster[i];
		temp = temp * temp;
		distance += temp;
		temp = 0;
	}
	return distance;
}


/* finds the index of the cluster which is the closest to the point*/
int findMinPoint(double* point, double** centroids, int K, int d) {
	double tempDist = 0;
	double* cluster;
	double minDist = 1000; /*need to find max for C*/
	int indicator = 0, i;
	for (i = 0; i < K; i++) {
		cluster = &centroids[i][0];
		tempDist = norm(point, cluster, d);
		if (tempDist < minDist) {
			minDist = tempDist;
			indicator = i;
		}
	}
	return indicator;

}

/* assign point to centroid*/
void assignPointsToCluster(int* indices, double** points, double** centroids, int N, int K, int d) {
	int j;
	for (j = 0; j < N; j++) {
		indices[j] = findMinPoint(&points[j][0], centroids, K, d);
	}
}


/* updates cluster's centroid*/
void updateCentroids(double** sumOfPoints, int* indices, double** points, double** centroids, int N, int K, int d) {
	int i = 0, j = 0;
	double* sumOfind = malloc(K * sizeof(double));
	assert(sumOfind != NULL); /*make sure points is empty*/
	for (i = 0; i < K; i++) {
		sumOfind[i] = 0;
		for (j = 0; j < d; j++) {
			sumOfPoints[i][j] = 0;
		}
	}
	for (i = 0; i < N; i++) {
		sumOfind[indices[i]]++;
		for (j = 0; j < d; j++) {
			sumOfPoints[indices[i]][j] += points[i][j];
		}
	}
	i = 0;
	j = 0;
	for (i = 0; i < K; i++) {
		for (j = 0; j < d; j++) {
			if (sumOfind[i] > 0) {
				centroids[i][j] = sumOfPoints[i][j] / sumOfind[i];
			}
			else {
				centroids[i][j] = 0;
			}

		}
	}
}


double round(double val) {
	if (val < 0) { return ceil(val - 0.5); }
	else { return floor(val + 0.5); }
}

double power(double b, int p) {
	double ret = b;
	int i;
	for (i = 0; i < p - 1; i++) {
		ret = ret * b;
	}
	return ret;
}

void printCentroids(double** centroids, int K, int d) {
	int i, j;
	for (i = 0; i < K; i++) {
		for (j = 0; j < d; j++) {
			if (j == 0) {
				printf("%.4f", centroids[i][j]);
			}
			else {
				printf(",%.4f", centroids[i][j]);
			}
		}
		printf("\n");
	}
}

int isInteger(double val) {
	int truncated = 0;
	truncated = (int)val;
	if (val == truncated) {
		return 1;
	}
	else {
		return 0;
	}
}


void kmeans(double** points, double** centroids, int N, int K, int d, int max_iter) {

	int k = 0, i = 0;
	int* indices;
	double** centCopy;
	double** sumOfPoints;

	indices = (int*)malloc(N * sizeof(int));
	assert(indices != NULL); /*make sure points is empty*/

	centCopy = (double**)malloc(K * sizeof(double*));
	assert(centCopy != NULL); /*make sure points is empty*/
	for (k = 0; k < K; k++)
	{
		centCopy[k] = (double*)malloc(d * sizeof(double));
	}


	sumOfPoints = (double**)malloc(K * sizeof(double*));
	assert(sumOfPoints != NULL); /*make sure points is empty*/
	for (k = 0; k < K; k++)
	{
		sumOfPoints[k] = (double*)malloc(d * sizeof(double));
	}


	while (i < max_iter && checkConvergence(centroids, centCopy, K, d)) {
		copy(centroids, centCopy, K, d);
		assignPointsToCluster(indices, points, centroids, N, K, d);
		updateCentroids(sumOfPoints, indices, points, centroids, N, K, d);
		i++;
	}

	printCentroids(centroids, K, d);
	for (i = 0; i < K; i++) {
		free(sumOfPoints[i]);
	}
	free(sumOfPoints);
	for (i = 0; i < K; i++) {
		free(centCopy[i]);
	}
	free(centCopy);
	free(indices);
}

int main(int argc, char* argv[]) {

	/* initialize*/
	int K = 0; /* num of clusters from user*/
	int max_iter; /* max num of iterations from user*/

	double K_float; /* for condition use */
	double max_iter_float; /* for condition use */

	int N = 0; /* num of points --> need to initialize somehow*/
	int d = 0; /* dimension --> need to initialize somehow*/

	double** points;   /* initialize double array of points*/
	double** centroids; /* initialize double array of centroids*/

	int i = 0, j = 0, sz = 0, com = 0, count = 0, minus = 0;
	double n1 = 0, num = 0;
	char ch;

	/* assign variables from arguments*/
	if ((argc != 2) && (argc != 3)) {
		printf("Error: Number of arguments unknown!");
		return 0;
	}
	else if ((argc == 2)) {
		K = atoi(argv[1]);
		K_float = atof(argv[1]);
		if (isInteger(K_float) != 1) {
			printf("Error: Arguments must be integers!");
			assert((isInteger(K_float) == 1));
		}
		max_iter = 200; /* add default value to max_iter if not added from user*/
	}
	else if ((argc == 3)) {
		K = atoi(argv[1]);
		max_iter = atoi(argv[2]);
		K_float = atof(argv[1]);
		max_iter_float = atof(argv[2]);
		if ((isInteger(K_float) != 1) || (isInteger(max_iter_float) != 1)) {
			printf("Error: Arguments must be integers!\n");
			assert((isInteger(K_float) == 1));
			assert((isInteger(max_iter_float) == 1));
		}
	}

	/* scanf from input used to scan the input, put each point in an array */


	while ((ch = getc(stdin)) != '\n') {
		if (ch == ',') {
			com++;
		}
	}

	d = com + 1;

	fseek(stdin, 0L, SEEK_SET);

	while ((ch = getc(stdin)) != EOF) {
		if (ch == '\n') {
			sz++;
		}
	}

	N = sz;
	fseek(stdin, 0L, SEEK_SET);



	/* check conditions */

	if (N < 1) {
		printf("Error: No points in file!\n");
		assert(N >= 1);
	}

	if (d < 1) {
		printf("Error: Dimensions should be bigger than 1!\n");
		assert(d >= 1);
	}

	if (K >= N) {
		printf("Error: Number of clusters cannot be larger or equal to the number of points!\n");
		assert(K < N);
	}

	if (max_iter < 1) {
		printf("Error: Max iterations argument is not as expected!\n");
		assert(max_iter >= 1);
	}

	if (K < 1) {
		printf("Error: K argument is not as expected!\n");
		assert(K >= 1);
	}


	/* dynamic memory allocate allocate 2D */

	points = (double**)malloc(N * sizeof(double*));
	assert(points != NULL); /*make sure points is empty*/
	for (i = 0; i < N; i++)
	{
		points[i] = (double*)malloc(d * sizeof(double));
	}
	centroids = (double**)malloc(K * sizeof(double*));
	assert(centroids != NULL); /*make sure centroids is empty*/
	for (i = 0; i < K; i++)
	{
		centroids[i] = (double*)malloc(d * sizeof(double));
	}

	i = 0;
	j = 0;

	while (!feof(stdin)) {
		count = 0;
		num = 0;
		minus = 0;
		if (i == sz) {
			break;
		}
		while (ch != '.') {
			ch = getc(stdin);
			if (ch == '.') {
				break;
			}
			if (ch == '-') {
				minus = 1;
				continue;
			}
			n1 = ch - '0';
			num = num * 10 + n1;
		}
		ch = getc(stdin);
		while (ch != ',' && ch != '\n' && !feof(stdin)) {
			count++;
			n1 = ch - '0';
			num = num + n1 * power(0.1, count);
			ch = getc(stdin);

		}
		if (minus == 1) {
			num = num * (-1);
		}

		num = round(num * 10000) / 10000;
		points[i][j] = num; /* put coordinate of point*/


		if (i < K) {
			centroids[i][j] = num; /* put coordinate of point in centroids*/
		}

		/* update row and column to continue iterating*/
		if (j == (d - 1)) {
			if (ch == '\n') {
				++i;
				j = 0;
			}
			else if (ch != '\n') {  /* if dimensions bigger */
				printf("Error: Dimensions of points not equal!\n");
				assert(ch == '\n');
			}
		}
		else {
			if (ch != '\n') {
				++j;
			}
			else if (ch == '\n') {
				printf("Error: Dimensions of points not equal!\n");
				assert(ch != '\n');
			}
		}
	}

	/* k means algorithm*/
	kmeans(points, centroids, N, K, d, max_iter);


	i = 0;

	/* free*/
	for (i = 0; i < N; i++) {
		free(points[i]);
		points[i] = NULL;
	}
	free(points);
	points = NULL;

	for (i = 0; i < K; i++) {
		free(centroids[i]);
		centroids[i] = NULL;
	}
	free(centroids);
	centroids = NULL;

	return 0;

}