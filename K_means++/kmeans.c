#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#define _CRT_SECURE_NO_WARNINGS

/*
* Helper functions prototypes
*/

static PyObject* kmeans_capi(PyObject* self, PyObject* args);
static PyObject* fit(int K, int max_iter, int d, int N, PyArrayObject* points, PyArrayObject* centroids);
double** kmeans(double**, double**, int N, int K, int d, int max_iter);
double norm(double*, double*, int d);
int findMinPoint(double*, double**, int K, int d);
void copy(double**, double**, int K, int d);
void updateCentroids(double** sumOfPoints, int* indices, double** points, double** centroids, int N, int K, int d);
void assignPointsToCluster(int*, double**, double**, int, int, int);
int checkConvergence(double**, double**, int K, int d);

/*
 * Python - C API functions 
 * Implementation of functions to allocate memory and pass arguments from
 * Python to C, as defined in the hierarchy  
 */

// fit function to run the K means algorithm - pass in py objects and cast to C types
static PyObject* fit(int K, int max_iter, int d, int N, PyArrayObject* points, PyArrayObject* centroids)
{
	int i, j;
	double** pointsC;
	double** centroidsC;
	double** res;
	npy_intp stride0, stride1;
	const char* dataptr;
	PyObject* finalCent;
	PyObject* item;
	Py_ssize_t len;

	// MEMORY ALLOCATION for points for C array
	pointsC = (double**)malloc(N * sizeof(double*));
	assert(pointsC != NULL); /*make sure points is empty*/
	for (i = 0; i < N; i++)
	{
		pointsC[i] = (double*)malloc(d * sizeof(double));
	}
	centroidsC = (double**)malloc(K * sizeof(double*));

	// MEMORY ALLOCATION for centroids for C array
	assert(centroidsC != NULL); /*make sure centroids is empty*/
	for (i = 0; i < K; i++)
	{
		centroidsC[i] = (double*)malloc(d * sizeof(double));
	}

	// PASS IN PYTHON OBJECTS AS C ARRAY for points
	stride0 = PyArray_STRIDE(points, 0);
	stride1 = PyArray_STRIDE(points, 1);
	dataptr = PyArray_BYTES(points);
	for (i = 0; i < N; i++) {
		for (j = 0; j < d; j++) {
			item = PyArray_GETITEM(points, dataptr + i * stride0 + j * stride1);
			if (!PyFloat_Check(item))
				pointsC[i][j] = 0.0;
			pointsC[i][j] = PyFloat_AsDouble(item);
		}
	}

	// PASS IN PYTHON OBJECTS AS C ARRAY for centroids
	stride0 = PyArray_STRIDE(centroids, 0);
	stride1 = PyArray_STRIDE(centroids, 1);
	dataptr = PyArray_BYTES(centroids);
	for (i = 0; i < K; i++) {
		for (j = 0; j < d; j++) {
			item = PyArray_GETITEM(centroids, dataptr + i * stride0 + j * stride1);
			if (!PyFloat_Check(item))
				centroidsC[i][j] = 0.0;
			centroidsC[i][j] = PyFloat_AsDouble(item);
		}
	}

	// Run the K means algorithm using the C implementation
	res = kmeans(pointsC, centroidsC, N, K, d, max_iter);

	// RECREATE THE PYTHON OBJECTS FROM C ARRAYS for the updates centroids
	// derived using the K means algorithm
	len = K;

	// PASS IN C ARRAYS AS PYTHON OBJECTS for updated centroids
	finalCent = PyList_New(len);
	for (i = 0; i < K; i++) {
		len = d;
		item = PyList_New(len);
		for (j = 0; j < d; j++) {
			PyList_SetItem(item, j, PyFloat_FromDouble(res[i][j]));			
		}
		PyList_SetItem(finalCent, i, item);
	}

	// FREE MEMORY
	for (i = 0; i < N; i++) {
		free(pointsC[i]);
		pointsC[i] = NULL;
	}
	free(pointsC);
	pointsC = NULL;
	for (i = 0; i < K; i++) {
		free(centroidsC[i]);
		centroidsC[i] = NULL;
	}
	free(centroidsC);
	centroidsC = NULL;


	return finalCent;
}

// gate function to the C - Python API 
// reads arguments from python and runs them in the fit function
static PyObject* kmeans_capi(PyObject* self, PyObject* args)
{
	int K = 0;
	int max_iter;
	int N, d;
	PyArrayObject* points;   /* initialize double array of points*/
	PyArrayObject* centroids; /* initialize double array of centroids*/

	if (!PyArg_ParseTuple(args, "OOiiii", &points, &centroids, &N, &K, &d, &max_iter))
		return NULL;
	return fit(K, max_iter, d, N, points, centroids);
}


static PyMethodDef capiMethods[] = {
	{"kmeans_pp",  (PyCFunction)kmeans_capi, METH_VARARGS, PyDoc_STR("divides a list of points to a given number of centroids by distance")},
	{NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef moduledef = {
	PyModuleDef_HEAD_INIT,
	"mykmeanssp",   /* name of module */
	NULL, /* module documentation */
	-1,       /* size of per-interpreter state of the module,
				 or -1 if the module keeps state in global variables. */
	capiMethods
};


PyMODINIT_FUNC PyInit_mykmeanssp(void)
{
	PyObject* m;
	m = PyModule_Create(&moduledef);
	if (!m) {
		return NULL;
	}
	return m;
}

/*
 * Helper functions implementations
 */

/* check if convergence condition is true, to break from while loop in kmeans implementation */
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
	double minDist = 1000;  /*CHECK MIN DIST VALUE*/
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

/*
* K means algorithm main implementation, using Helper functions
* accepts C arrays and types
*/
double** kmeans(double** points, double** centroids, int N, int K, int d, int max_iter) {

	int k = 0, i = 0;
	int* indices;
	double** centCopy;
	double** sumOfPoints;

	// MEMORY ALLOCATION
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

	// Loop until convergence or up to maximum iteration steps
	// For every iteration, use the helper functions to assign points to the centroids
	// and updating the centroids
	while (i < max_iter && checkConvergence(centroids, centCopy, K, d)) {
		copy(centroids, centCopy, K, d);
		assignPointsToCluster(indices, points, centroids, N, K, d);
		updateCentroids(sumOfPoints, indices, points, centroids, N, K, d);
		i++;
	}

	// FREE MEMORY
	for (i = 0; i < K; i++) {
		free(sumOfPoints[i]);
	}
	free(sumOfPoints);
	for (i = 0; i < K; i++) {
		free(centCopy[i]);
	}
	free(centCopy);
	free(indices);

	return centroids;
}