#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream>
using namespace std;
typedef double DATATYPE;
#define BUFFSIZE 100

DATATYPE sigmoid(DATATYPE x);
DATATYPE vecMultiply(DATATYPE* x,DATATYPE* y, int len);
void diagonal(DATATYPE** featureMatrix, DATATYPE* theta, int M,int N, DATATYPE* mydiagonal);
void hessian(DATATYPE** featureMatrix,DATATYPE* theta, int M, int N,DATATYPE* mydiagonal, DATATYPE** featureMatrixTranspose,DATATYPE** myhessian);
void firstDerivative(DATATYPE** featureMatrix,DATATYPE* theta, DATATYPE* labelData, int M, int N,DATATYPE** featureMatrixTranspose,DATATYPE* myfirstDerivative);
//inverse function
DATATYPE detrminant(DATATYPE** x, DATATYPE y);
void cofactors(DATATYPE** num, DATATYPE f,DATATYPE** inv);
void trans(DATATYPE** num, DATATYPE** fac, DATATYPE r, DATATYPE** inv);

void irls(DATATYPE** featureMatrix, DATATYPE* labelData, DATATYPE* theta, int N,int M,DATATYPE* mydiagonal, DATATYPE** featureMatrixTranspose,DATATYPE** myhessian,DATATYPE* myfirstDerivative);
DATATYPE getMaxCoeff(DATATYPE* diff,int len);
void initializeMatrix(DATATYPE** init,int M, int N);

int main(int argc, char* argv[]) {
	
	const int record_num = 100;
	const int dim_num = 2 + 1;//2 features, 1 x_0;
	DATATYPE* labelData;
	DATATYPE* mydiagonal;
	DATATYPE* myfirstDerivative;
	DATATYPE** featureMatrix;
	DATATYPE** featureMatrixTranspose;
	DATATYPE** myhessian;

	int i,j;
	FILE* myFile;
	char line[BUFFSIZE];
	char* token;
	DATATYPE* weights;

	labelData = (DATATYPE*)malloc(record_num*sizeof(DATATYPE));
	mydiagonal = (DATATYPE*)malloc(record_num*sizeof(DATATYPE));
	myfirstDerivative = (DATATYPE*)malloc(dim_num*sizeof(DATATYPE));

	weights = (DATATYPE*)malloc(dim_num*sizeof(DATATYPE));
	featureMatrix = (DATATYPE**)malloc(record_num*sizeof(DATATYPE*));
	for (i = 0; i < record_num; i++)
	{
		featureMatrix[i] = (DATATYPE*)malloc(dim_num*sizeof(DATATYPE));
	}
	featureMatrixTranspose = (DATATYPE**)malloc(dim_num*sizeof(DATATYPE*));
	for (i = 0; i < record_num; i++)
	{
		featureMatrixTranspose[i] = (DATATYPE*)malloc(record_num*sizeof(DATATYPE));
	}
	myhessian = (DATATYPE**)malloc(dim_num*sizeof(DATATYPE*));
	for (i = 0; i < dim_num; i++)
	{
		myhessian[i] = (DATATYPE*)malloc(dim_num*sizeof(DATATYPE));
	}

	myFile=fopen("inputTrainingSet1.txt", "r");
	if (myFile == NULL)
	{
		fprintf(stderr,"Error opening file.\n");
		exit(2);
	}
	for (i = 0; i < record_num;i++)
	{
		featureMatrix[i][0]=1;
	}
	//loading data
	i=0;
	while (fgets(line,BUFFSIZE,myFile)) {
		j=1;
		token = strtok (line,",");  
		while (token != NULL)
		{
			if (j==dim_num)
			{
				labelData[i] = atoi(token);
			}
			else
			{
				featureMatrix[i][j] = atof(token);
			}
			token = strtok (NULL, ",");
			j++;
		}
		i++;
	}
	fclose(myFile);

	// lr_method
	for (i=0;i<dim_num;i++)
	{
		weights[i]=0;
		myfirstDerivative[i]=0;
	}
	for (i=0;i<record_num;i++)
	{
		mydiagonal[i]=0;
	}
	initializeMatrix(featureMatrixTranspose,dim_num,record_num);
	initializeMatrix(myhessian,dim_num,dim_num);
	irls(featureMatrix,labelData,weights,record_num,dim_num,mydiagonal,featureMatrixTranspose,myhessian,myfirstDerivative);
	
	for (i=0;i<dim_num;i++)
	{
		cout<<weights[i]<<endl;
	}

	getchar();
	return 0;
}

void initializeMatrix(DATATYPE** init,int M, int N)
{
	int i=0;
	int j=0;
	for (i=0;i<M;i++)
	{
		for (j=0;j<N;j++)
		{
			init[i][j]=0;
		}
	}
}

DATATYPE sigmoid(DATATYPE x) {
	return 1.0 / (1.0 + exp(-x));
}

DATATYPE vecMultiply(DATATYPE* x,DATATYPE* y, int len)
{
	int i=0;
	DATATYPE sum=0;
	for (i=0; i<len; i++)
	{
		sum = sum+ x[i]*y[i];
	}
	return sum;
}

void diagonal(DATATYPE** featureMatrix, DATATYPE* theta, int M,int N,DATATYPE* mydiagonal)
{
	int i=0;
	//DATATYPE* diagonal = (DATATYPE*)malloc(M*sizeof(DATATYPE));
	for (i=0;i<M;i++)
	{
		mydiagonal[i]=sigmoid(vecMultiply(featureMatrix[i],theta,N))*(1-sigmoid(vecMultiply(featureMatrix[i],theta,N)));
	}
}

// M is the record number, N is the feature dimensions
void hessian(DATATYPE** featureMatrix,DATATYPE* theta, int M, int N, DATATYPE* mydiagonal, DATATYPE** featureMatrixTranspose,DATATYPE** myhessian)
{
	int i=0;
	int j=0;
	int k=0;
	DATATYPE sum=0;
	DATATYPE** WX;
	
	diagonal(featureMatrix,theta,M,N,mydiagonal);

	WX = (DATATYPE**)malloc(M*sizeof(DATATYPE*));
	for (i = 0; i < M; i++)
	{
		WX[i] = (DATATYPE*)malloc(N*sizeof(DATATYPE));
	}

	//computing W*X 
	initializeMatrix(WX,M,N);
	for (i=0;i<M;i++)
	{
		for (j=0;j<N;j++)
		{
			WX[i][j] = mydiagonal[i]*featureMatrix[i][j];
		}
	}
	//now computing X^T*WX
	initializeMatrix(featureMatrixTranspose,N,M);
	for (i=0;i<M;i++)
	{
		for (j=0;j<N;j++)
		{
			featureMatrixTranspose[j][i] = featureMatrix[i][j];
		}
	}
	
	for (i=0;i<N;i++)
	{
		for (j=0;j<N;j++)
		{
			for (k=0;k<M;k++)
			{
				sum = sum + featureMatrixTranspose[i][k]*WX[k][j];
			}
			myhessian[i][j]  = sum;
			sum = 0;
		}
	}
}

void firstDerivative(DATATYPE** featureMatrix,DATATYPE* theta, DATATYPE* labelData, int M, int N,DATATYPE** featureMatrixTranspose,DATATYPE* myfirstDerivative)
{
	int i=0;
	int j=0;
	DATATYPE sum=0;
	DATATYPE* diff;
	diff = (DATATYPE*)malloc(M*sizeof(DATATYPE));

	//computing diff of lableData-predictiveData
	for (i=0;i<M;i++)
	{
		diff[i] = labelData[i] - sigmoid(vecMultiply(featureMatrix[i],theta,N));
	}

	for (i=0;i<N;i++)
	{
		for (j=0;j<M;j++)
		{
			sum = sum + featureMatrixTranspose[i][j]*diff[j];
		}
		myfirstDerivative[i]=sum;
		sum=0;
	}
}


DATATYPE detrminant(DATATYPE** a, DATATYPE k)
{
	DATATYPE s = 1, det = 0;
	DATATYPE** b;
	int i, j, m, n, c;
	b = (DATATYPE**)malloc(k*sizeof(DATATYPE*));
	for (i = 0; i < k; i++)
	{
		b[i] = (DATATYPE*)malloc(k*sizeof(DATATYPE));
	}
	if (k == 1) {
		return (a[0][0]);
	} else {
		det = 0;
		for (c = 0; c < k; c++) {
			m = 0;
			n = 0;
			for (i = 0; i < k; i++) {
				for (j = 0; j < k; j++) {
					b[i][j] = 0;
					if (i != 0 && j != c) {
						b[m][n] = a[i][j];
						if (n < (k - 2))
							n++;
						else {
							n = 0;
							m++;
						}
					}
				}
			}
			det = det + s * (a[0][c] * detrminant(b, k - 1));
			s = -1 * s;
		}
	}
	return (det);
}
void cofactors(DATATYPE** num, DATATYPE f,DATATYPE** inv) {

	DATATYPE** b;
	DATATYPE** fac;
	int p, q, m, n, i, j;
	b = (DATATYPE**)malloc(f*sizeof(DATATYPE*));
	for (i = 0; i < f; i++)
	{
		b[i] = (DATATYPE*)malloc(f*sizeof(DATATYPE));
	}
	fac = (DATATYPE**)malloc(f*sizeof(DATATYPE*));
	for (i = 0; i < f; i++)
	{
		fac[i] = (DATATYPE*)malloc(f*sizeof(DATATYPE));
	}

	for (q = 0; q < f; q++) {
		for (p = 0; p < f; p++) {
			m = 0;
			n = 0;
			for (i = 0; i < f; i++) {
				for (j = 0; j < f; j++) {
					b[i][j] = 0;
					if (i != q && j != p) {
						b[m][n] = num[i][j];
						if (n < (f - 2))
							n++;
						else {
							n = 0;
							m++;
						}
					}
				}
			}
			fac[q][p] = pow(-1, q + p) * detrminant(b, f - 1);
		}
	}
	trans(num, fac, f,inv);
}
void trans(DATATYPE** num, DATATYPE** fac, DATATYPE r, DATATYPE** inv)
{
	int i, j;
	DATATYPE** b;
	DATATYPE d;

	b = (DATATYPE**)malloc(r*sizeof(DATATYPE*));
	for (i = 0; i < r; i++)
	{
		b[i] = (DATATYPE*)malloc(r*sizeof(DATATYPE));
	}

	for (i = 0; i < r; i++) {
		for (j = 0; j < r; j++) {
			b[i][j] = fac[j][i];
		}
	}

	d = detrminant(num, r);
	cout<<"dert"<<d<<endl;
	for (i = 0; i < r; i++) {
		for (j = 0; j < r; j++) {
			inv[i][j] = b[i][j] / d;
		}
	}
}

DATATYPE getMaxCoeff(DATATYPE* diff,int len)
{
	DATATYPE max_Coff = 0;
	int i=0;
	for (i=0;i<len;i++)
	{
		if (max_Coff<diff[i])
		{
			max_Coff = diff[i];
		}
	}
	return max_Coff;
}

void vecSubtract(DATATYPE* a, DATATYPE* b, DATATYPE* result, int len)
{
	int i;
	for (i=0;i<len;i++)
	{
		result[i] = fabs(a[i]-b[i]);
	}
}

void irls(DATATYPE** featureMatrix, DATATYPE* labelData, DATATYPE* theta, int M,int N,DATATYPE* mydiagonal, DATATYPE** featureMatrixTranspose,DATATYPE** myhessian,DATATYPE* myfirstDerivative)
{
	DATATYPE** inv;
	DATATYPE* theta_;
	DATATYPE* diff;
	DATATYPE max_Coeff;
	DATATYPE det;
	int i=0;
	int j=0;
	DATATYPE sum=0;

	inv = (DATATYPE**)malloc(N*sizeof(DATATYPE*));
	for (i = 0; i < N; i++)
	{
		inv[i] = (DATATYPE*)malloc(N*sizeof(DATATYPE));
	}

	theta_ = (DATATYPE*)malloc(N*sizeof(DATATYPE));
	for (i = 0; i < N; i++)
	{
		theta_[i] = -100;
	}
	
	diff = (DATATYPE*)malloc(N*sizeof(DATATYPE));
	vecSubtract(theta,theta_,diff,N);
	max_Coeff = getMaxCoeff(diff,N);
	
	while (max_Coeff>1*exp(-6))
	{
		for (i=0;i<N;i++)
		{
			theta_[i]=theta[i];
		}
		hessian(featureMatrix,theta,M,N,mydiagonal,featureMatrixTranspose,myhessian);
		firstDerivative(featureMatrix,theta,labelData,M,N,featureMatrixTranspose,myfirstDerivative);

		det = detrminant(myhessian,N);
		cofactors(myhessian,N,inv);
		
		for (i=0;i<N;i++)
		{
			for (j=0;j<N;j++)
			{
				sum = sum + inv[i][j]*myfirstDerivative[j];
			}
			theta[i]=theta[i]+sum;
			sum=0;
		}

		vecSubtract(theta,theta_,diff,N);
		max_Coeff = getMaxCoeff(diff,N);

	}
}