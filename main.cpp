#include <iostream>  
#include <fstream>
#include <sstream>
#include <string>
#include <Eigen/Dense>  
using namespace Eigen;  
using namespace std;

double logisticfun(double x)
{
	return 1.0/(1+exp(-x));
}

VectorXd logistic(MatrixXd X, VectorXd theta,int M)
{
	VectorXd tmp = X * theta;
	VectorXd result(M);
	/*
	for (int i=0;i<M;i++)
	{
		result(i)=1.0/(1+exp(-tmp(i)));
	}*/
	result = tmp.unaryExpr(&logisticfun);
	return result;//size M*1
}

void irls(MatrixXd x, MatrixXd y, VectorXd &theta, int N,int M)
{
	VectorXd theta_(N);
	VectorXd diff = theta - theta_;
	MatrixXd hessian;
	VectorXd first_derivative;
	MatrixXd diagonalize;
	VectorXd ones = VectorXd::Ones(M);
	int i=0;
	while (diff.maxCoeff()>1*exp(-6))
	{
		first_derivative = x.transpose() * (y - logistic(x,theta,M));
		diagonalize =(logistic(x,theta,M)* (ones-logistic(x,theta,M)).transpose()).diagonal().asDiagonal();
		hessian = x.transpose() * diagonalize * x;
		cout << "diag: " << endl;
		cout<< y-logistic(x,theta,M)<<endl;
		cout <<  first_derivative << endl;
		return;
		cout << "hessian: " << endl;
		cout << hessian << endl;
		theta_ = theta;
		theta = theta + hessian.inverse() * first_derivative;
		diff = theta - theta_;
		cout<<"Iterative Number:"<<i++<<endl<<"theta:"<<theta<<endl;
		cout<<hessian.inverse()<<endl;
	}
	//cout<<theta<<endl;
}

void test(MatrixXd x,MatrixXd y,VectorXd &theta, int N,int M)
{
	VectorXd predict(M);
	predict = logistic(x,theta,M);
	int j=0;
	for (int i=0;i<M;i++)
	{
		predict(i) = predict(i)>0.5?1:0;
		if (predict(i)!=y(i))
		{
			j++;
		}
	}
	cout<<"error ratio:"<<j*1.0/100<<endl;
}


int main()  
{
	int M,N,x,y;
	M = 100; 
	N = 3;
	MatrixXd dataMatrix(M,N);
	VectorXd label(M);  
	VectorXd theta = VectorXd::Zero(N);

	string line;
	ifstream myfile("inputTrainingSet1.txt");	
	
	x=0;
	while (getline(myfile,line))
	{
		string token;
		istringstream ss(line);
		y=1;
		dataMatrix(x,0)=1;
		while(getline(ss, token, ',')) {
			//std::cout << token << '\n';
			if (y==3)
			{
				label(x)=stod(token);
				//cout<<"target"<<x<<label(x)<<endl;
			}
			else
			{
				dataMatrix(x,y)=stod(token);
				//cout<<dataMatrix(x,y)<<",";
			}
			y++;
		}
		x++;
	}
	
	irls(dataMatrix,label,theta,N,M);
	test(dataMatrix,label,theta,N,M);
	getchar();
	return 0;
}