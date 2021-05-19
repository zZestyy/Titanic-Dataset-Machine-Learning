#pragma GCC diagnostic push										// warning suppression
#pragma GCC diagnostic ignored "-Wsign-compare"

#include <iostream>
#include <chrono>
#include <fstream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>			// 'remove' function
#include <cmath>				// math functions


using namespace std;
using namespace std::chrono;

const int numOfIterations = 9000;			// num of iterations for logistic regression loop

void printVector(vector<double> vect);
void print2DVector(vector<vector<double>> vect);
vector<vector<double>> transpose(vector<vector<double>> vect);
vector<vector<double>> matMultiply(vector<vector<double>> matA, vector<vector<double>> matB);
vector<vector<double>> matConstantMultiply(double c, vector<vector<double>> matA);
vector<vector<double>> matDivide(vector<vector<double>> matA, vector<vector<double>> matB);
vector<vector<double>> matAddition(vector<vector<double>> matA, vector<vector<double>> matB);
vector<vector<double>> matConstantAddition(double c, vector<vector<double>> matA);
vector<vector<double>> matSubtract(vector<vector<double>> matA, vector<vector<double>> matB);
vector<vector<double>> matEXP(vector<vector<double>> matA);
vector<vector<double>> sigmoid(vector<vector<double>> vect);
vector<vector<double>> normalize(vector<vector<double>> matA);
double accuracy(vector<vector<double>> matA, vector<vector<double>> matB);
vector<vector<double>> confusionMatrix(vector<vector<double>> matA, vector<vector<double>> matB);

int main() {

	// load csv file
	string fileName = "titanic_project.csv";		// file name
	ifstream inputFile;								// ifstream obj
	
	inputFile.open(fileName);						// open csv file
	
	// file open error checking
	if(!inputFile.is_open()) {
		cout << "File failed to open." << endl;
		return 0;
	}
	
	// declare vectors
	vector<double> x;
	vector<double> pclass;			// vector for pclass column
	vector<double> survived;		// vector for survived column
	vector<double> sex;			// vector for sex column	
	vector<double> age;			// vector for age column
	
	string header;				// string to hold headers parsing
	string cell; 				// string to hold each cell value in csv file
	
	// Retrieve & Trash Headers	
	getline(inputFile, header);
	
	// temp doubles for parsing
	double xVal;
	double pclassVal;
	double survivedVal;
	double sexVal;
	double ageVal;
	
	while(inputFile.good()) {
				
		getline(inputFile, cell, ','); 			// read x value in row into string			
		cell.erase(remove( cell.begin(), cell.end(), '\"' ),cell.end());	// remove double quotes
	
		if(!cell.empty()) {								// continue parse only on rows with values
						
			xVal = stod(cell);					// convert x value from string to integer
			x.push_back(xVal);					// append x value into x vector
							
			getline(inputFile, cell, ','); 		// read pclass value in row into string		
			pclassVal = stod(cell);				// convert pclass value from string to integer	
			pclass.push_back(pclassVal);		// append pclass value to pclass vector
				
			getline(inputFile, cell, ',');		// read survived value in row into string		
			survivedVal = stod(cell);			// convert survived value from string to integer	
			survived.push_back(survivedVal);	// append survived value to survived vector
			
			getline(inputFile, cell, ',');		// read sex value in row into string		
			sexVal = stod(cell);				// convert sex value from string to integer	
			sex.push_back(sexVal);				// append sex value to sex vector
				
			getline(inputFile, cell);			// read age value in row into string		
			ageVal = stod(cell);				// convert age value from string to integer	
			age.push_back(ageVal);				// append age value to age vector		
		}
		else {											// if row empty break loop
			break;
		}	
	}
	



/* -------------------------------------------------------------
 * ----L-O-G-I-S-T-I-C---R-E-G-R-E-S-S-I-O-N---S-C-R-A-T-C-H----
 * ------------------------------------------------------------- */
 
	vector<vector<double>> weights(2, vector<double> (1, 1)); // declare weights as a 2 x 1
	
	// create data matrix
	vector<vector<double>> data_matrix;
	for(int i = 0; i < 900; i++) {
		vector<double> temp;
		for(int j = 0; j < 2; j++) {
			if(j == 0) {				
				temp.push_back(1.0);
			}
			else {
				temp.push_back(pclass.at(i));			// pclass = predictors
			}
		}
		data_matrix.push_back(temp);
	}
			
	// create train labels
	vector<vector<double>> labels;
	for(int i = 0; i < 900; i++) {
		vector<double> temp2;
		for(int j = 0; j < 1; j++) {
			temp2.push_back(survived.at(i));
		}
		labels.push_back(temp2);
		
	}
	
	
	
	double learning_rate = 0.001;		// learning rate
	
	cout << "---Logistic Regression C++ Implementation---" << endl << endl;
	
	cout << "Iterating " << numOfIterations << " times. Please Wait." << endl << endl;

	auto start = high_resolution_clock::now();   // stopwatch start
	
	// first iteration
	vector<vector<double>> dMxW = matMultiply(data_matrix, weights);             	// data_matrix %*% weights
	vector<vector<double>> prob_vector = sigmoid(dMxW);								// sigmoid(data_matrix %*% weights)
	vector<vector<double>> error = matSubtract(labels, prob_vector);				// labels - prob_vector
	vector<vector<double>> tdata_matrix = transpose(data_matrix);					// transpose([labels - prob_vector])
	vector<vector<double>> tdmXe = matMultiply(tdata_matrix, error);				// [transpose(data_matrix)] %*% error
	vector<vector<double>> cXdmXe = matConstantMultiply(learning_rate, tdmXe);		// learning_rate * [transpose(data_matrix) %*% error]
	weights = matAddition(weights, cXdmXe);											// weights + [learning_rate * t(data_matrix) %*% error]
	
	// remaining iteration
	for(int i = 1; i < numOfIterations; i++) {
		dMxW = matMultiply(data_matrix, weights);									/* refer to first iteration comments */
		prob_vector = sigmoid(dMxW);
		error = matSubtract(labels, prob_vector);
		tdmXe = matMultiply(tdata_matrix, error);
		cXdmXe = matConstantMultiply(learning_rate, tdmXe);	
		weights = matAddition(weights, cXdmXe);
	}
	
	auto stop = high_resolution_clock::now();	 // stopwatch stop
	
	cout << "Weights" << endl;					// print out coefficients
	print2DVector(weights);
	cout << endl;
		
	std::chrono::duration<double> elapsed_sec = stop-start;		// calculate time elapsed
	cout << "Time Elapsed: " << elapsed_sec.count() << endl << endl;	// print time elapsed
	

/* ----------------------------------------------------
 * ------------------P-R-E-D-I-C-T---------------------
 * ---------------------------------------------------- */
	
	// create test_matrix
	vector<vector<double>> test_matrix;
	for(int i = 900; i < x.size(); i++) {
		vector<double> temp3;
		for(int j = 0; j < 2; j++) {
			if(j == 0) {				
				temp3.push_back(1.0);
			}
			else {
				temp3.push_back(pclass.at(i));			// pclass = predictors
			}
		}
		test_matrix.push_back(temp3);
	}		
	
	// create test_labels/test data
	vector<vector<double>> test_labels;
	for(int i = 900; i < x.size(); i++) {
		vector<double> temp4;
		for(int j = 0; j < 1; j++) {
			temp4.push_back(survived.at(i));
		}
		test_labels.push_back(temp4);
	}		
	vector<vector<double>> predicted = matMultiply(test_matrix, weights);	
	vector<vector<double>> EP = matEXP(predicted);
	vector<vector<double>> ONEplusEP = matConstantAddition(1, EP);
	vector<vector<double>> probabilities = matDivide(EP, ONEplusEP);
	vector<vector<double>> predictions = normalize(probabilities);
	
	/*		TP FP
	 * 		FN TN		*/ 
	cout << "Confusion Matrix:" << endl;
	vector<vector<double>> table = confusionMatrix(predictions, test_labels);
	print2DVector(table);
	cout << endl << endl;
	
	double acc = accuracy(predictions, test_labels);
	cout << "Accuracy: " << acc << endl;
	
	// sensitivity = TP / (TP + FN)
	double sensitivity = (table.at(0).at(0) / ( table.at(0).at(0) + table.at(1).at(0)));
	cout << "Sensitivity: " << sensitivity << endl;
	
	// specificity = TN / (TN + FP)
	double specificity = (table.at(1).at(1) / ( table.at(1).at(1) + table.at(0).at(1)));
	cout << "Specificity: " << specificity << endl << endl;
		
	return 0;
}


/* -------------------------------------------------  
 * -----------------FUNCTIONS-----------------------    
 * ------------------------------------------------- */

/* prints the given a one-dimensional vector */
void printVector(vector<double> vect) {
	for(int i = 0; i < vect.size(); i++) {
		cout << vect[i] << endl;
	}
}

/* prints the given a two-dimensional vector */
void print2DVector(vector<vector<double>> vect) {
	for(int i = 0; i < vect.size(); i++) {
		for(int j = 0; j < vect[i].size(); j++) {
			cout << vect[i][j] << " ";
		}
		cout << endl;
	}
}

/* returns the transpose of the given two-dimensional vectors */
vector<vector<double>> transpose(vector<vector<double>> vect) {
	vector<vector<double>> t(vect[0].size(), vector<double>());
	
	for(int i = 0; i < vect.size(); i++) {
		for(int j = 0; j < vect[i].size(); j++) {
			t[j].push_back(vect[i][j]);
		}
	}
	return t;
}

/* returns the matrix multiplication of two given two-dimensional vectors */
vector<vector<double>> matMultiply(vector<vector<double>> matA, vector<vector<double>> matB) {
	int matARow = matA.size();
	int matACol = matA[0].size();	
	int matBRow = matB.size();
	int matBCol = matB[0].size();
	
	if(matACol != matBRow) {
		cout << "Error. Matrix Multiplication Rule: mxn X nxp = mxp." << endl;
	}
	
	vector<vector<double>> product(matARow, vector<double>(matBCol, 0));		
	
	for(int i = 0; i < matARow; i++) {
		for(int j = 0; j < matBCol; j++) {
			for(int k = 0; k < matACol; k++) {
				product[i][j] += matA[i][k] * matB[k][j];
			}
		}
	}	
	return product;
}

/* returns the matrix multiplication of a constant and a two-dimensional vector */
vector<vector<double>> matConstantMultiply(double c, vector<vector<double>> matA) {
	int matARow = matA.size();
	int matACol = matA[0].size();	
		
	vector<vector<double>> product(matARow, vector<double>(matACol, 0));		
	
	for(int i = 0; i < matARow; i++) {
		for(int j = 0; j < matACol; j++) {
			product[i][j] = c * matA.at(i).at(j);
		}
	}
	
	return product;
}

/* returns the division of two given two-dimensional vectors */
vector<vector<double>> matDivide(vector<vector<double>> matA, vector<vector<double>> matB) {
	int matARow = matA.size();
	int matACol = matA[0].size();	
	int matBRow = matB.size();
	int matBCol = matB[0].size();
	
	vector<vector<double>> quotient(matARow, vector<double>(1, 0));
		
	if((matARow != matBRow) && (matACol != matBCol)) {
		cout << "Error. Matrix Addition Rule: Must Be Same Dimensions." << endl;
	}	
	
	for(int i = 0; i < matA.size(); i++) {
		for(int j = 0; j < matA[i].size(); j++) {
			quotient[i][j] = matA.at(i).at(j) / matB.at(i).at(j);
		}
	}
	
	return quotient;
}

/* returns the matrix addition of two given two-dimensional vectors */
vector<vector<double>> matAddition(vector<vector<double>> matA, vector<vector<double>> matB) {
	int matARow = matA.size();
	int matACol = matA[0].size();	
	int matBRow = matB.size();
	int matBCol = matB[0].size();
	
	vector<vector<double>> sum(matARow, vector<double>(1, 0));
		
	if((matARow != matBRow) && (matACol != matBCol)) {
		cout << "Error. Matrix Addition Rule: Must Be Same Dimensions." << endl;
	}
	
	for(int i = 0; i < matA.size(); i++) {
		for(int j = 0; j < matA[i].size(); j++) {
			sum[i][j] = matA.at(i).at(j) + matB.at(i).at(j);
		}
	}
	
	return sum;
}

/* returns the matrix addition of a given constant and a two-dimensional vector */
vector<vector<double>> matConstantAddition(double c, vector<vector<double>> matA) {
	int matARow = matA.size();
//	int matACol = matA[0].size();	
	
	vector<vector<double>> sum(matARow, vector<double>(1, 0));
			
	for(int i = 0; i < matA.size(); i++) {
		for(int j = 0; j < matA[i].size(); j++) {
			sum[i][j] = c + matA.at(i).at(j);
		}
	}
	
	return sum;
}

/* returns the matrix subtraction of two given two-dimensional vectors */
vector<vector<double>> matSubtract(vector<vector<double>> matA, vector<vector<double>> matB) {
	int matARow = matA.size();
	int matACol = matA[0].size();	
	int matBRow = matB.size();
	int matBCol = matB[0].size();
	
	vector<vector<double>> difference(matARow, vector<double>(1, 0));
		
	if((matARow != matBRow) && (matACol != matBCol)) {
		cout << "Error. Matrix Subtraction Rule: Must Be Same Dimensions." << endl;
	}
	
	for(int i = 0; i < matA.size(); i++) {
		for(int j = 0; j < matA[i].size(); j++) {
			difference[i][j] = matA.at(i).at(j) - matB.at(i).at(j);
		}
	}
	
	return difference;
}

/* returns the a matrix where the values of the given two-dimensional vector are e^x */
vector<vector<double>> matEXP(vector<vector<double>> matA) {
	int matARow = matA.size();
	int matACol = matA[0].size();	
		
	vector<vector<double>> x(matARow, vector<double>(matACol, 0));
		
	for(int i = 0; i < matARow; i++) {
		for(int j = 0; j < matACol; j++) {
			x[i][j] = exp(matA.at(i).at(j));
		}
	}
	return x;
}

/* returns the a matrix where the values of the given two-dimensional vector are "sigmoid-ed"/log functoned */
vector<vector<double>> sigmoid(vector<vector<double>> vect) {
	vector<vector<double>> sigmoidVect(vect.size(), vector<double>(1, 1));
			
	for(int i = 0; i < vect.size(); i++) {	
		for(int j = 0; j < vect[i].size(); j++) {	
			sigmoidVect[i][j] = ( 1.0 / (1.0 + exp(-(vect.at(i).at(j))) ) );
		}
	}
	return sigmoidVect;
}

/* returns the a matrix where the values of the given two-dimensional vector are normalized to 0 and 1 with 0.5 as cutoff*/
vector<vector<double>> normalize(vector<vector<double>> matA) {
	int matARow = matA.size();
//hb	int matACol = matA[0].size();	
	
	vector<vector<double>> norm(matARow, vector<double>(1, 0));
			
	for(int i = 0; i < matA.size(); i++) {
		for(int j = 0; j < matA[i].size(); j++) {
			if(matA.at(i).at(j) > 0.5) {
				norm[i][j] = 1;
			}
			else {
				norm[i][j] = 0;
			}
		}
	}
	
	return norm;
}

/* returns the accuracy in comparing two given two-dimensional matrices */
double accuracy(vector<vector<double>> matA, vector<vector<double>> matB) {
	int matARow = matA.size();
	int matACol = matA[0].size();	
	int matBRow = matB.size();
	int matBCol = matB[0].size();	
	
	if((matARow != matBRow) && (matACol != matBCol)) {
		cout << "Error in Accuracy. Must Be Same Dimensions." << endl;
	}
		
	double sum = 0;
	
	for(int i = 0; i < matA.size(); i++) {
		for(int j = 0; j < matA[i].size(); j++) {
			if(matA.at(i).at(j) == matB.at(i).at(j)) {
				sum++;
			}
			else {
				
			}
		}
	}
	
	return sum / matA.size();
}

/* returns the TP, FP, FN, TN accuracy values in comparing two given two-dimensional matrices */
vector<vector<double>> confusionMatrix(vector<vector<double>> matA, vector<vector<double>> matB) {
	vector<vector<double>> table(2, vector<double>(2, 0));
	
	// matA = predicted, matB = test$survived
	
	/*		TP FP
	 * 		FN TN		*/ 
	
	for(int i = 0; i < matA.size(); i++) {
		for(int j = 0; j < matA[i].size(); j++) {
			if( matA.at(i).at(j) == 0 && matB.at(i).at(j) == 0 ) {			// true negative
				table.at(0).at(0)++;
			}
			else if( matA.at(i).at(j) == 1 && matB.at(i).at(j) == 1 ) {		// true positive
				table.at(1).at(1)++;
			}
			else if( matA.at(i).at(j) == 1 && matB.at(i).at(j) == 0 ) {		// false positive
				table.at(1).at(0)++;
			}
			else if( matA.at(i).at(j) == 0 && matB.at(i).at(j) == 1 ) {		// false negative
				table.at(0).at(1)++;
			}
			else {}
		}
	}	
	return table;
}

#pragma GCC diagnostic pop
