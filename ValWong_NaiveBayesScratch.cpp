#pragma GCC diagnostic push												// warning suppression
#pragma GCC diagnostic ignored "-Wsign-compare"

#include <iostream>
#include <chrono>
#include <fstream>				// filestream
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>			// 'remove' function
#include <cmath>				// math functions
#include <math.h>				// math functions


using namespace std;
using namespace std::chrono;

const int startTest = 900;				// start index of test data
const int numOfIterations = 5;			// num of raw predictions to be shown

void printVector(vector<double> vect);
void print2DVector(vector<vector<double>> vect);
vector<vector<double>> priorProb(vector<double> vect);
vector<vector<double>> countSurvived(vector<double> vect);
vector<vector<double>> likelihoodPClass (vector<double> survived, vector<double> pclass, vector<vector<double>> count_survived);
vector<vector<double>> likelihoodSex (vector<double> survived, vector<double> sex, vector<vector<double>> count_survived);
double calcMean(vector<double> vect);
double calcVariance(vector<double> vect);
vector<vector<double>> ageMean (vector<double> survived, vector<double> age, vector<vector<double>> count_survived);
vector<vector<double>> ageVar (vector<double> survived, vector<double> age, vector<vector<double>> count_survived);
vector<vector<double>> age_metrics (vector<vector<double>> ageMean, vector<vector<double>> ageVar);
double calc_age_lh (double v, double mean_v, double var_v);
vector<vector<double>> calc_raw_prob(double pclass, double sex, double age, vector<vector<double>> apriori, vector<vector<double>> lh_pclass, vector<vector<double>> lh_sex, vector<vector<double>> age_mean, vector<vector<double>> age_var);
vector<vector<double>> confusionMatrix(vector<double> matA, vector<double> matB);
double accuracy(vector<double> matA, vector<double> matB);

int main() {
	
	// load csv file
	string fileName = "titanic_project.csv";		// file name
	ifstream inputFile;					// ifstream obj
	
	inputFile.open(fileName);	// open csv file
	
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
	
	cout << "---Naive Bayes C++ Implementation---" << endl << endl;
	
	// partition train data
	vector<double> pclasstrain_data;
	for(int i = 0; i < startTest; i++) {
		pclasstrain_data.push_back(pclass.at(i));
	}
	vector<double> survivedtrain_data;
	for(int i = 0; i < startTest; i++) {
		survivedtrain_data.push_back(survived.at(i));
	}
	vector<double> sextrain_data;
	for(int i = 0; i < startTest; i++) {
		sextrain_data.push_back(sex.at(i));
	}	
	vector<double> agetrain_data;
	for(int i = 0; i < startTest; i++) {
		agetrain_data.push_back(age.at(i));
	}
	
	auto start = high_resolution_clock::now();   // stopwatch start
	
	// prior probabilities
	vector<vector<double>> apriori = priorProb(survivedtrain_data);					// 1x2 matrix
	cout << "A-priori Probabilities: " << endl;
	print2DVector(apriori);
	cout << endl;
	
	// count survived
	vector<vector<double>> count_survived = countSurvived(survivedtrain_data);		// 1x2 matrix
	
	cout << "Conditional Probabilities:" << endl;
	
	// likelihood for pclass
	vector<vector<double>> lh_pclass = likelihoodPClass(survivedtrain_data, pclasstrain_data, count_survived); // 2x3
	cout << "\tpClass " << endl;
	print2DVector(lh_pclass);
	cout << endl;
	
	// likelihood for sex
	vector<vector<double>> lh_sex = likelihoodSex(survivedtrain_data, sextrain_data, count_survived); // 2x2
	cout << "\tsex: " << endl;
	print2DVector(lh_sex);
	cout << endl;
	
	// age mean and variance
	vector<vector<double>> age_mean = ageMean(survivedtrain_data, agetrain_data, count_survived);	// 1x2
	vector<vector<double>> age_var = ageVar(survivedtrain_data, agetrain_data, count_survived);     // 1x2

	// age metrics mean and std deviation
	cout << "\tage: " << endl;
	vector<vector<double>> ageMetrics = age_metrics(age_mean, age_var);
	print2DVector(ageMetrics);
	cout << endl << endl;
	
	auto stop = high_resolution_clock::now();	 // stopwatch stop
	
	// age mean
	cout << "Age Mean: " << endl;
	print2DVector(age_mean);
	cout << endl;
	
	// age variance
	cout << "Age Variance: " << endl;
	print2DVector(age_var);
	cout << endl << endl;
	
	vector<vector<double>> raw(1, vector<double> (2, 0)); // declare as a 1 x 2
		
	cout << "Predicted Raw Probabilities on Test:" << endl;
	
	// print out first 5 predicted raw probabilities
	for(int i = startTest; i < (startTest + numOfIterations); i++) {		
		raw = calc_raw_prob(pclass.at(i), sex.at(i), age.at(i), apriori, lh_pclass, lh_sex, age_mean, age_var);   // 1x2 vector
		print2DVector(raw);
	}
	cout << endl << endl;
	

		
	std::chrono::duration<double> elapsed_sec = stop-start;		// calculate time elapsed
	cout << "Time Elapsed: " << elapsed_sec.count() << endl << endl;	// print time elapsed
	
	// partition test data
	vector<double> pclasstest_data;
	for(int i = startTest; i < x.size(); i++) {
		pclasstest_data.push_back(pclass.at(i));
	}
	vector<double> survivedtest_data;
	for(int i = startTest; i < x.size(); i++) {
		survivedtest_data.push_back(survived.at(i));
	}
	vector<double> sextest_data;
	for(int i = startTest; i < x.size(); i++) {
		sextest_data.push_back(sex.at(i));
	}	
	vector<double> agetest_data;
	for(int i = startTest; i < x.size(); i++) {
		agetest_data.push_back(age.at(i));
	}
	
	// normalize raw probabilities
	vector<double> p1(146); // declare as a 146 x 1 
	for(int i = 0; i < pclasstest_data.size(); i++) {
		raw = calc_raw_prob(pclasstest_data.at(i), sextest_data.at(i), agetest_data.at(i), apriori, lh_pclass, lh_sex, age_mean, age_var);   // 1x2 vector
		if((raw.at(0).at(0)) > 0.5 ) {
			p1.at(i) = 0;
		}
		else if((raw.at(0).at(1)) > 0.5) {
			p1.at(i) = 1;
		}
		else {}
	}
	
	// confusion matrix
	cout << "Confusion Matrix: " << endl;
	vector<vector<double>> table = confusionMatrix(p1, survivedtest_data);
	print2DVector(table); 
	cout << endl;
	
	double acc = accuracy(p1, survivedtest_data);
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

// print one-dimensional vector 
void printVector(vector<double> vect) {
	for(int i = 0; i < vect.size(); i++) {
		cout << vect[i] << endl;
	}
}

// print two-dimensional vector
void print2DVector(vector<vector<double>> vect) {
	for(int i = 0; i < vect.size(); i++) {
		for(int j = 0; j < vect[i].size(); j++) {
			cout << vect[i][j] << " ";
		}
		cout << endl;
	}
}

// calculates prior probabilities of train data
vector<vector<double>> priorProb(vector<double> vect) {
	vector<vector<double>> prior(1, vector<double> (2, 0)); // declare as a 1 x 2
	
	for(int i = 0; i < vect.size(); i++) {
		if(vect.at(i) == 0) {
			prior.at(0).at(0)++;
		}
		else {
			prior.at(0).at(1)++;
		}
	}

	prior.at(0).at(0) = prior.at(0).at(0) / vect.size();
	prior.at(0).at(1) = prior.at(0).at(1) / vect.size();
		
	return prior;
}

// calculates count survived of train data
vector<vector<double>> countSurvived(vector<double> vect) {
	vector<vector<double>> prior(1, vector<double> (2, 0)); // declare as a 1 x 2
	
	for(int i = 0; i < vect.size(); i++) {
		if(vect.at(i) == 0) {
			prior.at(0).at(0)++;
		}
		else {
			prior.at(0).at(1)++;
		}
	}
	return prior;
}

// calculates pclass survived likelihood of train data
vector<vector<double>> likelihoodPClass (vector<double> survived, vector<double> pclass, vector<vector<double>> count_survived) {
	
	vector<vector<double>> lh_pclass (2, vector<double>(3,0)); 				// declare as a 2 x 3 matrix.
	
	for(int i = 0; i < survived.size(); i++) {
		if(survived.at(i) == 0) {
			if(pclass.at(i) == 1) {
				lh_pclass.at(0).at(0)++;
			}
			else if(pclass.at(i) == 2) {
				lh_pclass.at(0).at(1)++;
			}
			else if (pclass.at(i) == 3) {
				lh_pclass.at(0).at(2)++;
			}
			else {}
		}
		else if(survived.at(i) == 1) {
			if(pclass.at(i) == 1) {
				lh_pclass.at(1).at(0)++;
			}
			else if(pclass.at(i) == 2) {
				lh_pclass.at(1).at(1)++;
			}
			else if (pclass.at(i) == 3) {
				lh_pclass.at(1).at(2)++;
			}
			else {}
		}
		else{}
	}
	
	for(int i = 0; i < lh_pclass.size(); i++) {
		for(int j = 0; j < lh_pclass[i].size(); j++) {
			if(i == 0) {
				lh_pclass.at(i).at(j) = lh_pclass.at(i).at(j) / count_survived.at(0).at(0);
			}
			else if(i == 1) {
				lh_pclass.at(i).at(j) = lh_pclass.at(i).at(j) / count_survived.at(0).at(1);
			}
		}
	}
	
	return lh_pclass;
}

// calculates sex survived likelihood of train data
vector<vector<double>> likelihoodSex (vector<double> survived, vector<double> sex, vector<vector<double>> count_survived) {
	
	vector<vector<double>> lh_sex (2, vector<double>(2,0)); 				// declare as a 2 x 2 matrix.
	
	for(int i = 0; i < survived.size(); i++) {
		if(survived.at(i) == 0) {
			if(sex.at(i) == 0) {
				lh_sex.at(0).at(0)++;
			}
			else if(sex.at(i) == 1) {
				lh_sex.at(0).at(1)++;
			}
			else {}
		}
		else if(survived.at(i) == 1) {
			if(sex.at(i) == 0) {
				lh_sex.at(1).at(0)++;
			}
			else if(sex.at(i) == 1) {
				lh_sex.at(1).at(1)++;
			}
			else {}
		}
		else{}
	}

	for(int i = 0; i < lh_sex.size(); i++) {
		for(int j = 0; j < lh_sex[i].size(); j++) {
			if(i == 0) {
				lh_sex.at(i).at(j) = lh_sex.at(i).at(j) / count_survived.at(0).at(0);
			}
			else if(i == 1) {
				lh_sex.at(i).at(j) = lh_sex.at(i).at(j) / count_survived.at(0).at(1);
			}
		}
	}
	
	return lh_sex;
}

// calculates mean given a vector
double calcMean(vector<double> vect) {
	double sum = 0;	
	for(int i = 0; i < vect.size(); i++) {
		sum += vect.at(i);
	}
	double mean = sum / vect.size();
	
	return mean;
}

// calculates variance given a vector
double calcVariance(vector<double> vect) {
	double var = 0;
	double mean = calcMean(vect);
	
	for(int i = 0; i < vect.size(); i++) {
		var += pow((vect.at(i) - mean), 2);
	}
		
	return var / (vect.size()-1);
}

// calculates mean of age train data
vector<vector<double>> ageMean (vector<double> survived, vector<double> age, vector<vector<double>> count_survived) {
	vector<vector<double>> mean(1, vector<double> (2, 0)); // declare as a 1 x 2
		
	for(int i = 0; i < survived.size(); i++) {
		if(survived.at(i) == 0) {
			mean.at(0).at(0) += age.at(i);
		}
		else if(survived.at(i) == 1) {
			mean.at(0).at(1) += age.at(i);
		}
		else{}
	}
		
	for(int i = 0; i < mean.size(); i++) {
		for(int j = 0; j < mean[i].size(); j++) {
			if(j == 0) {
				mean.at(i).at(j) = mean.at(i).at(j) / count_survived.at(0).at(0);
			}
			else if(j == 1) {
				mean.at(i).at(j) = mean.at(i).at(j) / count_survived.at(0).at(1);
			}
		}
	}
	return mean;
}

// calculates variance of age train data
vector<vector<double>> ageVar (vector<double> survived, vector<double> age, vector<vector<double>> count_survived) {
	vector<vector<double>> var(1, vector<double> (2, 0)); // declare as a 1 x 2	
	vector<vector<double>> mean = ageMean(survived, age, count_survived); // 1 x 2
			
	for(int i = 0; i < survived.size(); i++) {
		if(survived.at(i) == 0) {
			var.at(0).at(0) += pow( ( age.at(i) - mean.at(0).at(0) ), 2);
		}
		else if(survived.at(i) == 1) {
			var.at(0).at(1) += pow( ( age.at(i) - mean.at(0).at(1) ), 2);
		}
		else{}
	}
		
	for(int i = 0; i < var.size(); i++) {
		for(int j = 0; j < var[i].size(); j++) {
			if(j == 0) {
				var.at(i).at(j) = var.at(i).at(j) / ( count_survived.at(0).at(0) - 1 ) ;
			}
			else if (j == 1) {
				var.at(i).at(j) = var.at(i).at(j) / ( count_survived.at(0).at(1) - 1 ) ;
			}
			else {}
		}
	}	
	return var;
}

// formatting of age metrics into a 2x2 matrix
vector<vector<double>> age_metrics (vector<vector<double>> ageMean, vector<vector<double>> ageVar) {
	vector<vector<double>> metrics(2, vector<double>(2, 0));  // 2x2
	
	metrics.at(0).at(0) = ageMean.at(0).at(0);
	metrics.at(0).at(1) = sqrt(ageVar.at(0).at(0));
	metrics.at(1).at(0) = ageMean.at(0).at(1);
	metrics.at(1).at(1) = sqrt(ageVar.at(0).at(1));
	
	return metrics;
}

// calculates age quantative likelihood
double calc_age_lh (double v, double mean_v, double var_v) {
	double age_lh = 0;
	
	age_lh = (1 / (sqrt(2 * M_PI * var_v))) * exp( -(pow((v - mean_v), 2)) / (2*var_v));
	
	return age_lh;
}

// bayes theorem implementation
vector<vector<double>> calc_raw_prob(double pclass, double sex, double age, vector<vector<double>> apriori, vector<vector<double>> lh_pclass, vector<vector<double>> lh_sex, vector<vector<double>> age_mean, vector<vector<double>> age_var) {
	
	vector<vector<double>> raw(1, vector<double> (2, 0)); // declare as a 1 x 2
	
	// survived
	double num_s = lh_pclass.at(1).at(pclass-1) * lh_sex.at(1).at(sex) * apriori.at(0).at(1) *
					calc_age_lh(age, age_mean.at(0).at(1), age_var.at(0).at(1));		
					
	// perished	
	double num_p = lh_pclass.at(0).at(pclass-1) * lh_sex.at(0).at(sex) * apriori.at(0).at(0) *
					calc_age_lh(age, age_mean.at(0).at(0), age_var.at(0).at(0));				
		
	double denominator = lh_pclass.at(1).at(pclass-1) * lh_sex.at(1).at(sex) *
					calc_age_lh(age, age_mean.at(0).at(1), age_var.at(0).at(1)) * apriori.at(0).at(1)
					+ lh_pclass.at(0).at(pclass-1) * lh_sex.at(0).at(sex) * 
					calc_age_lh(age, age_mean.at(0).at(0), age_var.at(0).at(0)) * apriori.at(0).at(0);
		
	raw.at(0).at(1) = num_s / denominator;
	raw.at(0).at(0) = num_p / denominator;
	
	return raw;
}

/* returns the TP, FP, FN, TN accuracy values in comparing two given two-dimensional matrices */ 
vector<vector<double>> confusionMatrix(vector<double> matA, vector<double> matB) {
	vector<vector<double>> table(2, vector<double>(2, 0));  // 2x2
	
	// matA = predicted, matB = test$survived
	
	/*		TP FP
	 * 		FN TN		*/ 
	
	for(int i = 0; i < matA.size(); i++) {
			if( matA.at(i) == 0 && matB.at(i) == 0 ) {			// true negative
				table.at(0).at(0)++;
			}
			else if( matA.at(i) == 1 && matB.at(i) == 1 ) {		// true positive
				table.at(1).at(1)++;
			}
			else if( matA.at(i) == 1 && matB.at(i) == 0 ) {		// false positive
				table.at(1).at(0)++;
			}
			else if( matA.at(i) == 0 && matB.at(i) == 1 ) {		// false negative
				table.at(0).at(1)++;
			}
			else {}
		}	
	return table;
}

/* returns the accuracy in comparing two given two-dimensional matrices */
double accuracy(vector<double> matA, vector<double> matB) {
	int matARow = matA.size();
	int matBRow = matB.size();	
	
	if((matARow != matBRow)) {
		cout << "Error in Accuracy. Must Be Same Dimensions." << endl;
	}
		
	double sum = 0;
	
	for(int i = 0; i < matA.size(); i++) {
		if(matA.at(i) == matB.at(i)) {
				sum++;
		}

	}	
	return sum / matA.size();
}

#pragma GCC diagnostic pop
