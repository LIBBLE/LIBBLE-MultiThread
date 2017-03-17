/**
	* Copyright (c) 2016 LIBBLE team supervised by Dr. Wu-Jun LI at Nanjing University.
	* All Rights Reserved.
	* Licensed under the Apache License, Version 2.0 (the "License");
	* you may not use this file except in compliance with the License.
	* You may obtain a copy of the License at
	*
	* http://www.apache.org/licenses/LICENSE-2.0
	*
	* Unless required by applicable law or agreed to in writing, software
	* distributed under the License is distributed on an "AS IS" BASIS,
	* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
	* See the License for the specific language governing permissions and
	* limitations under the License. */
	
#ifndef MATH_FUNC_HPP
#define MATH_FUNC_HPP

/*the definations of math helper functions*/

#include <vector>
#include <cmath>
#include <iostream>
using namespace std;

void normalize(vector<double> &vec) {
	double sum = 0;
	for (auto x : vec) {
		sum += x * x;
	}
	sum = sqrt(sum);
	for (auto &x : vec) {
		x /= sum;
	}
}

double sign(double x) {
	if (x>0)
		return 1;
	else if (x<0)
		return -1;
	else 
		return 0;
}

double sum(const vector<double> &vec) {
	double total = 0;
	for (auto &x:vec)
		total += x;
	return total;
}

double sqr_sum(const vector<double> &vec) {
	double total = 0;
	for (auto &x:vec)
		total += x * x;
	return total;
}

double abs_sum(const vector<double> &vec) {
	double total = 0;
	for (auto &x :vec)
		total += fabs(x);
	return total;
}

void reset(vector<double> &vec) {
	for (auto &x:vec)
		x = 0;
}

double dot_product(const vector<double> &vec1, const vector<double> &vec2) {
	if (vec1.size() != vec2.size()) {
		cout << "dimension not mapping!" << endl;
		exit(1);
	}
	
	double total = 0;
	for (int i=0; i<vec1.size(); i++)
		total += vec1[i] * vec2[i];
	return total;
}

void vec_add(vector<double> &vec1, const vector<double> &vec2, double rate) {	//add rate * vec2 to vec1
	if (vec1.size() != vec2.size()) {
		cout << "dimension not mapping!" << endl;
		exit(1);
	}
	
	for (int i=0; i<vec1.size(); i++)
		vec1[i] += rate * vec2[i];
}

double sqr(double x) {
	return x * x;
}

#endif