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

#ifndef TRANSBASE_HPP
#define TRANSBASE_HPP

#include <iostream>
#include <vector>
#include <string>

using namespace std;

class transbase {
public:
	vector<vector<double> > entity_vec, relation_vec;
	vector<vector<double> > entity_vec_old, relation_vec_old;
	int embedding_dim, entity_num, relation_num;
	int l1_norm = 1;
	double learning_rate = 0.1;
	int use_tmp = 1;
	
	transbase() = default;
	transbase(int e_num, int r_num, int dim, int l1, double rate, int tmp):
		entity_num(e_num), relation_num(r_num), embedding_dim(dim), l1_norm(l1), learning_rate(rate), use_tmp(tmp) {}
	virtual void initial() = 0;	//initialize parameters
	virtual void init_from_file(string data_path) {} 
	virtual double triple_loss(int h, int r, int t) = 0;	//calculate the loss of a triplet
	virtual void gradient(int r, int h, int t, int h_, int t_) = 0 ;	//update parameters based on its gradient
	virtual void save_to_file(string data_path) = 0;	
	virtual void read_from_file(string data_path) = 0;	//used in test
	virtual void batch_update() = 0; 
	virtual ~transbase() = default;
};

#endif