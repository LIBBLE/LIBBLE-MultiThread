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

#ifndef TRANSE_HPP
#define TRANSE_HPP

#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <string>
using namespace std;

#include "math_func.hpp"
#include "transbase.hpp"

class transE: public transbase {
public:
	transE() = default;
	
	transE(int e_num, int r_num, int dim, int l1, double rate, int tmp): transbase(e_num, r_num, dim, l1, rate, tmp) {}
	
	void initial() override {	//initialize parameters
		default_random_engine generator(random_device{}());
		uniform_real_distribution<double> unif(-6/sqrt(embedding_dim), 6/sqrt(embedding_dim));
	
		entity_vec.resize(entity_num);
		for (int i=0; i<entity_num; i++) { 
			entity_vec[i].resize(embedding_dim);
			for (int j=0; j<embedding_dim; j++)
				entity_vec[i][j] = unif(generator);
			normalize(entity_vec[i]);
		}
		relation_vec.resize(relation_num);
		for (int i=0; i<relation_num; i++) {
			relation_vec[i].resize(embedding_dim);
			for (int j=0; j<embedding_dim; j++)
				relation_vec[i][j] = unif(generator);
			normalize(relation_vec[i]);
		}
		
		if (use_tmp) {
			entity_vec_old = entity_vec;
			relation_vec_old = relation_vec;
		}
	}
	
	double triple_loss(int h, int r, int t) override {	//calculate the loss of a triplet
		vector<double> &h_vec = entity_vec[h], &t_vec = entity_vec[t];
		vector<double> &r_vec = relation_vec[r];
		
		if (use_tmp) {
			h_vec = entity_vec_old[h];
			t_vec = entity_vec_old[t];
			r_vec =  relation_vec_old[r];
		}
		
		double sum = 0;
		for (int i=0; i<embedding_dim; i++) {
			double x = h_vec[i] + r_vec[i] - t_vec[i];
			if (l1_norm == 1)
				sum += fabs(x);
			else 
				sum += x * x;
		}
		return sum;	
	}	
	
	void gradient(int r, int h, int t, int h_, int t_) override {	//update parameters based on its gradient
		if (use_tmp) {
			vector<double> &h_vec = entity_vec[h], &t_vec = entity_vec[t];
			vector<double> &ch_vec = entity_vec[h_], &ct_vec = entity_vec[t_];
			vector<double> &r_vec = relation_vec[r];
			vector<double> &h_vec_o = entity_vec_old[h], &t_vec_o = entity_vec_old[t];
			vector<double> &ch_vec_o = entity_vec_old[h_], &ct_vec_o = entity_vec_old[t_];
			vector<double> &r_vec_o = relation_vec_old[r];
			for (int i=0; i<embedding_dim; i++) {
				double x = 2 * (h_vec_o[i] + r_vec_o[i] - t_vec_o[i]);
				if (l1_norm)
					x = sign(x);
				h_vec[i] -= learning_rate * x;
				r_vec[i] -= learning_rate * x;
				t_vec[i] += learning_rate * x;
			
				x = 2 * (ch_vec_o[i] + r_vec_o[i] - ct_vec_o[i]);
				if (l1_norm)
					x = sign(x);
				ch_vec[i] += learning_rate * x;
				r_vec[i] += learning_rate * x;
				ct_vec[i] -= learning_rate * x;
			}
		
			normalize(r_vec);
			normalize(h_vec);
			normalize(t_vec);
			normalize(ch_vec);
			normalize(ct_vec);
		} else {
			vector<double> &h_vec = entity_vec[h], &t_vec = entity_vec[t];
			vector<double> &ch_vec = entity_vec[h_], &ct_vec = entity_vec[t_];
			vector<double> &r_vec = relation_vec[r];
			
			double z1 = -learning_rate, z2 = learning_rate;
			
			for (int i=0; i<embedding_dim; i++) {
				double x1 = 2 * (h_vec[i] + r_vec[i] - t_vec[i]);
				double x2 = 2 * (ch_vec[i] + r_vec[i] - ct_vec[i]);
				if (l1_norm) {
					x1 = sign(x1);
					x2 = sign(x2);
				}
				
				r_vec[i] += z1 * x1 + z2 * x2;
				h_vec[i] += z1 * x1;				
				t_vec[i] += -z1 * x1;		
				ch_vec[i] += z2 * x2;
				ct_vec[i] += -z2 * x2;
			}
		
			normalize(r_vec);
			normalize(h_vec);
			normalize(t_vec);
			normalize(ch_vec);
			normalize(ct_vec);
		}
	}	
	
	void save_to_file(string data_path) override {
		ofstream file;
	
		file.open(data_path + "/TransE_entity_vec.txt");
		file << entity_num << ' '  << embedding_dim << endl;
		for (auto &vec: entity_vec) {
			for (auto x : vec)
				file << x << ' ';
			file << endl;
		}
		file.close();
	
		file.open(data_path + "/TransE_relation_vec.txt");
		file << relation_num << ' ' << embedding_dim << endl;
		for (auto &vec: relation_vec) {
			for (auto x : vec)
				file << x << ' ';
			file << endl;
		}
		file.close();
	}
	
	void read_from_file(string data_path) override {	//used in test
		ifstream file;
		file.open(data_path + "/TransE_entity_vec.txt");
		file >> entity_num >> embedding_dim;
		entity_vec.resize(entity_num);
		for (auto &vec : entity_vec) {
			vec.resize(embedding_dim);
			for (auto &x : vec)
				file >> x;		
		}
		file.close();
	
		file.open(data_path + "/TransE_relation_vec.txt");
		file >> relation_num >> embedding_dim;
		relation_vec.resize(relation_num);
		for (auto &vec : relation_vec) {
			vec.resize(embedding_dim);
			for (auto &x: vec)
				file >> x;
		}
		file.close();
		
		if (use_tmp) {
			entity_vec_old = entity_vec;
			relation_vec_old = relation_vec;
		}
	}	
	
	void batch_update() {
		entity_vec_old = entity_vec;
		relation_vec_old = relation_vec;
	}
};

#endif