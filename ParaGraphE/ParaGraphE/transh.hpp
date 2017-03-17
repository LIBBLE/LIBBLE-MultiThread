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

#ifndef TRANSH_HPP
#define TRANSH_HPP

#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <string>
using namespace std;

#include "math_func.hpp"
#include "transbase.hpp"

class transH: public transbase {
public:
	vector<vector<double> > project_vec;	//projection vector of TransH
	vector<vector<double> > project_vec_old;
	double orth_value = 0.1;	//the hyper-parameter of orthogonal constraint
	
	transH() = default;
	
	transH(int e_num, int r_num, int dim, int l1, double rate, double ov, int tmp)
		: transbase(e_num, r_num, dim, l1, rate, tmp), orth_value(ov) {}
	
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
		
		project_vec.resize(relation_num);
		for (int i=0; i<relation_num; i++) {
			project_vec[i].resize(embedding_dim);
			for (int j=0; j<embedding_dim; j++)
				project_vec[i][j] = unif(generator);
			normalize(project_vec[i]);
		}
		
		if (use_tmp) {
			entity_vec_old = entity_vec;
			relation_vec_old = relation_vec;
			project_vec_old = project_vec;
		}
	}
	
	void init_from_file(string data_path) {	//TransH may need to initialized from TransE
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
		
		default_random_engine generator(random_device{}());
		uniform_real_distribution<double> unif(-6/sqrt(embedding_dim), 6/sqrt(embedding_dim));
		
		project_vec.resize(relation_num);
		for (int i=0; i<relation_num; i++) {
			project_vec[i].resize(embedding_dim);
			for (int j=0; j<embedding_dim; j++)
				project_vec[i][j] = unif(generator);
			normalize(project_vec[i]);
		}
		
		if (use_tmp) {
			entity_vec_old = entity_vec;
			relation_vec_old = relation_vec;
			project_vec_old = project_vec;
		}
	}
	
	double triple_loss(int h, int r, int t) override {	//calculate the loss of a triplet
		vector<double> &h_vec = entity_vec[h], &t_vec = entity_vec[t];
		vector<double> &r_vec = relation_vec[r];
		vector<double> &proj_vec = project_vec[r];
		
		if (use_tmp) {
			h_vec = entity_vec_old[h];
			t_vec = entity_vec_old[t];
			r_vec = relation_vec_old[r];
			proj_vec = project_vec_old[r];
		}
		vector<double> differ_vec(embedding_dim);	// h-t
		
		for (int i=0; i<embedding_dim; i++)
			differ_vec[i] = h_vec[i] - t_vec[i];
		double pro = dot_product(differ_vec, proj_vec);
		
		double sum = 0;
		for (int i=0; i<embedding_dim; i++) {
			double tmp = r_vec[i] + differ_vec[i] - pro * proj_vec[i];
			if (l1_norm) 
				sum += fabs(tmp);
			else
				sum += tmp * tmp;
		}
		
		return sum;
 	}	
	
	void gradient(int r, int h, int t, int h_, int t_) override {	//update parameters based on its gradient
		if (use_tmp) {
			vector<double> &r_vec = relation_vec[r], &proj_vec = project_vec[r];
			vector<double> &r_vec_o = relation_vec_old[r], &proj_vec_o = project_vec_old[r];
			
			double z1 = -learning_rate, z2 = learning_rate;
			vector<double> &h_vec = entity_vec[h], &t_vec = entity_vec[t];
			vector<double> &ch_vec = entity_vec[h_], &ct_vec = entity_vec[t_];
			vector<double> &h_vec_o = entity_vec_old[h], &t_vec_o = entity_vec_old[t];
			vector<double> &ch_vec_o = entity_vec_old[h_], &ct_vec_o = entity_vec_old[t_];
			
			vector<double> diff_vec1(embedding_dim), diff_vec2(embedding_dim);	//the difference vec of the triple, denote as v

			double diff1 = -dot_product(proj_vec_o, h_vec_o) + dot_product(proj_vec_o, t_vec_o);	//-(w, h) + (w, t) 
			double diff2 = -dot_product(proj_vec_o, ch_vec_o) + dot_product(proj_vec_o, ct_vec_o);
			
			for (int i=0; i<embedding_dim; i++) {
				double x1 = 2 * (h_vec_o[i] + r_vec_o[i] - t_vec_o[i] + diff1 * proj_vec_o[i]);
				double x2 = 2 * (ch_vec_o[i] + r_vec_o[i] - ct_vec_o[i] + diff2 * proj_vec_o[i]);
				if (l1_norm) {
					x1 = sign(x1);
					x2 = sign(x2);
				}
				diff_vec1[i] = x1;
				diff_vec2[i] = x2;
			}
			
			double w_v1 = dot_product(proj_vec_o, diff_vec1); //(w, v)
			double w_v2 = dot_product(proj_vec_o, diff_vec2);
			double w_d = dot_product(proj_vec_o, r_vec_o);	//(w, r)
			
			for (int i=0; i<embedding_dim; i++) {	//update d, w, h, t
				double d_val = r_vec_o[i], w_val = proj_vec_o[i], 
					h_val = h_vec_o[i], t_val = t_vec_o[i], ch_val = ch_vec_o[i], ct_val = ct_vec_o[i];
				r_vec[i] += z1 * diff_vec1[i] + z2 * diff_vec2[i];
				proj_vec[i] += z1 * (diff1 * diff_vec1[i] + w_v1 * (t_val - h_val))
					+ z2 * (diff2 * diff_vec2[i] + w_v2 * (ct_val - ch_val));
				h_vec[i] += z1 * (diff_vec1[i] - diff1 * w_val);
				t_vec[i] += -z1 * (diff_vec1[i] - diff1 * w_val);
				ch_vec[i] += z2 * (diff_vec2[i] - diff2 * w_val);
				ct_vec[i] += -z2 * (diff_vec2[i] - diff2 * w_val);
			}
			
			normalize(h_vec);
			normalize(t_vec);
			normalize(ch_vec);
			normalize(ct_vec);
			normalize(proj_vec);
			normalize(r_vec);
			orthogonalize(proj_vec, r_vec);	
		} else {
			vector<double> &r_vec = relation_vec[r], &proj_vec = project_vec[r];
			
			double z1 = -learning_rate, z2 = learning_rate;
			vector<double> &h_vec = entity_vec[h], &t_vec = entity_vec[t];
			vector<double> &ch_vec = entity_vec[h_], &ct_vec = entity_vec[t_];
			
			vector<double> diff_vec1(embedding_dim), diff_vec2(embedding_dim);	//the difference vec of the triple, denote as v

			double diff1 = -dot_product(proj_vec, h_vec) + dot_product(proj_vec, t_vec);	//-(w, h) + (w, t) 
			double diff2 = -dot_product(proj_vec, ch_vec) + dot_product(proj_vec, ct_vec);
			
			for (int i=0; i<embedding_dim; i++) {
				double x1 = 2 * (h_vec[i] + r_vec[i] - t_vec[i] + diff1 * proj_vec[i]);
				double x2 = 2 * (ch_vec[i] + r_vec[i] - ct_vec[i] + diff2 * proj_vec[i]);
				if (l1_norm) {
					x1 = sign(x1);
					x2 = sign(x2);
				}
				diff_vec1[i] = x1;
				diff_vec2[i] = x2;
			}
			
			double w_v1 = dot_product(proj_vec, diff_vec1); //(w, v)
			double w_v2 = dot_product(proj_vec, diff_vec2);
			double w_d = dot_product(proj_vec, r_vec);	//(w, r)
			
			for (int i=0; i<embedding_dim; i++) {	//update d, w, h, t
				double d_val = r_vec[i], w_val = proj_vec[i], 
					h_val = h_vec[i], t_val = t_vec[i], ch_val = ch_vec[i], ct_val = ct_vec[i];
				r_vec[i] += z1 * diff_vec1[i] + z2 * diff_vec2[i];
				proj_vec[i] += z1 * (diff1 * diff_vec1[i] + w_v1 * (t_val - h_val))
					+ z2 * (diff2 * diff_vec2[i] + w_v2 * (ct_val - ch_val));
				h_vec[i] += z1 * (diff_vec1[i] - diff1 * w_val);
				t_vec[i] += -z1 * (diff_vec1[i] - diff1 * w_val);
				ch_vec[i] += z2 * (diff_vec2[i] - diff2 * w_val);
				ct_vec[i] += -z2 * (diff_vec2[i] - diff2 * w_val);
			}
			
			normalize(h_vec);
			normalize(t_vec);
			normalize(ch_vec);
			normalize(ct_vec);
			normalize(proj_vec);
			normalize(r_vec);
			orthogonalize(proj_vec, r_vec);
		}
	}	
		
	void orthogonalize(vector<double> &proj_vec, vector<double> &r_vec) {	//make w and d orthogonal
		while (1) {
			double sum = 0;
			for (int i=0; i<embedding_dim; i++) 
				sum += proj_vec[i] * r_vec[i];
			if (sum < orth_value)
				break;
			
			//update w and d
			for (int i=0; i<embedding_dim; i++) {
				double w_val = proj_vec[i], d_val = r_vec[i];
				proj_vec[i] -= learning_rate * d_val;
				r_vec[i] -= learning_rate * w_val;
			}
			
			normalize(r_vec);
		}
	}
	
	void save_to_file(string data_path) override {
		ofstream file;
	
		file.open(data_path + "/TransH_entity_vec.txt");
		file << entity_num << ' '  << embedding_dim << endl;
		for (auto &vec: entity_vec) {
			for (auto x : vec)
				file << x << ' ';
			file << endl;
		}
		file.close();
	
		file.open(data_path + "/TransH_relation_vec.txt");
		file << relation_num << ' ' << embedding_dim << endl;
		for (auto &vec: relation_vec) {
			for (auto x : vec)
				file << x << ' ';
			file << endl;
		}
		file.close();
	
		file.open(data_path + "/TransH_project_vec.txt");
		file << relation_num << ' ' << embedding_dim << endl;
		for (auto &vec: project_vec) {
			for (auto x : vec)
				file << x << ' ';
			file << endl;
		}
		file.close();
	}
	
	void read_from_file(string data_path) override {	//used in test
		ifstream file;
		file.open(data_path + "/TransH_entity_vec.txt");
		file >> entity_num >> embedding_dim;
		entity_vec.resize(entity_num);
		for (auto &vec : entity_vec) {
			vec.resize(embedding_dim);
			for (auto &x : vec)
				file >> x;		
		}
		file.close();
	
		file.open(data_path + "/TransH_relation_vec.txt");
		file >> relation_num >> embedding_dim;
		relation_vec.resize(relation_num);
		for (auto &vec : relation_vec) {
			vec.resize(embedding_dim);
			for (auto &x: vec)
				file >> x;
		}
		file.close();
		
		file.open(data_path + "/TransH_project_vec.txt");
		file >> relation_num >> embedding_dim;
		project_vec.resize(relation_num);
		for (auto &vec : project_vec) {
			vec.resize(embedding_dim);
			for (auto &x: vec)
				file >> x;
		}
		file.close();
		
		if (use_tmp) {
			entity_vec_old = entity_vec;
			relation_vec_old = relation_vec;
			project_vec_old = project_vec;
		}
	}	
	
	void batch_update() {
		entity_vec_old = entity_vec;
		relation_vec_old = relation_vec;
		project_vec_old = project_vec;
	}
};

#endif