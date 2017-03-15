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

#ifndef TRANSR_HPP
#define TRANSR_HPP

#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <string>
using namespace std;

#include "math_func.hpp"
#include "transbase.hpp"

class transR: public transbase {
public:
	int dim2;	//the dimension of relation space 
	vector<vector<vector<double> > > mat;	//projection matrices (each \in R^{dim2 \times \embedding_dim})
	vector<vector<vector<double> > > mat_old;
	
	transR() = default;
	
	transR(int e_num, int r_num, int dim, int dim_2, int l1, double rate, int tmp)
		: transbase(e_num, r_num, dim, l1, rate, tmp), dim2(dim_2) {}
	
	void initial() override {	//initialize parameters
		default_random_engine generator(random_device{}());
		uniform_real_distribution<double> unif1(-6/sqrt(embedding_dim), 6/sqrt(embedding_dim));
		uniform_real_distribution<double> unif2(-6/sqrt(dim2), 6/sqrt(dim2));
		
		entity_vec.resize(entity_num);
		for (int i=0; i<entity_num; i++) { 
			entity_vec[i].resize(embedding_dim);
			for (int j=0; j<embedding_dim; j++)
				entity_vec[i][j] = unif1(generator);
			normalize(entity_vec[i]);
		}
		relation_vec.resize(relation_num);
		for (int i=0; i<relation_num; i++) {
			relation_vec[i].resize(dim2);
			for (int j=0; j<dim2; j++)
				relation_vec[i][j] = unif2(generator);
			normalize(relation_vec[i]);
		}
		
		mat.resize(relation_num);
		for (int r=0; r<relation_num; r++) {
			mat[r].resize(dim2);
			for (int i=0; i<dim2; i++) {
				mat[r][i].resize(embedding_dim);
				mat[r][i][i] = 1;	//initialized as diagonal matrix (TransR code)
			}
		}
		
		if (use_tmp) {
			entity_vec_old = entity_vec;
			relation_vec_old = relation_vec;
			mat_old = mat;
		}
	}
	
	void init_from_file(string data_path) {	//TransR may need to initialized from TransE
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
		file >> relation_num >> dim2;
		relation_vec.resize(relation_num);
		for (auto &vec : relation_vec) {
			vec.resize(embedding_dim);
			for (auto &x: vec)
				file >> x;
		}
		file.close();
		
		mat.resize(relation_num);
		for (int r=0; r<relation_num; r++) {
			mat[r].resize(dim2);
			for (int i=0; i<dim2; i++) {
				mat[r][i].resize(embedding_dim);
				mat[r][i][i] = 1;	//initialized as diagonal matrix (TransR code)
			}
		}
		
		if (use_tmp) {
			entity_vec_old = entity_vec;
			relation_vec_old = relation_vec;
			mat_old = mat;
		}
	}
	
	double triple_loss(int h, int r, int t) override {	//calculate the loss of a triplet
		vector<double> &h_vec = entity_vec[h], &t_vec = entity_vec[t];
		vector<double> &r_vec = relation_vec[r];
		vector<vector<double> > &r_mat = mat[r];
		
		if (use_tmp) {
				h_vec = entity_vec_old[h];
				t_vec = entity_vec_old[t];
				r_vec = relation_vec_old[r];
				r_mat = mat_old[r];
		}
		
		double h_val, r_val, t_val;	
		double sum = 0;
		for (int i=0; i<dim2; i++) {
			r_val = r_vec[i];
			h_val = 0;
			t_val = 0;
			for (int j=0; j<embedding_dim; j++) {
				h_val += r_mat[i][j] * h_vec[j];
				t_val += r_mat[i][j] * t_vec[j];
			}
			double tmp = h_val + r_val - t_val;
			if (l1_norm == 1) 
				sum += fabs(tmp);
			else
				sum += tmp * tmp;
		}
		return sum;
	}	
	
	void gradient(int r, int h, int t, int h_, int t_) override {	//update parameters based on its gradient
		if (use_tmp) {
			vector<double> &r_vec = relation_vec[r];
			vector<vector<double> > &r_mat = mat[r];
			vector<double> &r_vec_o = relation_vec_old[r];
			vector<vector<double> > &r_mat_o = mat_old[r];
			
			double z1  = -learning_rate, z2 = learning_rate;
			vector<double> &h_vec = entity_vec[h], &t_vec = entity_vec[t];
			vector<double> &ch_vec = entity_vec[h_], &ct_vec = entity_vec[t];
			vector<double> &h_vec_o = entity_vec_old[h], &t_vec_o = entity_vec_old[t];
			vector<double> &ch_vec_o = entity_vec_old[h_], &ct_vec_o = entity_vec_old[t_];
			
			
			for (int i=0; i<dim2; i++) {
				vector<double> &row = r_mat[i], &row_o = r_mat_o[i];
				double x1 = 2 * (dot_product(row_o, h_vec_o) + r_vec_o[i] - dot_product(row_o, t_vec_o));
				double x2 = 2 * (dot_product(row_o, ch_vec_o) + r_vec_o[i] - dot_product(row_o, ct_vec_o));
				if (l1_norm) {
					x1 = sign(x1);
					x2 = sign(x2);
				}
				r_vec[i] += z1 * x1 + z2 * x2;	//update r_vec
				for (int j=0; j<embedding_dim; j++) {	
					double h_val = h_vec_o[j], t_val = t_vec_o[j], row_val = row_o[j], 
						ch_val = ch_vec_o[j], ct_val = ct_vec_o[j];
					h_vec[j] += z1 * x1 * row_val;
					t_vec[j] += z1 * (-x1 * row_val);
					ch_vec[j] += z2 * x2 * row_val;
					ct_vec[j] += z2 * (-x2 * row_val);
					row[j] += z1 * x1 * (h_val - t_val) + z2 * x2 * (ch_val - ct_val);
				}		
			}
				
			normalize(r_vec);
			normalize(h_vec);
			normalize(t_vec);
			normalize(ch_vec);
			normalize(ct_vec);
			norm_product(r_mat, h_vec);
			norm_product(r_mat, t_vec);
			norm_product(r_mat, ch_vec);
			norm_product(r_mat, ct_vec);
		} else {
			vector<double> &r_vec = relation_vec[r];
			vector<vector<double> > &r_mat = mat[r];
			
			double z1  = -learning_rate, z2 = learning_rate;
			vector<double> &h_vec = entity_vec[h], &t_vec = entity_vec[t];
			vector<double> &ch_vec = entity_vec[h_], &ct_vec = entity_vec[t_];
			
			for (int i=0; i<dim2; i++) {
				vector<double> &row = r_mat[i];
				double x1 = 2 * (dot_product(row, h_vec) + r_vec[i] - dot_product(row, t_vec));
				double x2 = 2 * (dot_product(row, ch_vec) + r_vec[i] - dot_product(row, ct_vec));
				if (l1_norm) {
					x1 = sign(x1);
					x2 = sign(x2);
				}
				r_vec[i] += z1 * x1 + z2 * x2;	//update r_vec
				for (int j=0; j<embedding_dim; j++) {	
					double h_val = h_vec[j], t_val = t_vec[j], row_val = row[j], 
						ch_val = ch_vec[j], ct_val = ct_vec[j];
					h_vec[j] += z1 * x1 * row_val;
					t_vec[j] += z1 * (-x1 * row_val);
					ch_vec[j] += z2 * x2 * row_val;
					ct_vec[j] += z2 * (-x2 * row_val);
					row[j] += z1 * x1 * (h_val - t_val) + z2 * x2 * (ch_val - ct_val);
				}		
			}
				
			normalize(r_vec);
			normalize(h_vec);
			normalize(t_vec);
			normalize(ch_vec);
			normalize(ct_vec);
			norm_product(r_mat, h_vec);
			norm_product(r_mat, t_vec);
			norm_product(r_mat, ch_vec);
			norm_product(r_mat, ct_vec);
		}
	}	
	
	void norm_product(vector<vector<double> > &matrix, vector<double> &vec) {	//normalize the mat-vec product less than 1
		while (1) {
			double sum = 0;
			vector<double> pro_vec(dim2);
			for (int i=0; i<dim2; i++) {
				double x = dot_product(matrix[i], vec);
				sum += sqr(x);
				pro_vec[i] = 2 * x;
			}
			if (sum <= 1)
				break;
				
			//update matrix
			vector<double> vec_gra(embedding_dim);
			for (int i=0; i<dim2; i++) {
				for (int j=0; j<embedding_dim; j++) {
					vec_gra[j] += matrix[i][j] * pro_vec[i];
					matrix[i][j] -= learning_rate * vec[j] * pro_vec[i];
				}
			}
				
			//update vector
			vec_add(vec, vec_gra, -learning_rate);
		}
	}
	
	void save_to_file(string data_path) override {
		ofstream file;
	
		file.open(data_path + "/TransR_entity_vec.txt");
		file << entity_num << ' '  << embedding_dim << endl;
		for (auto &vec: entity_vec) {
			for (auto x : vec)
				file << x << ' ';
			file << endl;
		}
		file.close();
	
		file.open(data_path + "/TransR_relation_vec.txt");
		file << relation_num << ' ' << dim2 << endl;
		for (auto &vec: relation_vec) {
			for (auto x : vec)
				file << x << ' ';
			file << endl;
		}
		file.close();
		
		file.open(data_path + "/TransR_mat.txt");
		file << relation_num << ' ' << dim2 << ' ' << embedding_dim << endl;
		for (auto &r_mat: mat) {
			for (auto &row : r_mat) {
				for (auto x : row)
					file << x << ' ';
				file << endl;
			}
			file << endl;
		}
		file.close();
	}
	
	void read_from_file(string data_path) override {	//used in test
		ifstream file;
		file.open(data_path + "/TransR_entity_vec.txt");
		file >> entity_num >> embedding_dim;
		entity_vec.resize(entity_num);
		for (auto &vec : entity_vec) {
			vec.resize(embedding_dim);
			for (auto &x : vec)
				file >> x;		
		}
		file.close();
	
		file.open(data_path + "/TransR_relation_vec.txt");
		file >> relation_num >> dim2;
		relation_vec.resize(relation_num);
		for (auto &vec : relation_vec) {
			vec.resize(dim2);
			for (auto &x: vec)
				file >> x;
		}
		file.close();
		
		file.open(data_path + "/TransR_mat.txt");
		file >> relation_num >> dim2 >> embedding_dim;
		mat.resize(relation_num);
		for (auto &r_mat : mat) {
			r_mat.resize(dim2);
			for (auto &row : r_mat) {
				row.resize(embedding_dim);
				for (auto &x : row) {
					file >> x;
				}
			}
		}
		if (use_tmp) {
			entity_vec_old = entity_vec;
			relation_vec_old = relation_vec;
			mat_old = mat;
		}
	}	
	
	void batch_update() {
		entity_vec_old = entity_vec;
		relation_vec_old  = relation_vec;
		mat = mat_old;
	}
};

#endif