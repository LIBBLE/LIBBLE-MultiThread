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

#ifndef SPHEREE_HPP
#define SPHEREE_HPP

#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <string>
using namespace std;

#include "math_func.hpp"
#include "transbase.hpp"

class sphereE: public transbase {
public:
	
	vector<double> distance, distance_old;	//store Dr of each relation 
	double dis_init_value;	//init the distance equal to the margin
	
	sphereE() = default;
	
	sphereE(int e_num, int r_num, int dim, int l1, double rate, double dis_init, int tmp)
		: transbase(e_num, r_num, dim, l1, rate, tmp), dis_init_value(dis_init) {}
	
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
		
		distance.resize(relation_num);
		for (int i=0; i<relation_num; i++)
			distance[i] = dis_init_value;
		
		if (use_tmp) {
			entity_vec_old = entity_vec;
			relation_vec_old = relation_vec;
			distance_old = distance;
		}
	}
	
	void init_from_file(string data_path) {	//SphereE may need to initialized from TransE
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
		
		distance.resize(relation_num);
		for (int i=0; i<relation_num; i++)
			distance[i] = dis_init_value;
		
		if (use_tmp) {
			entity_vec_old = entity_vec;
			relation_vec_old = relation_vec;
			distance_old = distance;
		}
	}
	
	double triple_loss(int h, int r, int t) override {	//calculate the loss of a triplet
		vector<double> &r_vec = relation_vec[r];
		vector<double> &h_vec = entity_vec[h], &t_vec = entity_vec[t];
		double &dis = distance[r];
		
		if (use_tmp) {
			r_vec = relation_vec_old[r];
			h_vec = entity_vec_old[h];
			t_vec = entity_vec_old[t];
			dis = distance_old[r];
		}
		double sum = 0;
		for (int i=0; i<embedding_dim; i++) {
			double x = h_vec[i] + r_vec[i] - t_vec[i];
			if (l1_norm)
				sum += fabs(x);
			else
				sum += x * x;
		}
		
		if (l1_norm)
			return fabs(sum - sqr(dis));
		else
			return sqr(sum - sqr(dis));
	}	
	
	void gradient(int r, int h, int t, int h_, int t_) override {	//update parameters based on its gradient
		if (use_tmp) {
			vector<double> &r_vec = relation_vec[r];
			vector<double> &r_vec_o = relation_vec_old[r];
			
			double &dis = distance[r], &dis_o = distance_old[r];
			
			vector<double> &h_vec = entity_vec[h], &t_vec = entity_vec[t];
			vector<double> &ch_vec = entity_vec[h_], &ct_vec = entity_vec[t_];
			vector<double> &h_vec_o = entity_vec_old[h], &t_vec_o = entity_vec_old[t];
			vector<double> &ch_vec_o = entity_vec_old[h_], &ct_vec_o = entity_vec_old[t_];
			
			vector<double> diff_vec1(embedding_dim), diff_vec2(embedding_dim);
			for (int i=0; i<embedding_dim; i++) {
				diff_vec1[i] = h_vec_o[i] + r_vec_o[i] - t_vec_o[i];
				diff_vec2[i] = ch_vec_o[i] + r_vec_o[i] - ct_vec_o[i];
			}
			
			double val1, val2;
			if (l1_norm) {
				val1 = sign(abs_sum(diff_vec1) - sqr(dis_o));
				val2 = sign(abs_sum(diff_vec2) - sqr(dis_o));
				for (int i=0; i<embedding_dim; i++) {
					diff_vec1[i] = sign(diff_vec1[i]);
					diff_vec2[i] = sign(diff_vec2[i]);
				}
			} else {
				val1 = sqr_sum(diff_vec1) - sqr(dis_o);
				val2 = sqr_sum(diff_vec2) - sqr(dis_o);
			}
			
			double z1 = -learning_rate, z2 = learning_rate;
			double z3 = (l1_norm ? 1:4);
			dis += (z1 * (-2 * val1) + z2 * (-2 * val2)) * dis_o;		//update dis
			for (int i=0; i<embedding_dim; i++) {		
				h_vec[i] +=  z3 * z1 * val1 * diff_vec1[i];
				t_vec[i] -= z3 * z1 * val1 * diff_vec1[i];
				ch_vec[i] += z3 * z2 * val2 * diff_vec2[i];
				ct_vec[i] -= z3 * z2 * val2 * diff_vec2[i];
				r_vec[i] += z3 * (z1 * val1 * diff_vec1[i] + z2 * val2 * diff_vec2[i]);			
			}
			
			normalize(r_vec);
			normalize(h_vec);
			normalize(t_vec);
			normalize(ch_vec);
			normalize(ct_vec);
			
		}else {
			vector<double> &r_vec = relation_vec[r];
			vector<double> &h_vec = entity_vec[h], &t_vec = entity_vec[t];
			vector<double> &ch_vec = entity_vec[h_], &ct_vec = entity_vec[t_];
			double &dis = distance[r];
			
			vector<double> diff_vec1(embedding_dim), diff_vec2(embedding_dim);
			for (int i=0; i<embedding_dim; i++) {
				diff_vec1[i] = h_vec[i] + r_vec[i] - t_vec[i];
				diff_vec2[i] = ch_vec[i] + r_vec[i] - ct_vec[i];
			}
			
			double val1, val2;
			if (l1_norm) {
				val1 = sign(abs_sum(diff_vec1) - sqr(dis));
				val2 = sign(abs_sum(diff_vec2) - sqr(dis));
				for (int i=0; i<embedding_dim; i++) {
					diff_vec1[i] = sign(diff_vec1[i]);
					diff_vec2[i] = sign(diff_vec2[i]);
				}
			} else {
				val1 = sqr_sum(diff_vec1) - sqr(dis);
				val2 = sqr_sum(diff_vec2) - sqr(dis);
			}
			
			double z1 = -learning_rate, z2 = learning_rate;
			double z3 = (l1_norm ? 1:4);
			
			dis += (z1 * (-2 * val1) + z2 * (-2 * val2)) * dis;		//update dis
			for (int i=0; i<embedding_dim; i++) {		
				h_vec[i] +=  z3 * z1 * val1 * diff_vec1[i];
				t_vec[i] -= z3 * z1 * val1 * diff_vec1[i];
				ch_vec[i] += z3 * z2 * val2 * diff_vec2[i];
				ct_vec[i] -= z3 * z2 * val2 * diff_vec2[i];
				r_vec[i] += z3 * (z1 * val1 * diff_vec1[i] + z2 * val2 * diff_vec2[i]);			
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
	
		file.open(data_path + "/SphereE_entity_vec.txt");
		file << entity_num << ' '  << embedding_dim << endl;
		for (auto &vec: entity_vec) {
			for (auto x : vec)
				file << x << ' ';
			file << endl;
		}
		file.close();
	
		file.open(data_path + "/SphereE_relation_vec.txt");
		file << relation_num << ' ' << embedding_dim << endl;
		for (auto &vec: relation_vec) {
			for (auto x : vec)
				file << x << ' ';
			file << endl;
		}
		file.close();
	
		file.open(data_path + "/SphereE_distance.txt");
		file << relation_num << endl;
		for (auto &x : distance) {
			file << x << ' ';
		}
		file.close();
	}
	
	void read_from_file(string data_path) override {	//used in test
		ifstream file;
		file.open(data_path + "/SphereE_entity_vec.txt");
		file >> entity_num >> embedding_dim;
		entity_vec.resize(entity_num);
		for (auto &vec : entity_vec) {
			vec.resize(embedding_dim);
			for (auto &x : vec)
				file >> x;		
		}
		file.close();
	
		file.open(data_path + "/SphereE_relation_vec.txt");
		file >> relation_num >> embedding_dim;
		relation_vec.resize(relation_num);
		for (auto &vec : relation_vec) {
			vec.resize(embedding_dim);
			for (auto &x: vec)
				file >> x;
		}
		file.close();
		
		file.open(data_path + "/SphereE_distance.txt");
		file >> relation_num;
		distance.resize(relation_num);
		for (auto &x : distance)
			file >> x;
		file.close();
	}	
	
	void batch_update() {
		entity_vec_old = entity_vec;
		relation_vec_old = relation_vec;
		distance_old = distance;
	}
};

#endif