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

#ifndef TRANSD_HPP
#define TRANSD_HPP

#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <string>
using namespace std;

#include "math_func.hpp"
#include "transbase.hpp"

class transD: public transbase {
public:
	vector<vector<double> > ent_proj_vec, ent_proj_vec_old;	//entity projection vector 
	vector<vector<double> > rel_proj_vec, rel_proj_vec_old;	//relation projection vector
	
	
	transD() = default;
	
	transD(int e_num, int r_num, int dim, int l1, double rate, int tmp)
		: transbase(e_num, r_num, dim, l1, rate, tmp) {} 	//our transD only support same dimension
	
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
		
		ent_proj_vec.resize(entity_num);
		for (int i=0; i<entity_num; i++) {
			ent_proj_vec[i].resize(embedding_dim);
			for (int j=0; j<embedding_dim; j++)
				ent_proj_vec[i][j] = unif(generator);
			normalize(ent_proj_vec[i]);	//may not need to normalize
		}
		
		rel_proj_vec.resize(relation_num);
		for (int i=0; i<relation_num; i++) {
			rel_proj_vec[i].resize(embedding_dim);
			for (int j=0; j<embedding_dim; j++)
				rel_proj_vec[i][j] = unif(generator);
			normalize(rel_proj_vec[i]);	//may not need to normalize
		}
		if (use_tmp) {
			entity_vec_old = entity_vec;
			relation_vec_old = relation_vec;
			ent_proj_vec_old = ent_proj_vec;
			rel_proj_vec_old = rel_proj_vec;
		}
	}
	
	void init_from_file(string data_path) {	//TransD may init from TransE
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
		
		ent_proj_vec.resize(entity_num);
		for (int i=0; i<entity_num; i++) {
			ent_proj_vec[i].resize(embedding_dim);
			for (int j=0; j<embedding_dim; j++)
				ent_proj_vec[i][j] = unif(generator);
			normalize(ent_proj_vec[i]);	//may not need to normalize
		}
		
		rel_proj_vec.resize(relation_num);
		for (int i=0; i<relation_num; i++) {
			rel_proj_vec[i].resize(embedding_dim);
			for (int j=0; j<embedding_dim; j++)
				rel_proj_vec[i][j] = unif(generator);
			normalize(rel_proj_vec[i]);	//may not need to normalize
		}
		
		if (use_tmp) {
			entity_vec_old = entity_vec;
			relation_vec_old = relation_vec;
			ent_proj_vec_old = ent_proj_vec;
			rel_proj_vec_old = rel_proj_vec;
		}
	}
	
	double triple_loss(int h, int r, int t) override {	//calculate the loss of a triplet
		vector<double> &h_vec = entity_vec[h], &hp_vec = ent_proj_vec[h];
		vector<double> &r_vec = relation_vec[r], &rp_vec = rel_proj_vec[r];
		vector<double> &t_vec = entity_vec[t], &tp_vec = ent_proj_vec[t];
		
		if (use_tmp) {
			h_vec = entity_vec_old[h];
			hp_vec = ent_proj_vec_old[h];
			r_vec = relation_vec_old[r];
			rp_vec = rel_proj_vec_old[r];
			t_vec = entity_vec_old[t];
			tp_vec = ent_proj_vec_old[t];
		}
		
		double diff = dot_product(h_vec, hp_vec) - dot_product(t_vec, tp_vec);
		double sum = 0;
		for (int i=0; i<embedding_dim; i++) {
			double x = h_vec[i] + r_vec[i] - t_vec[i] + diff * rp_vec[i];
			if (l1_norm)
				sum += fabs(x);
			else
				sum += x * x;
		}
		
		return sum;
	}	
	
	void gradient(int r, int h, int t, int h_, int t_) override {	//update parameters based on its gradient
		if (use_tmp) {
			vector<double> &r_vec = relation_vec[r], &rp_vec = rel_proj_vec[r];
			vector<double> &r_vec_o = relation_vec_old[r], &rp_vec_o = rel_proj_vec_old[r];
			
			double z1 = -learning_rate, z2 = learning_rate;
			vector<double> &h_vec = entity_vec[h], &hp_vec = ent_proj_vec[h];
			vector<double> &t_vec = entity_vec[t], &tp_vec = ent_proj_vec[t];
			vector<double> &ch_vec = entity_vec[h_], &chp_vec = ent_proj_vec[h_];
			vector<double> &ct_vec = entity_vec[t_], &ctp_vec = ent_proj_vec[t_];
			vector<double> &h_vec_o = entity_vec_old[h], &hp_vec_o = ent_proj_vec_old[h];
			vector<double> &t_vec_o = entity_vec_old[t], &tp_vec_o = ent_proj_vec_old[t];
			vector<double> &ch_vec_o = entity_vec_old[h_], &chp_vec_o = ent_proj_vec_old[h_];
			vector<double> &ct_vec_o = entity_vec_old[t_], &ctp_vec_o = ent_proj_vec_old[t_];
			
			
			
			vector<double> diff_vec1(embedding_dim), diff_vec2(embedding_dim);	//the difference vector of a triple, denote as v
			
			double diff1 = dot_product(h_vec_o, hp_vec_o) - dot_product(t_vec_o, tp_vec_o);	// (hp, h) - (tp, t)
			double diff2 = dot_product(ch_vec_o, chp_vec_o) - dot_product(ct_vec_o, ctp_vec_o);
			
			for (int i=0; i<embedding_dim; i++) {	//calculate the difference vec 
				double x1 = 2 * (h_vec_o[i] + r_vec_o[i] - t_vec_o[i] + diff1 * rp_vec_o[i]);
				double x2 = 2 * (ch_vec_o[i] + r_vec_o[i] - ct_vec_o[i] + diff2 * rp_vec_o[i]);
				if (l1_norm) {
					x1 = sign(x1);
					x2 = sign(x2);
				}
				diff_vec1[i] = x1;
				diff_vec2[i] = x2;
			}
			
			double v_rp1 = dot_product(diff_vec1, rp_vec_o);	//(v, rp)
			double v_rp2 = dot_product(diff_vec2, rp_vec_o);
			
			for (int i=0; i<embedding_dim; i++) {	//update r, rp, hp, tp, h, t together
				double hp_val = hp_vec_o[i], tp_val = tp_vec_o[i], h_val = h_vec_o[i], t_val = t_vec_o[i],
					chp_val = chp_vec_o[i], ctp_val = ctp_vec_o[i], ch_val = ch_vec_o[i], ct_val = ct_vec_o[i];
				r_vec[i] += z1 * diff_vec1[i] + z2 * diff_vec2[i];
				rp_vec[i] += z1 * diff1 * diff_vec1[i] + z2 * diff2 * diff_vec2[i];
				hp_vec[i] += z1 * v_rp1 * h_val;
				tp_vec[i] += -z1 * v_rp1 * t_val;
				h_vec[i] += z1 * (diff_vec1[i] + diff_vec1[i] * hp_val);
				t_vec[i] += -z1 * (diff_vec1[i] + diff_vec1[i] * tp_val);
				chp_vec[i] += z2 * v_rp2 * ch_val;
				ctp_vec[i] += -z2 * v_rp2 * ct_val;
				ch_vec[i] += z2 * (diff_vec2[i] + diff_vec2[i] * chp_val);
				ct_vec[i] += -z2 * (diff_vec2[i] + diff_vec2[i] * ctp_val);
			}
			
			normalize(h_vec);
			normalize(t_vec);
			normalize(ch_vec);
			normalize(ct_vec);
			normalize(r_vec);
			norm_product(rp_vec, hp_vec, h_vec);
			norm_product(rp_vec, tp_vec, t_vec);
			norm_product(rp_vec, chp_vec, ch_vec);
			norm_product(rp_vec, ctp_vec, ct_vec);
		} else {
			vector<double> &r_vec = relation_vec[r], &rp_vec = rel_proj_vec[r];
		
			double z1 = -learning_rate, z2 = learning_rate;
			vector<double> &h_vec = entity_vec[h], &hp_vec = ent_proj_vec[h];
			vector<double> &t_vec = entity_vec[t], &tp_vec = ent_proj_vec[t];
			vector<double> &ch_vec = entity_vec[h_], &chp_vec = ent_proj_vec[h_];
			vector<double> &ct_vec = entity_vec[t_], &ctp_vec = ent_proj_vec[t_];
			
			
			vector<double> diff_vec1(embedding_dim), diff_vec2(embedding_dim);	//the difference vector of a triple, denote as v
			
			double diff1 = dot_product(h_vec, hp_vec) - dot_product(t_vec, tp_vec);	// (hp, h) - (tp, t)
			double diff2 = dot_product(ch_vec, chp_vec) - dot_product(ct_vec, ctp_vec);
			
			for (int i=0; i<embedding_dim; i++) {	//calculate the difference vec 
				double x1 = 2 * (h_vec[i] + r_vec[i] - t_vec[i] + diff1 * rp_vec[i]);
				double x2 = 2 * (ch_vec[i] + r_vec[i] - ct_vec[i] + diff2 * rp_vec[i]);
				if (l1_norm) {
					x1 = sign(x1);
					x2 = sign(x2);
				}
				diff_vec1[i] = x1;
				diff_vec2[i] = x2;
			}
			
			double v_rp1 = dot_product(diff_vec1, rp_vec);	//(v, rp)
			double v_rp2 = dot_product(diff_vec2, rp_vec);
			
			for (int i=0; i<embedding_dim; i++) {	//update r, rp, hp, tp, h, t together
				double hp_val = hp_vec[i], tp_val = tp_vec[i], h_val = h_vec[i], t_val = t_vec[i],
					chp_val = chp_vec[i], ctp_val = ctp_vec[i], ch_val = ch_vec[i], ct_val = ct_vec[i];
				r_vec[i] += z1 * diff_vec1[i] + z2 * diff_vec2[i];
				rp_vec[i] += z1 * diff1 * diff_vec1[i] + z2 * diff2 * diff_vec2[i];
				hp_vec[i] += z1 * v_rp1 * h_val;
				tp_vec[i] += -z1 * v_rp1 * t_val;
				h_vec[i] += z1 * (diff_vec1[i] + diff_vec1[i] * hp_val);
				t_vec[i] += -z1 * (diff_vec1[i] + diff_vec1[i] * tp_val);
				chp_vec[i] += z2 * v_rp2 * ch_val;
				ctp_vec[i] += -z2 * v_rp2 * ct_val;
				ch_vec[i] += z2 * (diff_vec2[i] + diff_vec2[i] * chp_val);
				ct_vec[i] += -z2 * (diff_vec2[i] + diff_vec2[i] * ctp_val);
			}
			
			normalize(h_vec);
			normalize(t_vec);
			normalize(ch_vec);
			normalize(ct_vec);
			normalize(r_vec);
			norm_product(rp_vec, hp_vec, h_vec);
			norm_product(rp_vec, tp_vec, t_vec);
			norm_product(rp_vec, chp_vec, ch_vec);
			norm_product(rp_vec, ctp_vec, ct_vec);
		}
		
	}	
	
	void norm_product(vector<double> &rp_vec, vector<double> &ep_vec, vector<double> &e_vec) {
		vector<double> pro_vec(embedding_dim);
		while (1) {
			double sum = 0;
			double e_ep = dot_product(e_vec, ep_vec);
			for (int i=0; i<embedding_dim; i++) {
				double x = e_vec[i] + e_ep * rp_vec[i];
				sum += sqr(x);
				pro_vec[i] = 2 * x;
			}
			if (sum <= 1)
				break;
			
			//update
			double v_rp = dot_product(pro_vec, rp_vec);
			for (int i=0; i<embedding_dim; i++) {
				double e_val = e_vec[i], ep_val = ep_vec[i];
				rp_vec[i] -= learning_rate * (e_ep * pro_vec[i]);
				e_vec[i] -= learning_rate * (pro_vec[i] + v_rp * ep_val);
				ep_vec[i] -= learning_rate * (v_rp * e_val);
			}
			
			//without this it may not converge
			normalize(rp_vec);	
			normalize(ep_vec);
		}
	}
	
	void save_to_file(string data_path) override {
		ofstream file;
	
		file.open(data_path + "/TransD_entity_vec.txt");
		file << entity_num << ' '  << embedding_dim << endl;
		for (auto &vec: entity_vec) {
			for (auto x : vec)
				file << x << ' ';
			file << endl;
		}
		file.close();
	
		file.open(data_path + "/TransD_relation_vec.txt");
		file << relation_num << ' ' << embedding_dim << endl;
		for (auto &vec: relation_vec) {
			for (auto x : vec)
				file << x << ' ';
			file << endl;
		}
		file.close();
	
		file.open(data_path + "/TransD_ent_proj_vec.txt");
		file << entity_num << ' '  << embedding_dim << endl;
		for (auto &vec: ent_proj_vec) {
			for (auto x : vec)
				file << x << ' ';
			file << endl;
		}
		file.close();
		
		file.open(data_path + "/TransD_rel_proj_vec.txt");
		file << relation_num << ' ' << embedding_dim << endl;
		for (auto &vec: rel_proj_vec) {
			for (auto x : vec)
				file << x << ' ';
			file << endl;
		}
		file.close();
	}
	
	void read_from_file(string data_path) override {	//used in test
		ifstream file;
		file.open(data_path + "/TransD_entity_vec.txt");
		file >> entity_num >> embedding_dim;
		entity_vec.resize(entity_num);
		for (auto &vec : entity_vec) {
			vec.resize(embedding_dim);
			for (auto &x : vec)
				file >> x;		
		}
		file.close();
	
		file.open(data_path + "/TransD_relation_vec.txt");
		file >> relation_num >> embedding_dim;
		relation_vec.resize(relation_num);
		for (auto &vec : relation_vec) {
			vec.resize(embedding_dim);
			for (auto &x: vec)
				file >> x;
		}
		file.close();
		
		file.open(data_path + "/TransD_ent_proj_vec.txt");
		file >> entity_num >> embedding_dim;
		ent_proj_vec.resize(entity_num);
		for (auto &vec : ent_proj_vec) {
			vec.resize(embedding_dim);
			for (auto &x : vec)
				file >> x;		
		}
		file.close();
		
		file.open(data_path + "/TransD_rel_proj_vec.txt");
		file >> relation_num >> embedding_dim;
		rel_proj_vec.resize(relation_num);
		for (auto &vec : rel_proj_vec) {
			vec.resize(embedding_dim);
			for (auto &x : vec)
				file >> x;		
		}
		file.close();
		
		if (use_tmp) {
			entity_vec_old = entity_vec;
			relation_vec_old = relation_vec;
			ent_proj_vec_old = ent_proj_vec;
			rel_proj_vec_old = rel_proj_vec;
		}
	}	
	
	void batch_update() {
		entity_vec_old = entity_vec;
		relation_vec_old = relation_vec;
		ent_proj_vec_old = ent_proj_vec;
		rel_proj_vec_old = rel_proj_vec;
	}
};

#endif