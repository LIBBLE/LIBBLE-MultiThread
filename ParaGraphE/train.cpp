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

#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <thread>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>
#include <tuple>
#include <set>
using namespace std;

#include "math_func.hpp"
#include "transbase.hpp"
#include "transe.hpp"
#include "transr.hpp"
#include "transh.hpp"
#include "transd.hpp"
#include "spheree.hpp"

//parameters of program
int nthreads = 1;	// number of threads
string data_path = "data/FB15k";	//input folder of the kg
vector<double> epoch_loss;
transbase *trans_ptr = nullptr;
string method = "TransE";
int init_from_file = 0;	//initialize from file
int use_tmp = 0;	//use a tmp value in each batch 

//hyper-parameters of algorithm
int embedding_dim = 50;	//the dimension of embeddings
int dim2 = 50;	//TransR only
double learning_rate = 0.01;  //learning rate
int corr_method = 0;	//sampling method, 0 = uniform, 1 = bernoulli 
double margin = 1;		//the margin of pairwise ranking loss 
int nepoches = 1000; 	//the num of epoches in training
int nbatches = 1;		//num of batches per epoch
int l1_norm = 1;		//0 = l2-norm, 1 = l1-norm
double orth_value = 0.1;	//TransH only

//parameters of the knowledge graph
int entity_num = 0, relation_num = 0;
int train_num = 0, valid_num = 0;

//data structure of algorithm
vector<int> train_h, train_r, train_t;
vector<int> valid_h, valid_r, valid_t; 
vector<set<int> > hset, tset;	//hset = <r, {h | <h,r,t> \in train_set}>
vector<int> r_count;	//count the times a relation appears
vector<double> tph, hpt;	
set<tuple<int, int, int> > triple_count;	

int arg_handler(string str, int argc, char **argv) {
	int pos;
	for (pos = 0; pos < argc; pos++) {
		if (str.compare(argv[pos]) == 0) {
			return pos;
		}
	}
}

void read_input() {
	int h, r, t;
	ifstream file;
	file.open(data_path + "/graph_info.txt");
	file >> entity_num >> relation_num;
	file.close();
	
	if (corr_method == 1) {
		hset.resize(relation_num);
		tset.resize(relation_num);
		tph.resize(relation_num);
		hpt.resize(relation_num);
		r_count.resize(relation_num);
	}
	
	file.open(data_path + "/train.txt");
	file >> train_num;
	for (int i=0; i<train_num; i++) {
		file >> h >> r >> t;
		train_h.push_back(h);
		train_r.push_back(r);
		train_t.push_back(t);
		if (corr_method==1) {
			hset[r].insert(h);
			tset[r].insert(t);
			r_count[r]++;
		}
		triple_count.insert(make_tuple(h,r,t));
	}
	file.close();
	
	if (corr_method==1) {
		for (int r=0; r<relation_num; r++) {
			tph[r] = (double)r_count[r] / hset[r].size();
			hpt[r] = (double)r_count[r] / tset[r].size();
		}
	}

	file.open(data_path + "/valid.txt");
	file >> valid_num;
	for (int i=0; i<valid_num; i++) {
		file >> h >> r >> t;
		valid_h.push_back(h);
		valid_r.push_back(r);
		valid_t.push_back(t);
	}
	file.close();
}

double valid_loss() {		//compute the loss on the valid set
	double total_loss = 0;
	for (int i=0; i<valid_num; i++) {
		total_loss += trans_ptr->triple_loss(valid_h[i], valid_r[i], valid_t[i]);
	}
	return total_loss;
}

double train_loss() {	//compute the loss on the train set
	double total_loss = 0;
	for (int i=0; i<train_num; i++) {
		total_loss += trans_ptr->triple_loss(train_h[i], train_r[i], train_t[i]);
	}
	return total_loss;
}

void pairwise_update(int r, int h, int t, int h_, int t_, int id) {	
	double gold_loss = trans_ptr->triple_loss(h, r, t), cor_loss = trans_ptr->triple_loss(h_, r, t_);
	double temp = gold_loss + margin - cor_loss;
	if (temp > 0) {
		epoch_loss[id] += temp;
		trans_ptr->gradient(r, h, t, h_, t_);
	}
}
void rand_train(int id, int size) {	//optimize using sgd without considering overlapping problem
	default_random_engine generator(random_device{}());
	uniform_int_distribution<int> triple_unif(0, train_num-1);
	uniform_int_distribution<int> entity_unif(0, entity_num-1);
	uniform_real_distribution<double> unif(-1, 1);
	for (int i=0; i<size; i++) {
		int tri_id = triple_unif(generator);
		int h = train_h[tri_id], r = train_r[tri_id], t = train_t[tri_id];
		int cor_id;	//the id of sub entity
		double prob;
		
		if (corr_method == 1) {
			uniform_real_distribution<double> bern(-hpt[r] ,tph[r]);
			prob = bern(generator);
		} else {
			prob = unif(generator);
		}
				
		if (prob > 0) {	//change head
			cor_id = entity_unif(generator);
			while (triple_count.find(make_tuple(cor_id,r,t)) != triple_count.end())
				cor_id = entity_unif(generator);
			pairwise_update(r, h, t, cor_id, t, id);
		} else {	//change tail
			cor_id = entity_unif(generator);
			while (triple_count.find(make_tuple(h,r,cor_id)) != triple_count.end())
				cor_id = entity_unif(generator);
			pairwise_update(r, h, t, h, cor_id, id);
		}
	}
}

void method_ptr_binding(string method) {
	if (method.compare("TransE") == 0) 
		trans_ptr = new transE(entity_num, relation_num, embedding_dim, l1_norm, learning_rate, use_tmp);
	else if (method.compare("TransR") == 0)
		trans_ptr = new transR(entity_num, relation_num, embedding_dim, dim2, l1_norm, learning_rate, use_tmp);
	else if (method.compare("TransH") == 0)
		trans_ptr = new transH(entity_num, relation_num, embedding_dim, l1_norm, learning_rate, orth_value, use_tmp);
	else if (method.compare("TransD") == 0)
		trans_ptr = new transD(entity_num, relation_num, embedding_dim, l1_norm, learning_rate, use_tmp);
	else if (method.compare("SphereE") == 0)	//dis in SphereE first init as margin
		trans_ptr = new sphereE(entity_num, relation_num, embedding_dim, l1_norm, learning_rate, margin, use_tmp);
	else {
		cout << "no such method!" << endl;
		exit(1);
	}
}

int main(int argc, char **argv) {
	//processing the args
	int pos;
	if ((pos = arg_handler("-nthreads", argc, argv)) > 0) nthreads = atoi(argv[pos + 1]);
	if ((pos = arg_handler("-dim", argc, argv)) > 0) embedding_dim = atoi(argv[pos + 1]);
	dim2 = embedding_dim;	//dim2 = embedding_dim in default setting
	if ((pos = arg_handler("-dim2", argc, argv)) > 0) dim2 = atoi(argv[pos + 1]);
	if ((pos = arg_handler("-rate", argc, argv)) > 0) learning_rate = atof(argv[pos + 1]);
	if ((pos = arg_handler("-orth_value", argc, argv)) > 0) orth_value = atof(argv[pos + 1]);
	if ((pos = arg_handler("-corr_method", argc, argv)) > 0) corr_method = atoi(argv[pos + 1]);
	if ((pos = arg_handler("-margin", argc, argv)) > 0) margin = atof(argv[pos + 1]);
	if ((pos = arg_handler("-nepoches", argc, argv)) > 0) nepoches = atoi(argv[pos + 1]);
	if ((pos = arg_handler("-nbatches", argc, argv)) > 0) nbatches = atoi(argv[pos + 1]);
	if ((pos = arg_handler("-l1_norm", argc, argv)) > 0) l1_norm = atoi(argv[pos + 1]);
	if ((pos = arg_handler("-path", argc, argv)) > 0) data_path = string(argv[pos + 1]);
	if ((pos = arg_handler("-method", argc, argv)) > 0) method = string(argv[pos + 1]);
	if ((pos = arg_handler("-init_from_file", argc, argv)) > 0) init_from_file = atoi(argv[pos + 1]);
	if ((pos = arg_handler("-use_tmp", argc, argv)) > 0) use_tmp = atoi(argv[pos + 1]);
	cout << "args process done." << endl;
	
	cout << "args settings:" << endl
		<< "----------" << endl
		<< "method " << method << endl
		<< "thread number " << nthreads << endl
		<< "dimension "  << embedding_dim << endl
		<< "dimension2 (only in transR) " << dim2 << endl
		<< "orthogonal value (only in transH) " << orth_value << endl
		<< "learning rate " << learning_rate << endl
		<< "sampling corrupt triple method " << (corr_method == 0 ? "unif" : "bern") << endl
		<< "margin " << margin << endl
		<< "epoch number " << nepoches << endl
		<< "batch number " << nbatches << endl
		<< "norm " << (l1_norm == 0 ? "L2" : "L1") << endl
		<< "initial from file " << (init_from_file == 1 ? "Yes" : "No") << endl
		<< "use tmp value in batches " << (use_tmp == 1 ? "Yes" : "No") << endl
		<< "data path " << data_path << endl
		<< "----------" << endl;
	
	this_thread::sleep_for(chrono::seconds(3));
	
	//initializing
	read_input();
	
	method_ptr_binding(method);
	
	if (init_from_file == 0)
		trans_ptr->initial();
	else 
		trans_ptr->init_from_file(data_path);
	
	int batch_size = train_num / nbatches;
	int thread_size = batch_size / nthreads;	//number of triples a thread handles in one run
	cout << "initializing process done." << endl;
	
	//set-up multi-thread and start learning
	thread workers[nthreads];
	epoch_loss.resize(nthreads);
	auto start = chrono::high_resolution_clock::now();
	for (int epoch = 0; epoch < nepoches; epoch++) {	
		reset(epoch_loss);
		for (int batch = 0; batch < nbatches; batch++) {
			for (int id=0; id<nthreads; id++) {
				workers[id] = thread(rand_train, id, thread_size);
			}
			for (auto &x: workers)
				x.join();
			if (use_tmp)	//if tmp is used then update for batch is needed
				trans_ptr->batch_update();
		} //batch iteration
		
		cout << "loss of epoch " << epoch << " is " << sum(epoch_loss) << endl;
		/*cout << "train and valid loss in epoch " << epoch << " is " << fixed << setprecision(6) 
			<< train_loss() << ' ' << valid_loss() << endl; */	
	} //epoch iteration
	auto end = chrono::high_resolution_clock::now();
	chrono::duration<double> diff = end-start;
	cout << "training process done, total time: " << diff.count() << " s." << endl;
	
	//finalizing
	trans_ptr->save_to_file(data_path);
	cout << "the embeddings are already saved in files." << endl;
}