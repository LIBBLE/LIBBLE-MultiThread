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
#include <fstream>
#include <thread>
#include <vector>
#include <algorithm>
#include <map>
#include <set>
#include <tuple>
#include <chrono>
using namespace std;

#include "transbase.hpp"
#include "transe.hpp"
#include "transr.hpp"
#include "transh.hpp"
#include "transd.hpp"
#include "spheree.hpp"

//parameters of program
int nthreads = 10;
string data_path = "data/FB15k";
int batch_size = 1000;
string method = "TransE";
transbase *trans_ptr;

//hyper-parameters of algorithm
int l1_norm = 1;

//parameters of knowledge_graph
int train_num, valid_num, test_num, entity_num, relation_num, embedding_dim;

//data structure of algorithm
vector<int> test_h, test_r, test_t;
vector<int> rank_sum, hits_10, hits_1, frank_sum, fhits_10, fhits_1;
vector<double> mrr_sum, fmrr_sum;
set<tuple<int, int, int> > triple_count;

int arg_handler(string str, int argc, char **argv) {
	int pos;
	for (pos = 0; pos < argc; pos++) {
		if (str.compare(argv[pos]) == 0) {
			return pos;
		}
	}
}

void initial() {
	ifstream file;	
	
	file.open(data_path + "/graph_info.txt");
	file >> entity_num >> relation_num;
	file.close();
	
	file.open(data_path + "/test.txt");
	file >> test_num;
	int h, r, t;
	for (int i=0; i<test_num; i++) {
		file >> h >> r >> t;
		test_h.push_back(h);
		test_r.push_back(r);
		test_t.push_back(t);
		triple_count.insert(make_tuple(h,r,t));
	}
	file.close();
	
	file.open(data_path + "/train.txt");
	file >> train_num;
	for (int i=0; i<train_num; i++) {
		file >> h >> r >> t;
		triple_count.insert(make_tuple(h,r,t));
	}
	file.close();
	
	file.open(data_path + "/valid.txt");
	file >> valid_num;
	for (int i=0; i<valid_num; i++) {
		file >> h >> r >> t;
		triple_count.insert(make_tuple(h,r,t));
	}
	file.close();
	
	rank_sum.resize(nthreads);
	hits_10.resize(nthreads);
	hits_1.resize(nthreads);
	frank_sum.resize(nthreads);
	fhits_10.resize(nthreads);
	fhits_1.resize(nthreads);
	mrr_sum.resize(nthreads);
	fmrr_sum.resize(nthreads);
}

bool cmp(pair<int, double> x, pair<int, double> y) {
	return x.second < y.second;
}

void test_triple(int id, int k) {
	int h = test_h[k], r = test_r[k], t = test_t[k];
	int rank, frank;
	vector<pair<int, double> > loss_vec;
	
	//calculating the result of replace head
	for (int e=0; e<entity_num; e++) {
		double loss = trans_ptr->triple_loss(e, r, t);
		loss_vec.push_back(make_pair(e, loss));
	}
	sort(loss_vec.begin(), loss_vec.end(), cmp);
	/*auto iter = find_if(loss_vec.begin(), loss_vec.end(),
		[&h] (const pair<int, double>& p) {return p.first == h;});
	int rank = iter - loss_vec.begin() + 1;*/
	frank = 0;
	for (rank=0; rank<entity_num; rank++) {
		int e = loss_vec[rank].first;
		if (e == h) 
			break;
		if (triple_count.find(make_tuple(e,r,t)) == triple_count.end())
			frank++;
	}
	rank++;
	frank++;
	rank_sum[id] += rank;
	hits_10[id] += (rank <= 10? 1:0);
	hits_1[id] += (rank == 1? 1:0);
	frank_sum[id] += frank;
	fhits_10[id] += (frank <= 10? 1:0);
	fhits_1[id] += (frank == 1? 1:0);
	mrr_sum[id] += (double)1 / rank;
	fmrr_sum[id] += (double)1 / frank;

	loss_vec.clear();	
	//calculating the result of replace tail
	for (int e=0; e<entity_num; e++) {
		double loss = trans_ptr->triple_loss(h, r, e);
		loss_vec.push_back(make_pair(e, loss));
	}
	sort(loss_vec.begin(), loss_vec.end(), cmp);
	/*iter = find_if(loss_vec.begin(), loss_vec.end(),
		[&t] (const pair<int, double>& p) {return p.first == t;});
	rank = iter - loss_vec.begin() + 1;*/
	frank = 0;
	for (rank=0; rank<entity_num; rank++) {
		int e = loss_vec[rank].first;
		if (e == t) 
			break;	
		if (triple_count.find(make_tuple(h,r,e)) == triple_count.end())
			frank++;
	}
	rank++;
	frank++;
	rank_sum[id] += rank;
	hits_10[id] += (rank <= 10? 1:0);
	hits_1[id] += (rank == 1? 1:0);
	frank_sum[id] += frank;
	fhits_10[id] += (frank <= 10? 1:0);
	fhits_1[id] += (frank == 1? 1:0);
	mrr_sum[id] += (double)1 / rank;
	fmrr_sum[id] += (double)1 / frank;
	
	loss_vec.clear();
}

void test(int id, int count) {
	int k = id + count;
	//cout << k << endl;
	while (k < test_num && k < count + batch_size) {
		test_triple(id, k);
		k += nthreads;
		//cout << k << " in " << id << endl;
	} 
}

void method_ptr_binding(string method) {
	if (method.compare("TransE") == 0) 
		trans_ptr = new transE(0, 0, 0, l1_norm, 0, 0);	//the other parameters may be modified in read_from_file
	else if (method.compare("TransR") == 0)
		trans_ptr = new transR(0, 0, 0, 0, l1_norm, 0, 0);
	else if (method.compare("TransH") == 0) 
		trans_ptr = new transH(0, 0, 0, l1_norm, 0, 0, 0);
	else if (method.compare("TransD") == 0)
		trans_ptr = new transD(0, 0, 0, l1_norm, 0, 0);
	else if (method.compare("SphereE") == 0)
		trans_ptr = new sphereE(0, 0, 0, l1_norm, 0, 0, 0);
	else {
		cout << "no such method!" << endl;
		exit(1);
	}
}

int main(int argc, char **argv) {
	//arg processing
	int pos;
	if ((pos = arg_handler("-nthreads", argc, argv)) > 0) nthreads = atoi(argv[pos + 1]);
	if ((pos = arg_handler("-path", argc, argv)) > 0) data_path = string(argv[pos + 1]);
	if ((pos = arg_handler("-batch_size", argc, argv)) > 0) batch_size = atoi(argv[pos + 1]);
	if ((pos = arg_handler("-method", argc, argv)) > 0) method = string(argv[pos + 1]);
	if ((pos = arg_handler("-l1_norm", argc, argv)) > 0) l1_norm = atoi(argv[pos + 1]); 
	cout << "arg processing done." << endl;
	
	cout << "args settings: " << endl
		<< "----------" << endl
		<< "method " << method << endl
		<< "norm " << (l1_norm == 0 ? "L2" : "L1") << endl
		<< "thread number " << nthreads << endl
		<< "data path " << data_path << endl
		<< "----------" << endl;
	this_thread::sleep_for(chrono::seconds(3));
	
	//initializing
	initial();
	
	method_ptr_binding(method);
	
	trans_ptr->read_from_file(data_path);
	cout << "initializing process done." << endl;
	
	//testing
	auto start = chrono::high_resolution_clock::now();
	thread workers[nthreads];
	int count = 0;
	while (count < test_num) {
		cout << count << endl;
		for (int id=0; id<nthreads; id++)
			workers[id] = thread(test, id, count);
		for (auto &x: workers)
			x.join();
		count += batch_size;
	}
	auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = end-start;
	cout << "testing process done, total time: " << diff.count() << " s." << endl;
	
	//calculate the final result
	double sum = 0;
	double r1, r2, r3, r4, r5, r6, r7, r8;
	for (auto x : rank_sum)
		sum += x;
	r1 = sum / (2*test_num);
	sum = 0;
	for (auto x : hits_10)
		sum += x;
	r2 = sum / (2*test_num);
	sum = 0;
	for (auto x : frank_sum)
		sum += x;
	r3 = sum / (2*test_num);
	sum = 0;
	for (auto x : fhits_10)
		sum += x;
	r4 = sum / (2*test_num);
	sum = 0;
	for (auto x : mrr_sum)
		sum += x;
	r5 = sum / (2*test_num);
	sum = 0;
	for (auto x : hits_1)
		sum += x;
	r6 = sum / (2*test_num);
	sum = 0;
	for (auto x : fmrr_sum)
		sum += x;
	r7 = sum / (2*test_num);
	sum = 0;
	for (auto x : fhits_1)
		sum += x;
	r8 = sum / (2*test_num);
	cout << "test result:" << endl
		<< "-----------------------------" << endl
		<< "mr			hits@10			f_mr		f_hits@10" << endl
		<< r1 << "\t" << r2 << "\t" << r3 << "\t" << r4 << endl
		<<"mrr			hits@1			f_mrr		f_hits@1" << endl
		<< r5 << "\t" << r6 << "\t" << r7 << "\t" << r8 << endl;
}