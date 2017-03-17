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


ParaGraphE is a parallel implementation of several knowledge graph embedding methods on a single machine with multiple cores and a shared memory.


-----------------
Compilation

We include the Makefile in this folder, just type "make" in command line to compile the code.

-----------------
Train

To start training, you can input a command like this:

	./train -nthreads 10 -method TransE -path data/WN18 -dim 50 -margin 4 -nepoches 1000 -nbatches 1 -use_tmp 0 

Several parameters can be set by the command line. They are:

-nthreads : The number of threads for executing. (default: 10)
-path: The path of data. (default: data/FB15k)
-method	: The method of embedding. Choice: TransE, TransH, TransR, TransD, SphereE. (default: TransE)
-nepoches : The number of epoches for training. (default: 1000)
-dim : The dimension of embeddings. (default: 50)
-rate : Learning rate. (default: 0.01)
-margin: The margin parameter in rank-based hinge loss. (default: 1)
-l1_norm : Use L1 norm in training. Choice: 0:L2_norm, 1:L1_norm. (default: 1)
-corr_method : Use "unif" or "bern" for sampling corrupted triples. Choice: 0:unif, 1:bern. (default: 0)
-nbatches: The number of batches in an epoch. Need to cooperate with "-use_tmp 1" to employ minibatch training. According to our experience, it's performance is usually not good due to asynchronous update in	multi-thread setting. So minibatch is not recommended. (default: 1)
-use_tmp : Whether to update parameters after a batch. (default: 0)
-init_from_file : Whether to init parameters from TransE result. TransH, TransR, TransD can have a perfermance boost initialized from TransE result. *Make sure you have already run TransE on this dataset once and have saved the result in that data path before you use this command.* (default: 0)
-orth_value (only in TransH) : The orthogonal parameter \episilon in TransH. (default: 0.1)
-dim2 (only in TransR) : The dimension of relational space in TransR. (default: equal to -dim)

After training with a notice "the embeddings have already been saved in files.", embeddings are saved in the same data path of that dataset. For example, after you run TransE on data/WN18, you can find two files named
 "TransE_entity_vec.txt" and "TransE_relation_vec.txt" in that path. Those files can be read by the testing program during the testing procedure.
 
-----------------
Testing

After you have saved embeddings in files, you can start testing with a command like this:

	./test -nthreads 10 -method TransE -path data/WN18 -l1_norm 1

Several parameters can be set by the command line. They are:

-nthreads : The number of threads for executing. (default: 10)
-path: The path of data. (default: data/FB15k)
-method	: The method of embedding. Choice: TransE, TransH, TransR, TransD, SphereE. (default: TransE)
-l1_norm : Use L1 norm in testing. Choice: 0:L2_norm, 1:L1_norm. (default: 1)

The program will report the result in several popular metrics of knowledge graph embedding. They are:

mr : The value of mean rank.
mrr : The value of mean reciprocal rank.
hits@10 : The proportion of ranks no larger than 10.
hits@1 : The porportion of ranks list at first.

In the output, those metrics with a "f_" mark before them are the results in "filter" settings. And those without "f_" are the results in "raw" settings.

-----------------
Implementing customized method

You can easily implement your own method based on our framework. Simply write a class by inheriting the class transbase, and then implement a construction function and the following pure virtual functions in transbase:

void initial() : How to initialize parameters.
double triple_loss(int h, int r, int t) : Calculating the loss of triple <h,r,t>.
void gradient(int r, int h, int t, int h_, int t_) : Doing gradient descent based on gold triple <h,r,t> and corrupted triple <h_, r, t_>.
void save_to_file(string data_path) : Save embeddings into files.
void read_from_file(string data_path) £ºRead embeddings from files when testing.
void batch_update() : Useful in use_tmp = 1. Update the parameters.

Then add one or two lines of code in train.cpp and test.cpp by following similar ways in our implemented methods. 
Modify Makefile and re-compile the program, then your method is ready to run. 