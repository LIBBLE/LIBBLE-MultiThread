## ParaGraphE

ParaGraphE provides a library for parallel knowledge graph embedding, which implements several knowledge graph embedding methods on a single machine with multiple cores and a shared memory. The
implemented methods in the current version include TransE[1], TransH[2], TransR[3], TransD[4], and SphereE (the sphere method in ManifoldE)[5].

ParaGraphE aims at accelerating the speed of training and testing of knowledge graph embedding methods. We re-implement these methods in a multi-thread way, which achieves a significant time reduction without influencing the accuracy. The parallel optimization/learning algorithms of ParaGraphE are based on the lock-free strategies in [6,7].

More details about ParaGraphE can be found at the arXiv draft[8].

### [Compilation](#compilation)

We include the Makefile in this folder, just type "make" in command line to compile the code.

### [Train](train)

To start training, you can input a command like this:

~~~shell
./train -nthreads 10 -method TransE -path data/WN18 -dim 50 -margin 4 -nepoches 1000 -nbatches 1 -use_tmp 0 
~~~

Several parameters can be set by the command line. They are:

​	-nthreads : The number of threads for executing. (default: 10)

​	-path: The path of data. (default: data/FB15k)

​	-method	: The method of embedding. Choice: TransE, TransH, TransR, TransD, SphereE. (default: TransE)
​	-nepoches : The number of epoches for training. (default: 1000)

​	-dim : The dimension of embeddings. (default: 50)

​	-rate : Learning rate. (default: 0.01)

​	-margin: The margin parameter in rank-based hinge loss. (default: 1)

​	-l1\_norm : Use L1 norm in training. Choice: 0:L2\_norm, 1:L1\_norm. (default: 1)

​	-corr_method : Use "unif" or "bern" for sampling corrupted triples. Choice: 0:unif, 1:bern. (default: 0)

​	-nbatches: The number of batches in an epoch. Need to cooperate with "-use_tmp 1" to employ minibatch training. According to our experience, it's performance is usually not good due to asynchronous update in	multi-thread setting. So minibatch is not recommended. (default: 1)

​	-use_tmp : Whether to update parameters after a batch. (default: 0)

​	-init_from_file : Whether to init parameters from TransE result. TransH, TransR, TransD can have a perfermance boost initialized from TransE result. ***Make sure you have already run TransE on this dataset once and have saved the result in that data path before you use this command.*** (default: 0)

​	-orth_value (only in TransH) : The orthogonal parameter \episilon in TransH. (default: 0.1)

​	-dim2 (only in TransR) : The dimension of relational space in TransR. (default: equal to -dim)

After training with a notice "the embeddings have already been saved in files.", embeddings are saved in the same data path of that dataset. For example, after you run TransE on data/WN18, you can find two files named "TransE_entity_vec.txt" and "TransE_relation_vec.txt" in that path. Those files can be read by the testing program during the testing procedure.

###	[Test](#test)

After you have saved embeddings in files, you can start testing with a command like this:

~~~shell
./test -nthreads 10 -method TransE -path data/WN18 -l1_norm 1
~~~

Several parameters can be set by the command line. They are:

​	-nthreads : The number of threads for executing. (default: 10)

​	-path: The path of data. (default: data/FB15k)

​	-method	: The method of embedding. Choice: TransE, TransH, TransR, TransD, SphereE. (default: TransE)

​	-l1\_norm : Use L1 norm in testing. Choice: 0:L2\_norm, 1:L1_norm. (default: 1)



The program will report the result in several popular metrics of knowledge graph embedding. They are:

​	mr : The value of mean rank.

​	mrr : The value of mean reciprocal rank.

​	hits@10 : The proportion of ranks no larger than 10.

​	hits@1 : The porportion of ranks list at first.



In the output, those metrics with a "f\_" mark before them are the results in "filter" settings. And those without "f_" are the results in "raw" settings.

### [Implementing customized method](#method)

You can easily implement your own method based on our framework. Simply write a class by inheriting the class transbase, and then implement a construction function and the following pure virtual functions in transbase:

void initial() : How to initialize parameters.
double triple_loss(int h, int r, int t) : Calculating the loss of triple \<h,r,t\>.
void gradient(int r, int h, int t, int h\_, int t\_) : Doing gradient descent based on gold triple \<h,r,t\> and corrupted triple \<h\_, r, t\_\>.
void save_to_file(string data_path) : Save embeddings into files.
void read_from_file(string data_path) : Read embeddings from files when testing.
void batch_update() : Useful in use_tmp = 1. Update the parameters.

Then add one or two lines of code in train.cpp and test.cpp by following similar ways in our implemented methods. 
Modify Makefile and re-compile the program, then your method is ready to run.

### [Reference](#Reference)

[1] Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, Oksana Yakhnenko. Translating Embeddings for Modeling Multi-relational Data. *Proceedings of the 28th Advances in Neural Information Processing Systems (NIPS), 2013.*

[2] Zhen Wang, Jianwen Zhang, Jianlin Feng, Zheng Chen. Knowledge Graph Embedding by Translating on Hyperplanes. *Proceedings of the 28th AAAI Conference on Artificial Intelligence (AAAI), 2014.*

[3] Yankai Lin, Zhiyuan Liu, Maosong Sun, Yang Liu, Xuan Zhu. Learning Entity and Relation Embeddings for Knowledge Graph Completion. *Proceedings of the 29th AAAI Conference on Artificial Intelligence (AAAI), 2015.*

[4] Guoliang Ji, Shizhu He, Liheng Xu, Kang Liu, Jun Zhao. Knowledge Graph Embedding via Dynamic Mapping Matrix. *Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics (ACL), 2015.*

[5] Han Xiao, Minlie Huang, Xiaoyan Zhu. From One Point to a Manifold: Knowledge Graph Embedding for Precise Link Prediction. *Proceedings of the 25th International Joint Conference on Artificial Intelligence (IJCAI), 2016.*

[6] Benjamin Recht, Christopher Ré, Stephen J. Wright, Feng Niu. Hogwild: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent. *Proceedings of the 25th Annual Conference on Neural Information Processing Systems (NIPS), 2011.*

[7] Shen-Yi Zhao, Gong-Duo Zhang, Wu-Jun Li. Lock-Free Optimization for Non-Convex Problems. *Proceedings of the 31th AAAIConference on Artificial Intelligence (AAAI), 2017*.

[8] Xiao-Fan Niu, Wu-Jun Li. ParaGraphE: A Library for Parallel Knowledge Graph Embedding. *arXiv. 2017.*

### [Open Source](#open-source)

* GitHub URL: [https://github.com/LIBBLE/LIBBLE-MultiThread/](https://github.com/LIBBLE/LIBBLE-MultiThread/)

* Licence: This project follows [Apache Licence 2.0](https://www.apache.org/licenses/LICENSE-2.0)

* About Us: 

  Director: Wu-Jun Li

  Developer: Xiao-Fan Niu
