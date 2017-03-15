TRANS_FILE = transbase.hpp transe.hpp transr.hpp transh.hpp transd.hpp spheree.hpp
COMPILE_ARG = -std=c++11 -lpthread -D_GLIBCXX_USE_NANOSLEEP -O3 -march=native

all: train test
train: $(TRANS_FILE) math_func.hpp train.cpp
	g++ $(TRANS_FILE) math_func.hpp train.cpp -o train $(COMPILE_ARG) 
test: $(TRANS_FILE) test.cpp 
	g++ $(TRANS_FILE) test.cpp -o test $(COMPILE_ARG)
