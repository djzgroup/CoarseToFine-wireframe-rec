cuda_inc = /usr/local/cuda-10.0/include/
cuda_lib = /usr/local/cuda-10.0/lib64/
nvcc = /usr/local/cuda-10.0/bin/nvcc
tf_inc = /usr/local/lib/python3.6/dist-packages/tensorflow/include
tf_lib = /usr/local/lib/python3.6/dist-packages/tensorflow

all: tf_nndistance_so.so tf_approxmatch_so.so

tf_nndistance_g.cu.o: tf_nndistance_g.cu
	$(nvcc) tf_nndistance_g.cu -o tf_nndistance_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

tf_nndistance_so.so: tf_nndistance.cpp tf_nndistance_g.cu.o
	g++ tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so \
	-I $(cuda_inc) -I $(tf_inc) -L $(cuda_lib) -lcudart -L $(tf_lib) -ltensorflow_framework \
    -shared -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -fPIC -O2

tf_approxmatch.cu.o: tf_approxmatch.cu
	$(nvcc) tf_approxmatch.cu -o tf_approxmatch.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

tf_approxmatch_so.so: tf_approxmatch.cpp tf_approxmatch.cu.o
	g++ -shared $(CPPFLAGS) tf_approxmatch.cpp tf_approxmatch.cu.o -o tf_approxmatch_so.so \
	-I $(cuda_inc) -I $(tf_inc) -L $(cuda_lib) -lcudart -L $(tf_lib) -ltensorflow_framework \
    -shared -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -fPIC -O2

clean:
	rm -rf *.o *.so
