# This Makefile is used under Linux

MATLABDIR ?= /local/MATLAB/R2012b
CXX ?= g++
CFLAGS = -Wall -O3 -fPIC -I$(MATLABDIR)/extern/include

MEX = $(MATLABDIR)/bin/mex
MEX_OPTION = CC\#$(CXX) CXX\#$(CXX) CFLAGS\#"$(CFLAGS)" CXXFLAGS\#"$(CFLAGS)"
MEX_OPTION += -largeArrayDims
MEX_EXT = $(shell $(MATLABDIR)/bin/mexext)

all: read_sparse_ml.$(MEX_EXT)

read_sparse_ml.$(MEX_EXT):	read_sparse_ml.c
	$(MEX) $(MEX_OPTION) read_sparse_ml.c

clean:
	rm -f *~ *.o; rm -f *.mex*; rm -f *obj

