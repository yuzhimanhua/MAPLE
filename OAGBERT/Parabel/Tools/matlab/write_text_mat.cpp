#include <iostream>
#include <string>

#include "utils.h"
#include "mat.h"
#include "mex_utils.h"

using namespace std;


void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	massert(nlhs==0,"No lhs allowed");
	massert(nrhs==2,"Exactly 2 rhs (matrix to be written,filename to be written to) allowed");

	string fname = mxArrayToString((mxArray*)prhs[1]);
	SMat<float>* smat = SMatF_mex_to_cpp((mxArray*)prhs[0]);
	smat->write(fname);

	delete smat;
}
