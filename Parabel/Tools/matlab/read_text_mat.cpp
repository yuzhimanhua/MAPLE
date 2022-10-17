#include <iostream>
#include <string>

#include "utils.h"
#include "mat.h"
#include "mex_utils.h"

using namespace std;


void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	massert(nlhs==1,"Only one lhs (constructed matrix) allowed");
	massert(nrhs==1,"Only one rhs (filename to be read) allowed");
	
	string fname = mxArrayToString(prhs[0]);
	SMat<float>* smat = new SMat<float>(fname);
	plhs[0] = SMatF_cpp_to_mex(smat);
	delete smat;
}
