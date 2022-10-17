#include <iostream>
#include <string>
#include <vector>

#include <mex.h>

#include "utils.h"
#include "mat.h"
#include "mex_utils.h"

#define llint long long int

using namespace std;

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	// Sorts the columns of a sparse matrix (ascending/descending) and outputs sorted value and index matrices

	massert(nrhs>=1 && nrhs<=2, "Allowed inputs: <matrix to sort>,<sort order ('ascend' or 'descend'), default='descend'>");
	massert(nlhs==1,"Exactly 1 output allowed: <rank-matrix>");

	string order = "descend";
	if(nrhs>1)
		order = mxArrayToString(prhs[1]);
	massert(order=="ascend" || order=="descend", "2nd parameter should be either 'ascend' or 'descend'");

	mxArray* inarr = (mxArray*)prhs[0];
	mwIndex* inIr = mxGetIr(inarr);
	mwIndex* inJc = mxGetJc(inarr);
	double* inPr = mxGetPr(inarr);

	mxArray* outarr = mxDuplicateArray(inarr);
	mwIndex* outIr = mxGetIr(outarr);
	mwIndex* outJc = mxGetJc(outarr);
	double* outPr = mxGetPr(outarr);

	vector<pair<llint,double>> vec;
	for(int i=0; i<mxGetN(inarr); i++)
	{
		vec.clear();
		for(llint j=inJc[i]; j<inJc[i+1]; j++)
			vec.push_back(make_pair(j,inPr[j]));

		if(order == "ascend")
			sort(vec.begin(),vec.end(),comp_pair_by_second<llint,double>);
		else if(order == "descend")
			sort(vec.begin(),vec.end(),comp_pair_by_second_desc<llint,double>);

		for(int j=0; j<vec.size(); j++)
			outPr[vec[j].first] = j+1;
	}	

	plhs[0] = outarr;
}
