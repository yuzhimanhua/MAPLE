#include <iostream>
#include <string>

#include <mex.h>

#include "config.h"
#include "mex_utils.h"

using namespace std;

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{

	massert(nlhs==1,"No. of Output variables should be 1");
	massert(nrhs==3,"No. of Input variables should be 3");

	const mxArray* mat1 = prhs[0];
	const mxArray* mat2 = prhs[1];
	const mxArray* patmat = prhs[2];

	massert(mxGetM(mat1)==mxGetM(mat2),"Dim 2 of both matrices should be same");
	massert(mxGetN(mat1)==mxGetM(patmat),"Shapes of pattern matrix and input matrices don't match");
	massert(mxGetN(mat2)==mxGetN(patmat),"Shapes of pattern matrix and input matrices don't match");

	_int m = mxGetN(mat1);
	_int n = mxGetN(mat2);
	_int k = mxGetM(mat1);

	mwIndex* ir_mat1 = mxGetIr(mat1);
	mwIndex* jc_mat1 = mxGetJc(mat1);
	double* pr_mat1 = mxGetPr(mat1);

	mwIndex* ir_mat2 = mxGetIr(mat2);
	mwIndex* jc_mat2 = mxGetJc(mat2);
	double* pr_mat2 = mxGetPr(mat2);

	mwIndex* ir_patmat = mxGetIr(patmat);
	mwIndex* jc_patmat = mxGetJc(patmat);
	double* pr_patmat = mxGetPr(patmat);

	_int nnz = (_int)jc_patmat[n];
	plhs[0] = mxDuplicateArray(patmat);

	mwIndex* ir_resmat = mxGetIr(plhs[0]);
	mwIndex* jc_resmat = mxGetJc(plhs[0]);
	double* pr_resmat = mxGetPr(plhs[0]);

	_double* vec = new _double[k]();

	for(_int i=0; i<n; i++)
	{
		_int ind2 = i;
		for(_llint k=jc_mat2[ind2]; k<jc_mat2[ind2+1]; k++)
			vec[ir_mat2[k]] = pr_mat2[k];

		for(_int j=jc_patmat[i]; j<jc_patmat[i+1]; j++)
		{
			_int ind1 = ir_patmat[j];
			_double prod = 0;

			for(_llint k=jc_mat1[ind1]; k<jc_mat1[ind1+1]; k++)
				prod += vec[ir_mat1[k]]*pr_mat1[k];
			
			pr_resmat[j] = prod;
		}

		for(_llint k=jc_mat2[ind2]; k<jc_mat2[ind2+1]; k++)
			vec[ir_mat2[k]] = 0;
	}

	delete [] vec;

}
