#pragma once

#include <iostream>
#include <string>
#include <map>
#include <string>

#include <mex.h>

#include "config.h"
#include "utils.h"
#include "mat.h"

using namespace std;

inline void massert(_bool val,string mesg)
{
	if(!val)
		mexErrMsgTxt(mesg.c_str());
}

inline void fake_answer(_int nlhs, mxArray *plhs[])
{
  _int i;
  for(i=0;i<nlhs;i++)
    plhs[i] = mxCreateDoubleMatrix(0, 0, mxREAL);
}

inline SMat<float>* SMatF_mex_to_cpp(mxArray* marr)
{
	SMat<float>* mat = new SMat<float>;
	mat->nr = (_int)mxGetM(marr);
	mat->nc = (_int)mxGetN(marr);
	mat->size = new _int[mat->nc];
	mat->data = new pairIF*[mat->nc];

	double* Pr = mxGetPr(marr);
	mwIndex* Ir = mxGetIr(marr);
	mwIndex* Jc = mxGetJc(marr);

	for(_int i=0; i<mat->nc; i++)
	{
		_int n = Jc[i+1]-Jc[i];
		mat->size[i] = n;
		mat->data[i] = new pairIF[n];
		pairIF* vec = mat->data[i];

		for(_int j=0; j<n; j++)
		{
			vec[j].first = Ir[Jc[i]+j];
			vec[j].second = (float)Pr[Jc[i]+j];
		}
	}

	return mat;
}

inline mxArray* SMatF_cpp_to_mex(SMat<float>* mat)
{
	_llint nnz = 0;
	for(_int i=0; i<mat->nc; i++)
		nnz += mat->size[i];

	mxArray* marr = mxCreateSparse(mat->nr,mat->nc,nnz,mxREAL);
  	mwIndex* Ir = mxGetIr(marr);
  	mwIndex* Jc = mxGetJc(marr);
  	double* Pr = mxGetPr(marr);

	Jc[0] = 0;
	nnz = 0;
	for(_int i=0; i<mat->nc; i++)
	{
		pairIF* vec = mat->data[i];
		for(_int j=0; j<mat->size[i]; j++)
		{
			Ir[nnz] = vec[j].first;
			Pr[nnz] = vec[j].second;
			nnz++;
		}
		Jc[i+1] = nnz;
	}

	return marr;
}

