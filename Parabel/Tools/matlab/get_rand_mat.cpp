#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <random>

#include <mex.h>
#include <ctime>

#include "config.h"
#include "utils.h"
#include "mat.h"
#include "mex_utils.h"

using namespace std;

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	// Generates random sparse pattern (binary) matrix whose each column contains given number of random entries as ones

	massert(nrhs>=3 && nrhs<=5,">=3 and <=5 inputs required: <no. rows>, <no. cols>, <''dense'' vector of nnzs in each col>, <optional: input pattern matrix>, <seed>");
	massert(nlhs==1,"Exactly 1 output required: <pattern matrix>");

	mxArray* marr = (mxArray*) prhs[0];
	_llint nr = mxGetPr(marr)[0];

	marr = (mxArray*) prhs[1];
	_llint nc = mxGetPr(marr)[0];

	massert(nc==mxGetN(prhs[2]),"arg-2 and column count in arg-3 should be exactly equal");

	marr = (mxArray*) prhs[2];
	double* ctPr = mxGetPr(marr);

	_llint nnz = 0;
	for(_llint i=0; i<nc; i++)
		nnz += (_llint)ctPr[i];

	mxArray* pat_mat = mxCreateSparse(nr,nc,nnz,mxREAL);
	mwIndex* Ir = mxGetIr(pat_mat);
	mwIndex* Jc = mxGetJc(pat_mat);
	double* Pr = mxGetPr(pat_mat);

	Jc[0] = 0;
	for(_llint i=1; i<=nc; i++)
		Jc[i] = Jc[i-1] + (_llint)ctPr[i-1];

	_llint seed = 0;
	if(nrhs>4)
	{
		marr = (mxArray*) prhs[4];
		seed = (_llint)mxGetPr(marr)[0];
	}

	_bool* mask = new _bool[nr]();

	if(nrhs>3)
	{
		marr = (mxArray*) prhs[3];
		mwIndex* inIr = mxGetIr(marr);
		mwIndex* inJc = mxGetJc(marr);
		double* inPr = mxGetPr(marr);

		for(_llint i=0; i<nc; i++)
		{
			massert(inJc[i+1]-inJc[i]>=ctPr[i], "nnzs of out patmat should be <= nnzs of in patmat condition fails");
		}

		mt19937 gen(seed);
		
		for(_llint i=0; i<nc; i++)
		{
			_llint n = Jc[i+1] - Jc[i];

			if(n==0)
				continue;

			uniform_int_distribution<> dist(inJc[i], inJc[i+1]-1);

			for(_llint j=Jc[i]; j<Jc[i+1]; j++)
			{
				_llint ind = inIr[dist(gen)];
				if(mask[ind])
				{
					j--;
				}
				else
				{
					mask[ind] = true;
					Ir[j] = ind;
					Pr[j] = 1;
				}
			}

			sort(Ir+Jc[i],Ir+Jc[i+1]);

			for(_llint j=Jc[i]; j<Jc[i+1]; j++)
			{
				mask[Ir[j]] = false;
			}
		}

	}
	else
	{
		mt19937 gen(seed);
		uniform_int_distribution<> dist(0, nr-1);

		for(_llint i=0; i<nc; i++)
		{
			for(_llint j=Jc[i]; j<Jc[i+1]; j++)
			{
				_llint ind = dist(gen);
				if(mask[ind])
				{
					j--;
				}
				else
				{
					mask[ind] = true;
					Ir[j] = ind;
					Pr[j] = 1;
				}
			}

			sort(Ir+Jc[i],Ir+Jc[i+1]);

			for(_llint j=Jc[i]; j<Jc[i+1]; j++)
			{
				mask[Ir[j]] = false;
			}
		}
	}

	plhs[0] = pat_mat;

	delete [] mask;
}
