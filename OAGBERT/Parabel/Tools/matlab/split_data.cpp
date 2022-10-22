#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <random>
#include <numeric>

#include <cstdio>
#include <cstdlib>

#include <mex.h>

#include "mex_utils.h"

using namespace std;

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	// [split] = split_data(lbl_mat,split_frac_vec,split_no,small_or_large);	

	massert(nlhs==1,"No. of Output variables should be 1");	
	massert(nrhs>=1,"No. of Input variables should be at least 1");

	vector<float> parts;
	if(nrhs<=1 || mxIsEmpty(prhs[1]))
	{
		parts.push_back(0.8);
		parts.push_back(0.2);
	}
	else
	{
		int num = (int)mxGetM(prhs[1])*mxGetN(prhs[1]);
		double* vec = mxGetPr(prhs[1]);
		copy(vec,vec+num,back_inserter(parts));
	}

	int split_no;
	if(nrhs<=2 || mxIsEmpty(prhs[2]))
		split_no = 0;
	else
		split_no = (int)mxGetPr(prhs[2])[0];

	bool sl;
	if(nrhs<=3 || mxIsEmpty(prhs[3]))
		sl = false;
	else
		sl = (bool)mxGetPr(prhs[3])[0];

	int nparts = parts.size();

	const mxArray* mat = prhs[0];
	int num_inst = mxGetN(mat);
	int num_lbl = mxGetM(mat);

	plhs[0] = mxCreateDoubleMatrix(num_inst,1,mxREAL);
	double* split = mxGetPr(plhs[0]);

	mt19937 reng(split_no+1);
	discrete_distribution<int> dist(parts.begin(),parts.end());

	if(sl==false)
	{
		for(int i=0; i<num_inst; i++)
			split[i] = dist(reng);
	}
	else
	{
		vector<int> sorder;
		sorder.resize(nparts);

		mxArray* rmat = (mxArray*)mat;
		mxArray* tmat;
		mexCallMATLAB(1,&tmat,1,&rmat,"transpose");
		mwIndex* Ir = mxGetIr(tmat);
		mwIndex* Jc = mxGetJc(tmat);
		double* Pr = mxGetPr(tmat);
	
		for(int i=0; i<num_inst; i++)
			split[i] = -1;

		for(int i=0; i<num_lbl; i++)
		{
			iota(sorder.begin(),sorder.end(),0);
			shuffle(sorder.begin(),sorder.end(),reng);
			shuffle(Ir+Jc[i],Ir+Jc[i+1],reng);

			int pno = 0;
			for(int j=Jc[i]; j<Jc[i+1]; j++)
			{
				if(split[Ir[j]]==-1)
				{
					split[Ir[j]] = sorder[pno++];
					if(pno==nparts)
						break;
				}
			}
		}

		for(int i=0; i<num_inst; i++)
		{
			if(split[i]!=-1)
				continue;
			for(int i=0; i<num_inst; i++)
				split[i] = dist(reng);
		}
	
		mxDestroyArray(tmat);
	}	
}
