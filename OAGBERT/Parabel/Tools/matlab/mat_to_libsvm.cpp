#include <iostream>
#include <string>

#include "utils.h"
#include "mat.h"
#include "mex_utils.h"

using namespace std;


void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	massert(nlhs==0,"No lhs allowed");
	massert(nrhs==3,"Exactly 3 rhs (feature mat, label mat and filename to be written to) allowed");

	string fname = mxArrayToString((mxArray*)prhs[2]);
	SMat<float>* ft_mat = SMatF_mex_to_cpp((mxArray*)prhs[0]);
	SMat<float>* lbl_mat = SMatF_mex_to_cpp((mxArray*)prhs[1]);

	check_valid_filename(fname,false);
	ofstream fout;
	fout.open(fname);
	for(_int i=0; i<ft_mat->nc; i++)
	{
		for(_int j=0; j<lbl_mat->size[i]; j++)
		{
				if(j==0)
					fout<<lbl_mat->data[i][j].first;
				else
					fout<<","<<lbl_mat->data[i][j].first;
		}
		for(_int j=0; j<ft_mat->size[i]; j++)
			fout<<" "<<ft_mat->data[i][j].first<<":"<<ft_mat->data[i][j].second;
		fout<<endl;
	}

	fout.close();
	
	delete ft_mat;
	delete lbl_mat;
}
