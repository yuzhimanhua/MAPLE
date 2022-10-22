#include <iostream>
#include <fstream>
#include <string>

#include "parabel.h"

using namespace std;

void help()
{
	cerr<<"Sample Usage :"<<endl;
	cerr<<"./parabel_predict [input feature file name] [input model folder name] [output score file name] -T 1 -s 0 -t 3 -B 10 -q 0"<<endl<<endl;
	cerr<<"-T = param.num_thread				: Number of threads									default=[value saved in trained model]"<<endl;
	cerr<<"-s = param.start_tree				: Starting index of the trees for prediction								default=[value saved in trained model]"<<endl;
	cerr<<"-t = param.num_tree				: Number of trees to be used for prediction								default=[value saved in trained model]"<<endl;
	cerr<<"-B = param.beam_width				: Beam search width for fast, approximate prediction					default=10"<<endl;
	cerr<<"-q = param.quiet				: Quiet option to restrict the output for reporting progress and debugging purposes 0=no quiet, 1=quiet		default=[value saved in trained model]"<<endl<<endl;
	cerr<<"The feature and score files are expected to be in sparse matrix text format. Refer to README.txt for more details"<<endl;
	exit(1);
}

Param parse_param(int argc, char* argv[], string model_dir)
{
	Param param(model_dir+"/param");

	string opt;
	string sval;
	_float val;

	for(_int i=0; i<argc; i+=2)
	{
		opt = string(argv[i]);
		sval = string(argv[i+1]);
		val = stof(sval);

		if(opt=="-T")
			param.num_thread = (_int)val;
		else if(opt=="-s")
			param.start_tree = (_int)val;
		else if(opt=="-t")
			param.num_tree = (_int)val;
		else if(opt=="-B")
			param.beam_width = (_int)val;
		else if(opt=="-q")
			param.quiet = (_bool)val;
	}

	return param;
}

int main(int argc, char* argv[])
{
	std::ios_base::sync_with_stdio(false);

	if(argc < 4)
		help();

	string ft_file = string(argv[1]);
	SMatF* tst_X_Xf = new SMatF(ft_file);

	string model_dir = string(argv[2]);
	check_valid_foldername(model_dir);

	string score_file = string(argv[3]);
	check_valid_filename(score_file,false);

	Param param = parse_param(argc-4, argv+4, model_dir);

	_float prediction_time, model_size;
	SMatF* score_mat = predict_trees( tst_X_Xf, param, model_dir, prediction_time, model_size );

	cout << "prediction time: " << 1000*(prediction_time/tst_X_Xf->nc) << " ms/point" << endl;
	cout << "model size: " << model_size/1e+9 << " GB" << endl;

	score_mat->write(score_file);

	delete tst_X_Xf;
	delete score_mat;
}
