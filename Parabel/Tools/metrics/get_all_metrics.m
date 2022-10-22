function metrics = get_all_metrics(score_mat, tst_lbl_mat, inv_prop)

	%% Inputs:
	%%	score_file: test label scores matrix file in sparse format
	%%	tst_lbl_file: test label ground truth matrix file in sparse format
	%%		sizes of matrices in score_file and tst_lbl_file must match, otherwise code will break
	%%  inv_prop: inverse propensity label weights, calculated using "inv_propensity" function

	%% Prints and returns:
	%% - precision at 1--5
	%% - nDCG at 1--5
	%% - propensity weighted precision at 1--5
	%% - propensity weighted nDCG at 1--5

	
	fprintf('prec ');
	prec = precision_k(score_mat,tst_lbl_mat,5)*100;
	clear mex;
	fprintf('%.2f %.2f %.2f %.2f %.2f\n',prec(1),prec(2),prec(3),prec(4),prec(5));
	metrics.prec_k = prec;

	fprintf('nDCG ');
	nDCG = nDCG_k(score_mat,tst_lbl_mat,5)*100;
	clear mex;
	fprintf('%.2f %.2f %.2f %.2f %.2f\n',nDCG(1),nDCG(2),nDCG(3),nDCG(4),nDCG(5));
	metrics.nDCG_k = nDCG;

	fprintf('prec_wt ');
	prec_wt = precision_wt_k(score_mat,tst_lbl_mat,inv_prop,5)*100;
	clear mex;
	fprintf('%.2f %.2f %.2f %.2f %.2f\n',prec_wt(1),prec_wt(2),prec_wt(3),prec_wt(4),prec_wt(5));
	metrics.prec_wt_k = prec_wt;

	fprintf('nDCG_wt ');
	nDCG_wt = nDCG_wt_k(score_mat,tst_lbl_mat,inv_prop,5)*100;
	clear mex;
	fprintf('%.2f %.2f %.2f %.2f %.2f\n',nDCG_wt(1),nDCG_wt(2),nDCG_wt(3),nDCG_wt(4),nDCG_wt(5));
	metrics.nDCG_wt_k = nDCG_wt;

end
