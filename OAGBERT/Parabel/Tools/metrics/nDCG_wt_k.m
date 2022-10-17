function N = nDCG_wt_k(score_mat,true_mat,wts,K)
	num_inst = size(score_mat,2);
	num_lbl = size(score_mat,1);

	wt_true_mat = spdiags(wts,0,num_lbl,num_lbl)*true_mat;
	num = helper(score_mat,true_mat,wts,K);
	den = helper(wt_true_mat,true_mat,wts,K);

	N = num./den;
end

function P = helper(score_mat,true_mat,lbl_wts,K)
	num_inst = size(score_mat,2);
	num_lbl = size(score_mat,1);

	P = zeros(K,1);
	wts = 1./log2((1:num_lbl)+1)';
	cum_wts = cumsum(wts);

	rank_mat = sort_sparse_mat(score_mat);
	[X,Y,V] = find(rank_mat);
	V = 1./log2(V+1);
	coeff_mat = sparse(X,Y,V,num_lbl,num_inst);

	for k=1:K
		mat = coeff_mat;
		mat(rank_mat>k) = 0;
		mat = mat.*true_mat;
		mat = spdiags(lbl_wts,0,num_lbl,num_lbl)*mat;
		num = sum(mat,1);

		count = k*ones(1,num_inst);
		den = cum_wts(count)';
		
		P(k) = mean(num./den);
	end
end
