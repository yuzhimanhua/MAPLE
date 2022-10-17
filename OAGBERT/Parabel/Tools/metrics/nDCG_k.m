function N = nDCG_k(score_mat,true_mat,K)
	N = helper(score_mat,true_mat,K);
end

function P = helper(score_mat,true_mat,K)
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
		num = sum(mat,1);

		count = sum(true_mat,1);
		count = min(count,k);
		count(count==0) = 1;
		den = cum_wts(count)';
		
		P(k) = mean(num./den);
	end
end
