function R = recall_at_k(score_mat,true_mat,k)
	num_inst = size(score_mat,2);
	num_lbl = size(score_mat,1);

	rank_mat = sort_sparse_mat(score_mat);
	R = zeros(k,1);
	lbl_count = sum(true_mat,1);
	lbl_count(lbl_count==0) = 1;

	for i=k:-1:1
		rank_mat(rank_mat>i) = 0;
		bmat = rank_mat>0;
		R(i) = mean(sum(bmat.*true_mat,1)./lbl_count);
	end
	R = full(R);
end

