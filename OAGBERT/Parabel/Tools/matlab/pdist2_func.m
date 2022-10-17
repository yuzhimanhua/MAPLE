function simMat = pdist2_func(trnFtMat,tstFtMat,simfunc,K,P)

    blockSize = 1000;               % Due to limited memory, test vectors are processed in blocks of size 1000. Change this according to available ram.
    %       pdist2 is very slow ...
    
	numTrn = size(trnFtMat,2);
	numTst = size(tstFtMat,2);
	numFt = size(tstFtMat,1);

	prm.type = 'NORM_1';
	[~,trnFtMat] = norm_features.fit_transform(trnFtMat,prm);
	[~,tstFtMat] = norm_features.fit_transform(tstFtMat,prm);

	simMat = sparse(numTrn,0);
	parts = block_partition(numTst,blockSize);

	for i=1:size(parts,1)
		a = parts(i,1);
		b = parts(i,2);

		mat = tstFtMat(:,a:b);
		psimMat = trnFtMat'*mat;
		rankMat = sort_sparse_mat(sparse(psimMat));
		rankMat(rankMat>K) = 0;
		rankMat = spones(rankMat);
		psimMat = sparse(psimMat.*rankMat);
		simMat = [simMat psimMat]; 
	end
end
