function dist_mat = trunc_dist_mat(X,Y,param)
    %% 
    % param.trunc_val = int (default=5)
    % param.dist_type = 'cosine'/'euclidean' (default='cosine')
    % param.block_size = int (default=1000)
    %%
    
    param = complete_param(param);
    
    dist_mat = [];
    
    num1 = size(X,2);
    num2 = size(Y,2);
    num_ft = size(X,1);
    assert(size(X,1)==size(Y,1));

    if strcmp(param.dist_type,'euclidean')
        sq1 = sum(X.*X,1);  
    end
    
    if strcmp(param.dist_type,'cosine')
        prm.type = 'NORM_1';
        [~,X] = norm_features.fit_transform(X,prm);
        [~,Y] = norm_features.fit_transform(Y,prm);
    end
    
    blocks = block_partition(num2,param.block_size);
    
    for i=1:size(blocks,1)
        fprintf('%d\n',i);
        
        subY = Y(:,blocks(i,1):blocks(i,2));
        prod_mat = X'*subY;
        
        if strcmp(param.dist_type,'euclidean')
            sq2 = sum(subY.*subY,1);
            prod_mat = bsxfun(@plus,-2*prod_mat,sq1'+1e-10);
            prod_mat = bsxfun(@plus,prod_mat,sq2);
        end
        
        prod_mat = sparse(prod_mat);
        
        switch param.dist_type
            case 'cosine'
                rank_mat = sort_sparse_mat(prod_mat,'descend');
                
            case 'euclidean'
                rank_mat = sort_sparse_mat(prod_mat,'ascend');
        end
        
        rank_mat(rank_mat>param.trunc_val) = 0;
        rank_mat = spones(rank_mat);
        prod_mat = prod_mat.*rank_mat;
        
        dist_mat = [dist_mat prod_mat];
    end

end

function cparam = complete_param(param)
    cparam = param;

    if ~isfield(param,'trunc_val')
        cparam.trunc_val = 5;
    end
    
    if ~isfield(param,'dist_type')
        cparam.dist_type = 'cosine';
    end
    
    if ~isfield(param,'block_size')
        cparam.block_size = 1000;
    end
end
