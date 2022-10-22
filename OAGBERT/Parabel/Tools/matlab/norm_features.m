classdef norm_features
	
	methods(Static)
		function model = fit(mat,params)
			type = params.type;
			model.type = type;

			switch type
				case 'NORM_1'

				case 'MEAN_0_STD_1'
					dim = size(mat,1);
					model.means = mean(mat,2);
					cmat = bsxfun(@minus,mat,model.means);
					model.stds = std(cmat,1,2);

					
				case 'TF_IDF'
					if isfield(params,'beta')
						beta = params.beta;
					else
						beta = 0;
					end

					num = size(mat,2);
					dim = size(mat,1);

					freqs = sum(mat>0,2);
					model.idfs = beta + log(repmat(num+1,dim,1)./(freqs+1));
			end				


		end

		function tmat = transform(mat,model)
			type = model.type;

			switch type
				case 'NORM_1'
					num = size(mat,2);
    					norms = sqrt(sum(mat.*mat,1));
				    	norms(norms==0) = 1;
					inv_norms = ones(1,num)./norms;
					tmat = mat*spdiags(inv_norms',0,num,num);
 
				case 'MEAN_0_STD_1'
					dim = size(mat,1);

					means = model.means;
					tmat = bsxfun(@minus,mat,means);

					stds = model.stds;
					stds(stds==0) = 1;
					inv_stds = ones(dim,1)./stds;
					tmat = spdiags(inv_stds,0,dim,dim)*tmat;

				case 'TF_IDF'
					dim = size(mat,1);

					idfs = model.idfs;
					tmat = spdiags(idfs,0,dim,dim)*mat;

			end
		end

		function [model,tmat] = fit_transform(mat,params)
			model = norm_features.fit(mat,params);
			tmat = norm_features.transform(mat,model);
		end
	end
end

