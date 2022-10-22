function wts = inv_propensity_wrap(X_Y,dset)

	%% Returns inverse propensity weights
	%% A,B are parameters of the propensity model. Refer to paper for more details.
	%% A,B values used for different datasets in paper:
    %%	Wikipedia-LSHTC: A=0.5,  B=0.4
	%%	Amazon:          A=0.6,  B=2.6
	%%	Other:			 A=0.55, B=1.5

	switch dset
		case 'Wikipedia-LSHTC'
			A = 0.5; B = 0.4;

		case 'Amazon'
			A = 0.6; B = 2.6;

		otherwise
			A = 0.55; B = 1.5;
	end

	wts = inv_propensity(X_Y, A, B);
end
