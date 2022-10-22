function wts = inv_propensity(X_Y,A,B)

	%% Returns inverse propensity weights
	%% A,B are parameters of the propensity model. Refer to paper for more details.
	%% A,B values used for different datasets in paper:
    %%	Wikipedia-LSHTC: A=0.5,  B=0.4
	%%	Amazon:          A=0.6,  B=2.6
	%%	Other:			 A=0.55, B=1.5

	num_inst = size(X_Y,2);
	freqs = sum(X_Y,2);

	C = (log(num_inst)-1)*(B+1)^A;
	wts = 1 + C*(freqs+B).^-A;
end
