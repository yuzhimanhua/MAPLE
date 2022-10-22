function [trn_X_Xf, trn_X_Y, tst_X_Xf, tst_X_Y, inv_prop] = load_std_xmlc_dataset(dset)

	[X_Xf, X_Y] = load_dataset(dset);
	split = load_split(dset);

	trn_X_Xf = X_Xf(:, split==0);
	trn_X_Y = X_Y(:, split==0);
	tst_X_Xf = X_Xf(:, split==1);
	tst_X_Y = X_Y(:, split==1);

	inv_prop = inv_propensity_wrap( trn_X_Y, dset );
end
