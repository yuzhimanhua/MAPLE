function varargout = load_dataset(dset)
    %%% arguments allowed in varargout = [X_Xf,X_Y,Y_Yf,X,Xf,Y,Yf];

    global EXP_DIR;

	dset_dir = [ EXP_DIR filesep 'Datasets' filesep dset ];
	
	for i=1:nargout
		switch i
			case 1
				s = [ dset_dir filesep 'X_Xf.mat' ];
			case 2
				s = [ dset_dir filesep 'X_Y.mat' ];
			case 3
				s = [ dset_dir filesep 'Y_Yf.mat' ];
			case 4
				s = [ dset_dir filesep 'X.txt' ];
			case 5
				s = [ dset_dir filesep 'Xf.txt' ];
			case 6
				s = [ dset_dir filesep 'Y.txt' ];
			case 7				
				s = [ dset_dir filesep 'Yf.txt' ];
		end

		if exist(s,'file')==2
			if i<=3
				load(s);
				switch i
					case 1
						varargout{i} = X_Xf;
					case 2
						varargout{i} = X_Y;
					case 3
						varargout{i} = Y_Yf;
				end
			else
				varargout{i} = read_string_file(s);
			end
		else
			varargout{i} = [];
		end
	end
end
