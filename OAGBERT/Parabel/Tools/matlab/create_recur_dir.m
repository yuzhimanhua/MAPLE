function create_recur_dir( dirname )
	if isunix
		system([ 'mkdir -p ' dirname ]);
	elseif ispc
		error('Not yet implemented\n');
	else
		error('Unusual os type, not yet implemented\n');
	end
end
