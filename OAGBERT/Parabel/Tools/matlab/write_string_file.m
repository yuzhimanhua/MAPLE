% verified = no

function write_string_file(vec, s)
	% write_string_file([string cell array], [file name])

	fid = fopen(s,'w');
	for i=1:numel(vec)
		fprintf(fid, '%s\n', vec{i});
	end
	fclose(fid);
end
