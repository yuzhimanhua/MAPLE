% verified = yes

function T = read_string_file(s)
	fid = fopen(s,'r');
	T = textscan(fid,'%s','Delimiter','\n');
	T = T{1};
	fclose(fid);
end
