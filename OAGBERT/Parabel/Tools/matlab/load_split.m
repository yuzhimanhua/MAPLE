function split = load_split(dset,varargin)
    global EXP_DIR;
    sno = 0;
    if nargin>1
        sno = varargin{1};
    end
    split = csvread( [ EXP_DIR filesep 'Datasets' filesep dset filesep 'split.' num2str(sno) '.txt' ] );
end
