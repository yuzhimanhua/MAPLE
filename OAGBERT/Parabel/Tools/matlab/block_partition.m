function parts = block_partition(num,block)
    parts = [];
    a = 1;
    
    while a<=num
        if a+block-1 <= num
            b = a+block-1;
        else
            b = num;
        end
        
        parts = [parts;a b];
        a = b+1;
    end
end
