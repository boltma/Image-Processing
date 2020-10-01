function arr = num2bin(n)
%NUM2BIN Converts n with category into a binary array
%   arr = num2bin(n, category)
    if n == 0
        arr = [];
    else
        s = sign(n);
        n = abs(n);
        arr = de2bi(n, 'left-msb');
        if s == -1
            arr = 1 - arr;
        end
    end
end

