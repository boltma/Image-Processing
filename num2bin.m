function arr = num2bin(n, category)
%NUM2BIN Converts n with category into a binary array
%   arr = num2bin(n, category)
    arr = zeros(1, category);
    s = sign(n);
    n = abs(n);
    for i = 1:category
        arr(category - i + 1) = mod(n, 2);
        n = floor(n / 2);
    end
    if s == -1
        arr = 1 - arr;
    end
end

