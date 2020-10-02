function n = bin2num(arr)
%BIN2NUM Converts binary array to n
%   n = bin2num(arr)
    if isempty(arr)
        n = 0;
    else
        s = 1;
        if arr(1) == 0
            arr = 1 - arr;
            s = -1;
        end
        n = bi2de(arr, 'left-msb');
        n = s * n;
    end
end

