function n = bin2num(arr)
%BIN2NUM 此处显示有关此函数的摘要
%   此处显示详细说明
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

