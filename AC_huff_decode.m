function AC = AC_huff_decode(c, block_size, block_num, ACTAB)
%AC_HUFF_DECODE 此处显示有关此函数的摘要
%   此处显示详细说明
    EOB = [1, 0, 1, 0];
    ZRL = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1];
    idx = 1;
    AC = zeros(block_size ^ 2 - 1, block_num);
    i = 1;
    j = 1;
    while idx < length(c)
        if idx + 3 <= length(c) && all(c(idx:idx+3) == EOB)
            AC(j:end, i) = 0;
            i = i + 1;
            j = 1;
            idx = idx + 4;
        elseif idx + 10 <= length(c) && all(c(idx:idx+10) == ZRL)
            assert(j <= 49);
            AC(j:j+15, i) = 0;
            j = j + 16;
            idx = idx + 11;
        else
            flag = 0;
            for k = 1:size(ACTAB, 1)
                len = ACTAB(k, 3);
                if idx + len - 1 <= length(c) && all(c(idx:idx+len-1) == ACTAB(k, 4:3+len))
                    run = ACTAB(k, 1);
                    category = ACTAB(k, 2);
                    idx = idx + len;
                    assert(idx + category - 1 <= length(c))
                    num = c(idx:idx+category-1);
                    idx = idx + category;
                    AC(j:j+run-1, i) = 0;
                    AC(j+run, i) = bin2num(num);
                    j = j + run + 1;
                    assert(j <= 64);
                    flag = 1;
                    break
                end
            end
            assert(flag == 1);
        end
    end
    assert(idx == length(c) + 1);
end

