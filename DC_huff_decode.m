function DC = DC_huff_decode(c, block_num, DCTAB)
%DC_HUFF_DECODE 此处显示有关此函数的摘要
%   此处显示详细说明
%TODO: Add assertions
    category = 0;
    huff = [];
    idx = 0;
    i = 1;
    DC = zeros(1, block_num);
    while idx < length(c)
        idx = idx + 1;
        huff = [huff c(idx)]; %#ok<AGROW>
        while length(huff) == DCTAB(category+1, 1)
            if all(huff == DCTAB(category+1, 2:length(huff)+1))
                num = c(idx+1:idx+category);
                DC(i) = bin2num(num);
                i = i + 1;
                idx = idx + category;
                category = 0;
                huff = [];
                break
            end
            category = category + 1;
        end
    end
    assert(idx == length(c));
end

