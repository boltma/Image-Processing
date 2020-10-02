function AC = AC_huff_decode(c, block_size, block_num, ACTAB)
%AC_HUFF_DECODE Decode AC Huffman code
%   AC = AC_huff_decode(c, block_size, block_num, ACTAB)
    EOB = [1, 0, 1, 0];
    ZRL = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1];
    idx = 1;
    AC = zeros(block_size ^ 2 - 1, block_num);
    i = 1;
    j = 1;
    while idx < length(c)
        if idx + 3 <= length(c) && all(c(idx:idx+3) == EOB)
            % check if next codes are EOB
            AC(j:end, i) = 0;
            i = i + 1;
            j = 1;
            idx = idx + 4;
        elseif idx + 10 <= length(c) && all(c(idx:idx+10) == ZRL)
            % check if next codes are ZRL
            assert(j <= 49);    % currently less than 49 zeroes
            AC(j:j+15, i) = 0;
            j = j + 16;
            idx = idx + 11;
        else
            flag = 0; % use flag and assert to make sure that the code is inside ACTAB
            for k = 1:size(ACTAB, 1)
                % go throught all codes in ACTAB to find
                len = ACTAB(k, 3);
                if idx + len - 1 <= length(c) && all(c(idx:idx+len-1) == ACTAB(k, 4:3+len))
                    run = ACTAB(k, 1);
                    category = ACTAB(k, 2);
                    idx = idx + len;
                    assert(idx + category - 1 <= length(c));
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

