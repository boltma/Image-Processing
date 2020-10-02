function c = AC_huff(block, ACTAB)
%AC_HUFF Get AC Huffman code
%   c = AC_huff(block, ACTAB)
    EOB = [1, 0, 1, 0];
    ZRL = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1];
    idx = 1;
    run = 0;
    c = [];
    while idx <= length(block)
        % find next non-zero element
        if block(idx) == 0
            run = run + 1;
        else
            if run >= 16
                % more than 16 zeroes, use ZRL
                c = [c repmat(ZRL, 1, floor(run / 16))]; %#ok<AGROW>
                run = mod(run, 16);
            end
            e = block(idx);
            assert(abs(e) <= 1023 && run <= 15);
            category = max(0, floor(log2(abs(e))) + 1);
            huff = ACTAB(run * 10 + category, :);
            len = huff(3);
            h = huff(4:3+len);
            c = [c h num2bin(e)]; %#ok<AGROW>
            run = 0;
        end
        idx = idx + 1;
    end
    c = [c EOB];
end

