function [H, W, DC_code, AC_code] = hide_dct(img, msg, dct_start, dct_len, block_size, ACTAB, DCTAB, QTAB)
%HIDE_DCT Hide message in part of DCT
%   [H, W, DC_code, AC_code] = hide_dct(img, msg, dct_start, dct_len, block_size, ACTAB, DCTAB, QTAB)
    img = double(img) - 128;
    img_size = size(img);
    H = img_size(1);
    W = img_size(2);
    Q = block_quant(img, block_size, QTAB);
    
    padlen = size(Q, 2) * dct_len - length(msg);
    assert(dct_start + dct_len - 1 <= block_size ^ 2);
    assert(padlen >= 0);
    msg = [msg zeros(1, padlen)];
    msg = reshape(msg, dct_len, []);
    Q(dct_start:dct_start+dct_len-1, :) = 2 * floor(Q(dct_start:dct_start+dct_len-1, :) / 2) + msg;
    
    DC = Q(1, :);
    AC = Q(2:end, :);
    DC_diff = [DC(1), -diff(DC)];
    DC_code = [];
    AC_code = [];
    for idx = 1:length(DC_diff)
        DC_code = [DC_code DC_huff(DC_diff(idx), DCTAB)]; %#ok<AGROW>
        AC_code = [AC_code AC_huff(AC(:, idx), ACTAB)]; %#ok<AGROW>
    end
end

