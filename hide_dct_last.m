function [H, W, DC_code, AC_code] = hide_dct_last(img, msg, block_size, ACTAB, DCTAB, QTAB)
%HIDE_DCT_LAST Hide message after last non-zero coefficient of DCT
%   [H, W, DC_code, AC_code] = hide_dct_last(img, msg, block_size, ACTAB, DCTAB, QTAB)
    img = double(img) - 128;
    img_size = size(img);
    H = img_size(1);
    W = img_size(2);
    Q = block_quant(img, block_size, QTAB);
    
    padlen = size(Q, 2) - length(msg);
    assert(padlen >= 0);
    msg = [msg zeros(1, padlen)];
    msg = 2 * msg - 1; % map [0, 1] to [-1, 1]
    for col = 1:size(Q, 2)
        last = find(Q(:, col) ~= 0, 1, 'last');
        last = min(last + 1, size(Q, 1));
        Q(last, col) = msg(col);
    end
    
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

