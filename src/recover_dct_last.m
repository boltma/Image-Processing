function [msg, img] = recover_dct_last(H, W, block_size, AC_code, DC_code, ACTAB, DCTAB, QTAB)
%RECOVER_DCT_LAST Recover message from DCT coefficient after last non-zero one
%   [msg, img] = recover_dct_last(H, W, block_size, AC_code, DC_code, ACTAB, DCTAB, QTAB)
    block_num = ceil(H / block_size) * ceil(W / block_size);
    try
        DC_diff = DC_huff_decode(DC_code, block_num, DCTAB);
        DC_diff(1) = -DC_diff(1);
        DC = -cumsum(DC_diff);
        AC = AC_huff_decode(AC_code, block_size, block_num, ACTAB);
    catch
        error('Decode failed.');
    end
    Q = [DC; AC];
    
    % recover message from each column
    msg = zeros(1, size(Q, 2));
    for col = 1:size(Q, 2)
        last = find(Q(:, col) ~= 0, 1, 'last'); % find last non-zero element
        msg(col) = Q(last, col);
    end
    msg = (msg + 1) / 2;
    
    img = block_quant_inv(Q, H, W, block_size, QTAB);
    img = uint8(round(img) + 128);
end

