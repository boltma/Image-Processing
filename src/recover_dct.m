function [msg, img] = recover_dct(H, W, dct_start, dct_len, block_size, DC_code, AC_code, ACTAB, DCTAB, QTAB)
%RECOVER_DCT Recover message from part of DCT
%   [msg, img] = recover_dct(H, W, dct_start, dct_len, block_size, DC_code, AC_code, ACTAB, DCTAB, QTAB)
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
    msg = mod(Q(dct_start:dct_start+dct_len-1, :), 2);
    msg = reshape(msg, 1, []);
    img = block_quant_inv(Q, H, W, block_size, QTAB);
    img = uint8(round(img) + 128);
end

