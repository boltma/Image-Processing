function img = JPEG_decode(H, W, block_size, AC_code, DC_code, ACTAB, DCTAB, QTAB)
%JPEG_DECODE JPEG decode
%   img = JPEG_decode(H, W, block_size, AC_code, DC_code, ACTAB, DCTAB, QTAB)
    block_num = ceil(H / block_size) * ceil(W / block_size);
    try
        DC_diff = DC_huff_decode(DC_code, block_num, DCTAB);
        DC_diff(1) = -DC_diff(1);
        DC = -cumsum(DC_diff); % add all previous DC_diff
        AC = AC_huff_decode(AC_code, block_size, block_num, ACTAB);
    catch
        error('Decode failed.');
    end
    Q = [DC; AC];
    img = block_quant_inv(Q, H, W, block_size, QTAB);
    img = uint8(round(img) + 128);
end

