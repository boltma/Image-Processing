function [H, W, DC_code, AC_code] = JPEG_encode(img, block_size, ACTAB, DCTAB, QTAB)
%JPEG_ENCODE JPEG encode
%   [H, W, DC_code, AC_code] = JPEG_encode(img, block_size, ACTAB, DCTAB, QTAB)
    img = double(img) - 128;
    img_size = size(img);
    H = img_size(1);
    W = img_size(2);
    Q = block_quant(img, block_size, QTAB);
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

