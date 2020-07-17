function [DC_code, AC_code] = JPEG_encode(img, ACTAB, DCTAB, QTAB)
%JPEG_ENCODE JPEG encode
%   [DC_code, AC_code] = JPEG_encode(img, ACTAB, DCTAB, QTAB)
    block_size = 8;
    img = double(img) - 128;
    Q = block_quant(img, QTAB, block_size);
    DC = Q(1, :);
    AC = Q(2:end, :);
    DC_diff = [DC(1), -diff(DC)];
    DC_code = [];
    AC_code = [];
    for i = 1:length(DC_diff)
        DC_code = [DC_code DC_huff(DC_diff(i), DCTAB)]; %#ok<AGROW>
        AC_code = [AC_code AC_huff(AC(:, i), ACTAB)]; %#ok<AGROW>
    end
end

