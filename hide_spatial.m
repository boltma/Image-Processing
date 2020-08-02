function img_new = hide_spatial(img, msg)
%HIDE_SPATIAL Hide message into spatial
%   img_new = hide_spatial(img, msg)
    padlen = numel(img) - length(msg);
    assert(padlen >= 0);
    msg = [msg zeros(1, padlen)];
    msg = reshape(msg, size(img));
    img_new = uint8(2 * floor(double(img) / 2) + msg);
end

