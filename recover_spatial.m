function msg = recover_spatial(img)
%RECOVER_SPATIAL Recover message from spatial image
%   msg = recover_spatial(img)
    msg = mod(img, 2);
    msg = reshape(msg, 1, []);
end

