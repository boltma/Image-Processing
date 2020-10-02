function h = color_hist(img, L, N)
%COLOR_HIST Compute color histogram of img with L and N
%   h = color_hist(img, L, N)
%   img should be double type
    img = floor(img / 2 ^ (8 - L)); % floor instead of round
    color = img(:, :, 1) * 2 ^ (2 * L) + img(:, :, 2) * 2 ^ L + img(:, :, 3);
    h = histcounts(color, 0:N) / numel(color);
end

