function h = color_hist(img, L, N)
%COLOR_HIST 此处显示有关此函数的摘要
%   此处显示详细说明
%   img should be double type
    img = floor(img / 2 ^ (8 - L)); % floor instead of round
    color = img(:, :, 1) * 2 ^ (2 * L) + img(:, :, 2) * 2 ^ L + img(:, :, 3);
    h = histcounts(color, 0:N) / numel(color);
end

