function d = hist_distance(h1, h2)
%HIST_DISTANCE 此处显示有关此函数的摘要
%   此处显示详细说明
    d = 1 - sum(sqrt(h1 .* h2));
end

