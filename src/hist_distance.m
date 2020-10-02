function d = hist_distance(h1, h2)
%HIST_DISTANCE compute histogram distance of h1 and h2
%   d = hist_distance(h1, h2)
    d = 1 - sum(sqrt(h1 .* h2));
end

