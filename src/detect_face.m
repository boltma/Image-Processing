function f = detect_face(img, L, N, v, len, step, th)
%DETECT_FACE detect face with vector v, and given length, step and
%threshold
%   returns vector k * 4, each row is [col row len len], where [col row] is
%   upper-left corner
%   f = detect_face(img, L, N, v, len, step, th)
    [h, w, ~] = size(img);
    f = [];
    for row = 1:step:h-len+1 % make sure cropped img does not exceed border
        for col = 1:step:w-len+1
            u = color_hist(img(row:row+len-1, col:col+len-1, :), L, N);
            d = hist_distance(u, v);
            if d < th
                f = [f; [col row len len]]; %#ok<AGROW>
            end
        end
    end
end

