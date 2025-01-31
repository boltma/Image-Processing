function s_merged = merge_faces(s, th)
%MERGE_FACES merge rectangles if IoMin greater than threshold
%   s_merged = merge_faces(s, th)
    len = size(s, 1);
    flags = zeros(1, len); % shows whether the face is merged
    for m = 1:len
        for n = 1:len
            if m == n || flags(m) || flags(n)
                continue
            end
            rec1 = s(m, :);
            rec2 = s(n, :);
            u = IoMin(rec1, rec2);
            if u > th
                % could be merged, computer upper-left corner and width and
                % height
                flags(n) = 1;
                s(m, 1) = min(rec1(1), rec2(1));
                s(m, 2) = min(rec1(2), rec2(2));
                s(m, 3) = max(rec1(1) + rec1(3), rec2(1) + rec2(3)) - s(m, 1);
                s(m, 4) = max(rec1(2) + rec1(4), rec2(2) + rec2(4)) - s(m, 2);
            end
        end
    end
    
    s_merged = zeros(sum(~flags), 4);
    cnt = 1;
    for m = 1:len
        if ~flags(m)
            s_merged(cnt, :) = s(m, :);
            cnt = cnt + 1;
        end
    end
end


function u = IoMin(rec1, rec2)
%IOMIN compute intersection over minimum area
%   u = IoMin(rec1, rec2)
    area1 = rec1(3) * rec1(4);
    area2 = rec2(3) * rec2(4);
    area_inter = rectint(rec1, rec2);
    u = area_inter / min(area1, area2);
end

