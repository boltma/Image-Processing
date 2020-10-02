function output = zigzag_loop(input)
%ZIGZAG_LOOP Generates zigzag array using loop
%   output = zigzag_loop(input)
    [row, col] = size(input);
    output = zeros(1, row * col);
    cnt = 1;
    for u = 2:row+col   % u equals to current row num + col num
        if mod(u, 2) == 0
            % from up to down
            for v = max(u - row, 1):min(u - 1, col)
                output(cnt) = input(u - v, v);
                cnt = cnt + 1;
            end
        else
           % from down to up
           for v = max(u - col, 1):min(u - 1, row)
                output(cnt) = input(v, u - v);
                cnt = cnt + 1;
           end
        end
    end
end

