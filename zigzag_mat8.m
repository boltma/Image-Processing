function Z = zigzag_mat8(input)
%ZIGZAG_MAT8 Generates zigzag array with input size 8
%   Z = zigzag_mat8(input)
    load zigzag8.mat zigzag8;
    Z = input(zigzag8);
end

