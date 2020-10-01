function ori = zigzag_inv_mat8(Z)
%ZIGZAG_INV_MAT8 Generates original array of zigzag array with input size 8
%   Z = zigzag_mat8(input)
    load zigzag8.mat zigzag8_inv;
    ori = reshape(Z(zigzag8_inv), 8, 8);
end

