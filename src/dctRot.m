clear; clc; close all;

load data/hall.mat;
block_size = 8;

block = hall_gray(end-block_size+1:end, end-block_size+1:end);

D = my_dct(block_size);
block_trans = D * (double(block) - 128) * D';       % preprocess
block_trans_transpose = block_trans';
block_trans_rot90 = rot90(block_trans);
block_trans_rot180 = rot90(block_trans_rot90);      % rotate 180 degrees by rotating rot90 again
block_transpose = D' * block_trans_transpose * D;
block_rot90 = D' * block_trans_rot90 * D;
block_rot180 = D' * block_trans_rot180 * D;

figure;
subplot(2, 2, 1);
imshow(block);
title('original');
subplot(2, 2, 2);
imshow(uint8(block_transpose + 128));
title('transpose');
subplot(2, 2, 3);
imshow(uint8(block_rot90 + 128));
title('rotate90');
subplot(2, 2, 4);
imshow(uint8(block_rot180 + 128));
title('rotate180');
