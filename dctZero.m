clear; clc;

load data/hall.mat;
block_size = 8;

block = hall_gray(end-block_size+1:end, end-block_size+1:end);

D = my_dct(block_size);
block_trans = D * (double(block) - 128) * D';
block_trans_left = block_trans;
block_trans_right = block_trans;
block_trans_left(:, 1:4) = 0;
block_trans_right(:, end-3:end) = 0;
block_left = D' * block_trans_left * D;
block_right = D' * block_trans_right * D;

figure;
subplot(1, 3, 1);
imshow(block);
subplot(1, 3, 2);
imshow(uint8(block_left + 128));
subplot(1, 3, 3);
imshow(uint8(block_right + 128));
