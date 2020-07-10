clear; clc;

load data/hall.mat;
grid_size = 8;

grid = hall_gray(end-grid_size+1:end, end-grid_size+1:end);

D = my_dct(grid_size);
grid_trans = D * (double(grid) - 128) * D';
grid_trans_left = grid_trans;
grid_trans_right = grid_trans;
grid_trans_left(:, 1:4) = 0;
grid_trans_right(:, end-3:end) = 0;
grid_left = D' * grid_trans_left * D;
grid_right = D' * grid_trans_right * D;

figure;
subplot(1, 3, 1);
imshow(grid);
subplot(1, 3, 2);
imshow(uint8(grid_left + 128));
subplot(1, 3, 3);
imshow(uint8(grid_right + 128));
