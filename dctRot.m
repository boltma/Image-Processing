clear; clc;

load data/hall.mat;
grid_size = 8;

grid = hall_gray(end-grid_size+1:end, end-grid_size+1:end);

D = my_dct(grid_size);
grid_trans = D * (double(grid) - 128) * D';
grid_trans_transpose = grid_trans';
grid_trans_rot90 = rot90(grid_trans);
grid_trans_rot180 = rot90(grid_trans_rot90);
grid_transpose = D' * grid_trans_transpose * D;
grid_rot90 = D' * grid_trans_rot90 * D;
grid_rot180 = D' * grid_trans_rot180 * D;

figure;
subplot(2, 2, 1);
imshow(grid);
subplot(2, 2, 2);
imshow(uint8(grid_transpose + 128));
subplot(2, 2, 3);
imshow(uint8(grid_rot90 + 128));
subplot(2, 2, 4);
imshow(uint8(grid_rot180 + 128));
