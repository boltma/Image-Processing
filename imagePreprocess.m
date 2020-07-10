clear; clc;

load data/hall.mat;
grid_size = 8;

grid = hall_gray(1:grid_size, 1:grid_size);
grid_1 = double(grid) - 128;

D = my_dct(grid_size);
grid_trans = D * double(grid) * D';
grid_trans(1, 1) = grid_trans(1, 1) - 128 * grid_size;
grid_2 = D' * grid_trans * D;

disp(norm(grid_1 - grid_2));
