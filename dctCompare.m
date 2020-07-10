clear; clc;

load data/hall.mat;
grid_size = 20;

grid = hall_gray(1:grid_size, 1:grid_size);

D = my_dct(grid_size);
grid_trans = D * double(grid) * D';
grid_trans_2 = dct2(grid);

disp(norm(grid_trans - grid_trans_2));
