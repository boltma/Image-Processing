clear; clc; close all;

load data/hall.mat;
block_size = 8;

block = hall_gray(1:block_size, 1:block_size);
block_1 = double(block) - 128;

D = my_dct(block_size);
block_trans = D * double(block) * D';
block_trans(1, 1) = block_trans(1, 1) - 128 * block_size;
block_2 = D' * block_trans * D;

disp(norm(block_1 - block_2));
