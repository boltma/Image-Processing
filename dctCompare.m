clear; clc; close all;

load data/hall.mat;
block_size = 20;

block = hall_gray(1:block_size, 1:block_size);

D = my_dct(block_size);
block_trans = D * double(block) * D';
block_trans_2 = dct2(block);

disp(norm(block_trans - block_trans_2));
