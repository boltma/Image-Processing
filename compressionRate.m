clear; clc;

load data/hall.mat;
load results/jpegcodes.mat;

compressed = 8 * 2 + length(DCCODE) + length(ACCODE);
uncompressed = H * W * 8;
ratio = uncompressed / compressed;
disp(ratio);
