clear; clc; close all;

datapath = 'data/Faces';

h3 = train_color_hist(datapath, 3);
h4 = train_color_hist(datapath, 4);
h5 = train_color_hist(datapath, 5);

figure;
subplot(3, 1, 1);
plot(0:2^(3*3)-1, h3);
subplot(3, 1, 2);
plot(0:2^(3*4)-1, h4);
subplot(3, 1, 3);
plot(0:2^(3*5)-1, h5);
