clear; clc; close all;

load data/hall.mat;
load data/JpegCoeff.mat;

% generate random message, same length as hall_gray
rng(0);
msg = randi([0 1], 1, numel(hall_gray));

hall_msg = hide_spatial(hall_gray, msg);
[H, W, DCCODE, ACCODE] = JPEG_encode(hall_msg, 8, ACTAB, DCTAB, QTAB);
hall_decoded = JPEG_decode(H, W, 8, ACCODE, DCCODE, ACTAB, DCTAB, QTAB);
msg_decoded = recover_spatial(hall_decoded);

figure;
subplot(1, 3, 1);
imshow(hall_gray);
subplot(1, 3, 2);
imshow(hall_msg);
subplot(1, 3, 3);
imshow(hall_decoded);

disp(sum(msg == msg_decoded));
disp(sum(msg ~= msg_decoded));
