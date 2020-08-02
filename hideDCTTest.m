clear; clc; close all;

load data/hall.mat;
load data/JpegCoeff.mat;

rng(0);
msg = randi([0 1], 1, 64 * 315);

[H, W, DCCODE, ACCODE] = hide_dct(hall_gray, msg, 1, 64, 8, ACTAB, DCTAB, QTAB);
[msg_decoded, hall_decoded] = recover_dct(H, W, 1, 64, 8, DCCODE, ACCODE, ACTAB, DCTAB, QTAB);

figure;
subplot(1, 2, 1);
imshow(hall_gray);
subplot(1, 2, 2);
imshow(hall_decoded);

disp(sum(msg == msg_decoded));
disp(PSNR(hall_gray, hall_decoded));

msg2 = randi([0 1], 1, 8 * 315);

[H, W, DCCODE, ACCODE] = hide_dct(hall_gray, msg2, 16, 8, 8, ACTAB, DCTAB, QTAB);
[msg_decoded_2, hall_decoded_2] = recover_dct(H, W, 16, 8, 8, DCCODE, ACCODE, ACTAB, DCTAB, QTAB);

figure;
subplot(1, 2, 1);
imshow(hall_gray);
subplot(1, 2, 2);
imshow(hall_decoded_2);

disp(sum(msg2 == msg_decoded_2));
disp(PSNR(hall_gray, hall_decoded_2));

msg3 = randi([0 1], 1, 315);

[H, W, DCCODE, ACCODE] = hide_dct_last(hall_gray, msg3, 8, ACTAB, DCTAB, QTAB);
[msg_decoded_3, hall_decoded_3] = recover_dct_last(H, W, 8, DCCODE, ACCODE, ACTAB, DCTAB, QTAB);

figure;
subplot(1, 2, 1);
imshow(hall_gray);
subplot(1, 2, 2);
imshow(hall_decoded_3);

disp(sum(msg3 == msg_decoded_3));
disp(PSNR(hall_gray, hall_decoded_3));
