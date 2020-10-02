clear; clc; close all;

load data/hall.mat;
load data/JpegCoeff.mat;

% dct hide method 1
rng(0);
msg = randi([0 1], 1, 64 * 315);

[H, W, DCCODE, ACCODE] = hide_dct(hall_gray, msg, 1, 64, 8, ACTAB, DCTAB, QTAB);
[msg_decoded, hall_decoded] = recover_dct(H, W, 1, 64, 8, ACCODE, DCCODE, ACTAB, DCTAB, QTAB);

figure;
subplot(1, 2, 1);
imshow(hall_gray);
subplot(1, 2, 2);
imshow(hall_decoded);

% compute compression rate
compressed = 8 * 2 + length(DCCODE) + length(ACCODE);
uncompressed = H * W * 8;
ratio = uncompressed / compressed;

disp(sum(msg ~= msg_decoded)); % check if all messages are recovered
disp(my_PSNR(hall_gray, hall_decoded));
disp(ratio);

% dct hide method 2
msg2 = randi([0 1], 1, 6 * 315);

% hide from 16 to 21
[H, W, DCCODE, ACCODE] = hide_dct(hall_gray, msg2, 16, 6, 8, ACTAB, DCTAB, QTAB);
[msg_decoded_2, hall_decoded_2] = recover_dct(H, W, 16, 6, 8, ACCODE, DCCODE, ACTAB, DCTAB, QTAB);

figure;
subplot(1, 2, 1);
imshow(hall_gray);
subplot(1, 2, 2);
imshow(hall_decoded_2);

compressed = 8 * 2 + length(DCCODE) + length(ACCODE);
uncompressed = H * W * 8;
ratio = uncompressed / compressed;

disp(sum(msg2 ~= msg_decoded_2));
disp(my_PSNR(hall_gray, hall_decoded_2));
disp(ratio);

% dct hide method 3
msg3 = randi([0 1], 1, 315);

[H, W, DCCODE, ACCODE] = hide_dct_last(hall_gray, msg3, 8, ACTAB, DCTAB, QTAB);
[msg_decoded_3, hall_decoded_3] = recover_dct_last(H, W, 8, ACCODE, DCCODE, ACTAB, DCTAB, QTAB);

figure;
subplot(1, 2, 1);
imshow(hall_gray);
subplot(1, 2, 2);
imshow(hall_decoded_3);

compressed = 8 * 2 + length(DCCODE) + length(ACCODE);
uncompressed = H * W * 8;
ratio = uncompressed / compressed;

disp(sum(msg3 ~= msg_decoded_3));
disp(my_PSNR(hall_gray, hall_decoded_3));
disp(ratio);
