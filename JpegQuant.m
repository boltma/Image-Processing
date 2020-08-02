clear; clc; close all;

load data/hall.mat;
load data/JpegCoeff.mat;

[H, W, DCCODE, ACCODE] = JPEG_encode(hall_gray, 8, ACTAB, DCTAB, QTAB);
img_quant_ori = JPEG_decode(H, W, 8, DCCODE, ACCODE, ACTAB, DCTAB, QTAB);
[H, W, DCCODE, ACCODE] = JPEG_encode(hall_gray, 8, ACTAB, DCTAB, QTAB./2);
img_quant_half = JPEG_decode(H, W, 8, DCCODE, ACCODE, ACTAB, DCTAB, QTAB./2);
figure;
subplot(1, 3, 1);
imshow(hall_gray);
subplot(1, 3, 2);
imshow(img_quant_ori);
subplot(1, 3, 3);
imshow(img_quant_half);

P = PSNR(hall_gray, img_quant_half);
disp(P);
