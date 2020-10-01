clear; clc; close all;

load data/snow.mat;
load data/JpegCoeff.mat;

[H, W, DCCODE, ACCODE] = JPEG_encode(snow, 8, ACTAB, DCTAB, QTAB);
img_decoded = JPEG_decode(H, W, 8, DCCODE, ACCODE, ACTAB, DCTAB, QTAB);
figure;
subplot(1, 2, 1);
imshow(snow);
subplot(1, 2, 2);
imshow(img_decoded);

P = my_PSNR(snow, img_decoded);
disp(P);
