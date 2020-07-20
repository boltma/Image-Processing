clear; clc;

load data/hall.mat;
load data/JpegCoeff.mat;
load results/jpegcodes.mat

img_decoded = JPEG_decode(H, W, 8, DCCODE, ACCODE, ACTAB, DCTAB, QTAB);
figure;
subplot(1, 2, 1);
imshow(hall_gray);
subplot(1, 2, 2);
imshow(img_decoded);
