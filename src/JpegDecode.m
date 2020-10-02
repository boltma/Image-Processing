clear; clc; close all;

load data/hall.mat;
load data/JpegCoeff.mat;
load results/jpegcodes.mat

img_decoded = JPEG_decode(H, W, 8, ACCODE, DCCODE, ACTAB, DCTAB, QTAB);
figure;
subplot(1, 2, 1);
imshow(hall_gray);
subplot(1, 2, 2);
imshow(img_decoded);

% save as bmp file to ensure image not compressed
imwrite(hall_gray, 'results/hall_gray.bmp');
imwrite(img_decoded, 'results/hall_jpeg.bmp');

P = my_PSNR(hall_gray, img_decoded);
P2 = psnr(hall_gray, img_decoded);
disp(P);
disp(P - P2);
