clear; clc; close all;

load data/snow.mat;
load data/JpegCoeff.mat;

[H, W, DCCODE, ACCODE] = JPEG_encode(snow, 8, ACTAB, DCTAB, QTAB);
img_decoded = JPEG_decode(H, W, 8, ACCODE, DCCODE, ACTAB, DCTAB, QTAB);
figure;
subplot(1, 2, 1);
imshow(snow);
subplot(1, 2, 2);
imshow(img_decoded);

P = my_PSNR(snow, img_decoded);
disp(P);

compressed = 8 * 2 + length(DCCODE) + length(ACCODE); % need to store H and W
uncompressed = H * W * 8;
ratio = uncompressed / compressed;
disp(ratio);
