clear; clc;

load data/hall.mat;
load data/JpegCoeff.mat;
if ~exist('results', 'dir')
    mkdir results;
end

[H, W, DCCODE, ACCODE] = JPEG_encode(hall_gray, 8, ACTAB, DCTAB, QTAB);
save results/jpegcodes.mat H W DCCODE ACCODE;
