clear; clc; close all;

load data/hall.mat;
if ~exist('results', 'dir')
    mkdir results;
end
imshow(hall_color);
hall_color_copy = hall_color;
[width, height, channel] = size(hall_color);

radius = min(width, height) / 2;
center = [width/2 + 0.5, height/2 + 0.5];
theta = 0:1:359;
x = max(min(round(center(1) + radius * cos(theta)), width), 1);
y = max(min(round(center(2) + radius * sin(theta)), height), 1);
idx = sub2ind(size(hall_color), x, y, ones(1,360));
hall_color(idx) = 255;
imwrite(hall_color, 'results/hallCircle.png');

[xcord, ycord] = meshgrid(1:height, 1:width, 1:channel);
circle_mask = (xcord(:, :, 1) - center(2)) .^ 2 + (ycord(:, :, 1) - center(1)) .^ 2 < radius ^ 2;
mask_color = uint8([255, 0, 0]);
hall_R = hall_color_copy(:, :, 1);
hall_G = hall_color_copy(:, :, 2);
hall_B = hall_color_copy(:, :, 3);
hall_R = uint8(circle_mask) .* mask_color(1) + uint8(~circle_mask) .* hall_R;
hall_G = uint8(circle_mask) .* mask_color(2) + uint8(~circle_mask) .* hall_G;
hall_B = uint8(circle_mask) .* mask_color(3) + uint8(~circle_mask) .* hall_B;
hall_circle(:, :, 1) = hall_R;
hall_circle(:, :, 2) = hall_G;
hall_circle(:, :, 3) = hall_B;
imwrite(hall_circle, 'results/hallCircle2.png');

block_size = 24;
xgrid = mod(ceil(xcord / block_size), 2); % use ceil instead of floor
ygrid = mod(ceil(ycord / block_size), 2);
grid_mask = xor(xgrid, ygrid);
hall_grid = hall_color_copy .* uint8(grid_mask);
imwrite(hall_grid, 'results/hallGrid.png');
