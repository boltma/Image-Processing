clear; clc;

load hall.mat;
mkdir Results;
imshow(hall_color);
hall_color_copy = hall_color;
[width, height, channel] = size(hall_color);
radius = min(width, height) / 2;
center = [width/2 + 0.5, height/2 + 0.5];
% h = viscircles(center, radius, 'color', 'r');
theta = 0:1:359;
x = max(min(round(center(1) + radius * cos(theta)), width), 1);
y = max(min(round(center(2) + radius * sin(theta)), height), 1);
idx = sub2ind(size(hall_color), x, y, ones(1,360));
hall_color(idx) = 255;
imwrite(hall_color, 'Results/hallCircle.png');

grid_size = 24;
[xgrid, ygrid] = meshgrid(1:height, 1:width, 1:channel);
xgrid = mod(ceil(xgrid / grid_size), 2); % use ceil instead of floor
ygrid = mod(ceil(ygrid / grid_size), 2);
mask = xor(xgrid, ygrid);
hall_grid = hall_color_copy .* uint8(mask);
imwrite(hall_grid, 'Results/hallGrid.png');
