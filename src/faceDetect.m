clear; clc; close all;

datapath = 'data/Faces';
testimg = double(imread('data/test.jpg'));

L = 3;
N = 2 ^ (3 * L);
v = train_color_hist(datapath, L);
f = detect_face(testimg, L, N, v, 20, 5, 0.4);
f = merge_faces(f, 0.5);

% draw rectangles
figure;
imshow(uint8(testimg));
for n = 1:size(f, 1)
    rectangle('Position', f(n, :), 'EdgeColor', 'r');
end

L = 4;
N = 2 ^ (3 * L);
v = train_color_hist(datapath, L);
f = detect_face(testimg, L, N, v, 20, 5, 0.55);
f = merge_faces(f, 0.5);

figure;
imshow(uint8(testimg));
for n = 1:size(f, 1)
    rectangle('Position', f(n, :), 'EdgeColor', 'r');
end

L = 5;
N = 2 ^ (3 * L);
v = train_color_hist(datapath, L);
f = detect_face(testimg, L, N, v, 20, 5, 0.75);
f = merge_faces(f, 0.5);

figure;
imshow(uint8(testimg));
for n = 1:size(f, 1)
    rectangle('Position', f(n, :), 'EdgeColor', 'r');
end

% rotate clockwise 90 degrees
testimg_rotate = imrotate(testimg, 270);
f = detect_face(testimg_rotate, L, N, v, 20, 5, 0.75);
f = merge_faces(f, 0.5);

figure;
imshow(uint8(testimg_rotate));
for n = 1:size(f, 1)
    rectangle('Position', f(n, :), 'EdgeColor', 'r');
end

% resize to twice as wide
testimg_resize = imresize(testimg, [size(testimg, 1), 2 * size(testimg, 2)]);
f = detect_face(testimg_resize, L, N, v, 20, 5, 0.75);
f = merge_faces(f, 0.5);

figure;
imshow(uint8(testimg_resize));
for n = 1:size(f, 1)
    rectangle('Position', f(n, :), 'EdgeColor', 'r');
end

% adjust color
testimg_adjust = double(imadjust(uint8(testimg), [0.1 0.9]));
f = detect_face(testimg_adjust, L, N, v, 20, 5, 0.75);
f = merge_faces(f, 0.5);

figure;
imshow(uint8(testimg_adjust));
for n = 1:size(f, 1)
    rectangle('Position', f(n, :), 'EdgeColor', 'r');
end
