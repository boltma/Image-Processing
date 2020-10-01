clear; clc; close all;

datapath = 'data/Faces';
testimg = double(imread('data/test.jpg'));

L = 5;
N = 2 ^ (3 * L);
v = train_color_hist(datapath, L);
f = detect_face(testimg, L, N, v, 20, 5, 0.75);
f = merge_faces(f, 0.5);

imshow(uint8(testimg));
for n = 1:size(f, 1)
    rectangle('Position', f(n, :), 'EdgeColor', 'r');
end
