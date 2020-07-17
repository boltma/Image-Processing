clear; clc;

block_size = 8;
zigzag8 = zigzag_loop(reshape(1:block_size^2, block_size, block_size));
save('zigzag8.mat', 'zigzag8');

test = [0, 10, 2, 0, 0, 0, 0, 0;
        3, 0, 0, 0, 0, 1, 0, 0;
        0, 0, 0, 0, 0, 0, 0, 0;
        0, 0, 0, 0, 0, 0, 0, 0;
        0, 0, 0, 0, 0, 0, 0, 0;
        0, 0, 0, 0, 0, 0, 0, 0;
        0, 0, 0, 0, 0, 0, 0, 0;
        0, 0, 0, 0, 0, 0, 0, 0];
test_zigzag = zigzag_mat8(test);
test_zigzag_goal = [0, 10, 3, 0, 0, 2, zeros(1, 20), 1, zeros(1, 37)];
assert(all(all(test_zigzag == test_zigzag_goal)));
