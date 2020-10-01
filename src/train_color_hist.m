function h = train_color_hist(datapath, L)
%TRAIN_COLOR_HIST 此处显示有关此函数的摘要
%   此处显示详细说明
    assert(L <= 8);
    face_imgs = dir(datapath);
    N = 2 ^ (3 * L);
    h = zeros(1, N);
    for idx = 3:length(face_imgs) % first and second elements of face_imgs are . and ..
        img_path = fullfile(datapath, face_imgs(idx).name);
        face = double(imread(img_path));
        freq = color_hist(face, L, N);
        h = h + freq;
    end
    h = h / (length(face_imgs) - 2);
end

