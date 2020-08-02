function P = my_PSNR(img1, img2)
%MY_PSNR Computes PSNR of two images
%   P = my_PSNR(img1, img2)
    MSE = immse(double(img1), double(img2));
    P = 10 * log10(255 ^ 2 / MSE);
end

