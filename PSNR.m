function P = PSNR(img1, img2)
%PSNR Computes PSNR of two images
%   P = PSNR(img1, img2)
    MSE = immse(double(img1), double(img2));
    P = 10 * log10(255 ^ 2 / MSE);
end

