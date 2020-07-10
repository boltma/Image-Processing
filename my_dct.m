function D = my_dct(N)
%MY_DCT Generates DCT matrix with size NxN
%   D = my_dct(N)
A = (0:N-1)' * (1:2:2*N-1) .* (pi / 2 / N);
D = sqrt(2/N) * cos(A);
D(1, :) = sqrt(1/N);
end

