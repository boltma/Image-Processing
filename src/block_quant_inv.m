function img = block_quant_inv(Q, H, W, block_size, QTAB)
%BLOCK_QUANT_INV Inversion of block and quantize image
%   img = block_quant_inv(Q, H, W, block_size, QTAB)
    Q = reshape(Q, [], ceil(H / block_size))';
    D = my_dct(block_size);
    Q_handler = @(GQ) quant_inv(GQ.data, QTAB, D); 
    img = blockproc(Q, [1, block_size^2], Q_handler);
    img = img(1:H, 1:W);
end


function block = quant_inv(GQ, QTAB, D)
%QUANT_INV Inversion of quantization
%   block = quant_inv(Q, QTAB, D)
    C = zigzag_inv_mat8(GQ) .* QTAB;
    block = D' * C * D;
end

