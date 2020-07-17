function Q = block_quant(img, QTAB, block_size)
%BLOCK_QUANT Block and quantize image
%   Q = block_quant(img, QTAB, block_size)
    D = my_dct(block_size);
    block_handler = @(block_struct) quant(block_struct.data, QTAB, D); 
    Q = blockproc(img, [block_size, block_size], block_handler, 'PadPartialBlocks', true);
    Q = reshape(Q, block_size^2, []);
end


function GQ = quant(block, QTAB, D)
%QUANT Quantize a block with QTAB and DCT matrix D
%   GQ = quant(block, QTAB, D)
    C = D * block * D';
    CQ = round(C ./ QTAB);
    GQ = zigzag_mat8(CQ)';
end

