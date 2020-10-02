function Q = block_quant(img, block_size, QTAB)
%BLOCK_QUANT Block and quantize image
%   Q = block_quant(img, block_size, QTAB)
    D = my_dct(block_size);
    block_handler = @(block_struct) quant(block_struct.data, QTAB, D); 
    Q = blockproc(img, [block_size, block_size], block_handler, 'PadPartialBlocks', true);
    Q = reshape(Q', block_size^2, []);  % transpose Q and then reshape to get row wise DC
end


function GQ = quant(block, QTAB, D)
%QUANT Quantize a block with QTAB and DCT matrix D
%   GQ = quant(block, QTAB, D)
    C = D * block * D';     % DCT
    CQ = round(C ./ QTAB);  % Quantization
    GQ = zigzag_mat8(CQ);   % Zig-Zag, this is a row vector
end

