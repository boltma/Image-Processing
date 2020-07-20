function c = DC_huff(e, DCTAB)
%DC_HUFF Get DC Huffman code
%   c = DC_huff(e, DCTAB)
    assert(abs(e) <= 2047);
    category = max(0, floor(log2(abs(e))) + 1); % Also works when e == 0
    huff = DCTAB(category + 1, :);
    len = huff(1);
    h = huff(2:1+len);
    c = [h num2bin(e)];
end

