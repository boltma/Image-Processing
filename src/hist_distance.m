function d = hist_distance(h1, h2)
%HIST_DISTANCE �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
    d = 1 - sum(sqrt(h1 .* h2));
end

