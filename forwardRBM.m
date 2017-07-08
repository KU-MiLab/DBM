function output = forwardRBM( RBM, h, batchData )

if numel(RBM) > 1 && h<numel(RBM)
    factor = 2;
else
    factor = 1;
end

output = cell( 1, numel(batchData) );
for batch=1:numel(batchData)
    visData = batchData{batch};
    numData = size(visData, 1);
    
    output{batch} = sigmoid( factor*(visData*RBM{h}.W + repmat(RBM{h}.HB, numData, 1)) );    % p(h|x)
end