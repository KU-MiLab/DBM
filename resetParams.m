function DBM = resetParams( DBM, batchData, sparsetarget )
for h=1:numel(DBM)
    DBM{h}.HAvg = ones(size(DBM{h}.HAvg));
    DBM{h}.dHB = DBM{h}.dHB*0;
    DBM{h}.dVB = DBM{h}.dVB*0;
    DBM{h}.dW = DBM{h}.dW*0;
    
    if isfield(DBM{h}, 'OW')
        DBM{h}.dOB = DBM{h}.dOB*0;
        DBM{h}.dOW = DBM{h}.dOW*0;
    end
end

% Setup for negativePhase (taken from Salakhutdinov's code) - Really need this step?
numData = size(batchData{1}, 1);
for h=1:numel(DBM)
    if h<numel(DBM)
        DBM{h}.HB = (DBM{h}.HB + DBM{h+1}.VB);
    end
    DBM{h}.HAvg = sparsetarget*DBM{h}.HAvg;
end
rndData = rand( size(batchData{1}) );
if DBM{1}.binaryInputs == 1
    rndData = round( rndData );
end
DBM{1}.negHProbs = sigmoid( 2*rndData*DBM{1}.W + repmat(DBM{1}.HB, numData, 1) );
DBM{1}.negHStates = DBM{1}.negHProbs > rand(size(DBM{1}.negHProbs));

if isfield(DBM{end}, 'OW')
    DBM{end}.negOStates = ones( numData, size(DBM{end}.OW, 1) )/size(DBM{end}.OW, 1);
end

for h=2:numel(DBM)
    if h==numel(DBM)
        factor = 1;
    else
        factor = 2;
    end
    
    if h==numel(DBM) && isfield(DBM{h}, 'OW')
        DBM{h}.negHProbs = sigmoid( DBM{h-1}.negHStates*DBM{h}.W + DBM{h}.negOStates*DBM{h}.OW + repmat(DBM{h}.HB, numData, 1) );    % p(h|x)
    else
        DBM{h}.negHProbs = sigmoid( factor*DBM{h-1}.negHStates*DBM{h}.W + repmat(DBM{h}.HB, numData, 1) );    % p(h|x)
    end
    
%     if h==numel(DBM) && isfield(DBM{h}, 'OW')
%         DBM{h}.negHProbs = sigmoid( DBM{h-1}.negHProbs*DBM{h}.W + DBM{h}.negOProbs*DBM{h}.OW + repmat(DBM{h}.HB, numData, 1) );    % p(h|x)
%     else
%         DBM{h}.negHProbs = sigmoid( factor*DBM{h-1}.negHProbs*DBM{h}.W + repmat(DBM{h}.HB, numData, 1) );    % p(h|x)
%     end
    DBM{h}.negHStates = DBM{h}.negHProbs > rand( size(DBM{h}.negHProbs) );
end

