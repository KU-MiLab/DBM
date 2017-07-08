%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%                     Written by H.-I. Suk                    %%%%%%%
%%%%%%%`             based on Ruslan Salakhutdinov's codes          %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied. As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.

function DBM = pretrainDBM( DBM, h, visData, numHidUnits, numEpochs, batchTarget, varargin )
% DBM: structured format by initRBM.m
% visData, batchTarget: cell{numBatches}[numDataPerBatch, numVisUnits]
% numHidUnits: vector[bottom_hidden_layer, ..., top_hidden_layer]
% numEpochs: scalar
% batchTarget: binary class labels

if length(varargin) == 2
    testBatchData = varargin{1};
    testBatchTarget = varargin{2};
else
    testBatchData = [];
    testBatchTarget = [];
end


if ~isempty(batchTarget)
    numTargetUnits = size( batchTarget{1}, 2 );
end

numBatches = numel( visData );
numData = size( visData{1}, 1 );

isTargetAvailable = (h==numel(numHidUnits) && ~isempty(batchTarget) );

% Hyper-parametersfor learning
epsilonW0    = 0.01;     % Learning rate for weights
epsilonVB0   = 0.01;     % Learning rate for biases of visible units
epsilonHB0   = 0.01;     % Learning rate for biases of hidden units
epsilonOB0   = 0.01;     % Learning rate for biases of label units

weightCost      = 0.001;
initialMomentum = 0.5;
finalMomentum   = 0.9;

minError = Inf;
minErrorIdx = 1;
tolerate_cnt = 10;

for epoch=1:numEpochs
    if isTargetAvailable==1
        CD = ceil(epoch/20);
        epsilonW = epsilonW0/(1*CD);
        epsilonVB = epsilonVB0/(1*CD);
        epsilonHB = epsilonHB0/(1*CD);
        epsilonOB = epsilonOB0/(1*CD);
    else
        CD = 1;
        epsilonW = epsilonW0;
        epsilonVB = epsilonVB0;
        epsilonHB = epsilonHB0;
    end
    
    fprintf(1, 'epoch %d \t eps %f\r', epoch, epsilonW);
    errorSum = 0;
    for batch=1:numBatches    
        if rem(batch, 10) == 0
            fprintf( 1, '.' );
        end
        
        % If the value in data is continuous, scaled between 0 and 1, it is considered as a probability. 
        data = visData{batch};
        if DBM{h}.binaryInputs == 1 % binary visible units
            data = data > rand( size(data) );   % convert to binary
        end
        
        %%%%%%% POSITIVE PHASE %%%%%%%
        if isTargetAvailable==1
            targets = batchTarget{batch};
            DBM{h}.posHProbs = sigmoid( data*DBM{h}.W + targets*DBM{h}.OW + repmat(DBM{h}.HB, numData, 1) );    % p(h|x)*p(h|h^{u})
            DBM{h}.posOProds = targets' * DBM{h}.posHProbs;
            DBM{h}.posOAct = sum(targets);
        else
            if h<length(DBM)
                DBM{h}.posHProbs = sigmoid( 2*(data*DBM{h}.W + repmat(DBM{h}.HB, numData, 1)) );    % p(h|x)
            else
                DBM{h}.posHProbs = sigmoid( data*DBM{h}.W + repmat(DBM{h}.HB, numData, 1) );        % p(h|x)
            end
        end
        
        DBM{h}.posProds = data' * DBM{h}.posHProbs;           % p(x,h)=p(x)*p(h|x)
        DBM{h}.posHAct = sum( DBM{h}.posHProbs );
        DBM{h}.posVAct = sum( data );
        
        temp = DBM{h}.posHProbs;
        
        %%%%%%% NEGATIVE PHASE %%%%%%%
        for cd=1:CD
            DBM{h}.posHStates = temp > rand( size(temp) );    % Bernoulli sampling
            
            if isTargetAvailable==1
                totIn = DBM{h}.posHStates*DBM{h}.OW' + repmat(DBM{h}.OB, numData, 1);
                
                % logistic function
                DBM{h}.negOProbs = exp( totIn );
                DBM{h}.negOProbs = DBM{h}.negOProbs ./ (sum(DBM{h}.negOProbs, 2)*ones(1, numTargetUnits));
                
                xx = cumsum( DBM{h}.negOProbs, 2 );
                xx1 = rand( numData, 1 );
                DBM{h}.negOStates = zeros( size(DBM{h}.negOProbs) );
                for jj=1:numData
                    index = min( find(xx1(jj) <= xx(jj, :)) );
                    DBM{h}.negOStates(jj, index) = 1;
                end
                
                DBM{h}.negVProbs = sigmoid( 2*(DBM{h}.posHStates*DBM{h}.W' + repmat( DBM{h}.VB, numData, 1)) );  % p(x|h)
                
                if DBM{h}.binaryInputs == 1 % binary units
                    DBM{h}.negVStates = DBM{h}.negVProbs > rand( size(DBM{h}.negVProbs) );     % Bernoulli sampling
                else % real-valued inputs
                    DBM{h}.negVStates = DBM{h}.negVProbs;
                end
                
                temp = sigmoid( DBM{h}.negVStates*DBM{h}.W + DBM{h}.negOStates*DBM{h}.OW + repmat(DBM{h}.HB, numData, 1) ); % p(x,h)=p(x)*p(h|x)
            else
                if h>1
                    DBM{h}.negVProbs = sigmoid( 2*(DBM{h}.posHStates*DBM{h}.W' + repmat( DBM{h}.VB, numData, 1)) );  % p(x|h)
                else
                    DBM{h}.negVProbs = sigmoid( DBM{h}.posHStates*DBM{h}.W' + repmat( DBM{h}.VB, numData, 1) );  % p(x|h)
                end
                
                if DBM{h}.binaryInputs == 1 % binary units
                    DBM{h}.negVStates = DBM{h}.negVProbs > rand( size(DBM{h}.negVProbs) );     % Bernoulli sampling
                else  % continuous (probability) units
                    DBM{h}.negVStates = DBM{h}.negVProbs;
                end
                
                if h<length(DBM)
                    temp = sigmoid( 2*(DBM{h}.negVStates*DBM{h}.W + repmat(DBM{h}.HB, numData, 1)) ); % p(x,h)=p(x)*p(h|x)
                else
                    temp = sigmoid( DBM{h}.negVStates*DBM{h}.W + repmat(DBM{h}.HB, numData, 1) ); % p(x,h)=p(x)*p(h|x)
                end
            end
        end
        DBM{h}.negHProbs = temp;
        
        DBM{h}.negProds = DBM{h}.negVStates' * DBM{h}.negHProbs;
        DBM{h}.negHAct = sum( DBM{h}.negHProbs );
        DBM{h}.negVAct = sum( DBM{h}.negVStates );
        
        if isTargetAvailable==1
            DBM{h}.negOAct = sum(DBM{h}.negOStates);
            DBM{h}.negOProds = DBM{h}.negOStates' * DBM{h}.negHProbs;
        end
        
        err = sum( (data(:)-DBM{h}.negVStates(:)).^2);
        errorSum = err + errorSum;
        
        if epoch > 5
            momentum = finalMomentum;
        else
            momentum = initialMomentum;
        end        
        
        %%%%%%% PARAMETER UPDATE %%%%%%%
        DBM{h}.dW = momentum*DBM{h}.dW + ...
            epsilonW*( (DBM{h}.posProds-DBM{h}.negProds)/numData - weightCost*DBM{h}.W);      
        DBM{h}.dVB = momentum*DBM{h}.dVB + (epsilonVB/numData)*(DBM{h}.posVAct-DBM{h}.negVAct);
        DBM{h}.dHB = momentum*DBM{h}.dHB + (epsilonHB/numData)*(DBM{h}.posHAct-DBM{h}.negHAct);
        
        DBM{h}.W  = DBM{h}.W + DBM{h}.dW;
        DBM{h}.VB = DBM{h}.VB + DBM{h}.dVB;
        DBM{h}.HB = DBM{h}.HB + DBM{h}.dHB;
        
        if isTargetAvailable==1
            DBM{h}.dOW = momentum*DBM{h}.dOW + ...
                epsilonW*( (DBM{h}.posOProds-DBM{h}.negOProds)/numData - weightCost*DBM{h}.OW);
            DBM{h}.dOB = momentum*DBM{h}.dOB + (epsilonOB/numData)*(DBM{h}.posOAct-DBM{h}.negOAct);
            
            DBM{h}.OW = DBM{h}.OW + DBM{h}.dOW;
            DBM{h}.OB = DBM{h}.OB + DBM{h}.dOB;
        end
        
        if rem(batch, 100)==0 && h==1 
            figure(1);
            subplot(1, 2, 1);
            dispims(data', 28, 28);
            subplot(1, 2, 2);
            dispims(DBM{h}.negVStates', 28, 28);
            drawnow;
        end
    end
    
    fprintf( 1, '\n\tHidden Layer: %d/%d, Epoch: %4i, Error: %6.1f...\n', h, numel(numHidUnits), epoch, errorSum );
    
    if rem(epoch, 1)==0 && h==numel(DBM) && ~isempty(testBatchData) && ~isempty(testBatchTarget)
        [totalErr, totalTests] = testDBMErr(testBatchData, testBatchTarget, DBM);
        fprintf(1, '\t\tNumber of misclassified test examples: %d out of %d \n', totalErr, totalTests);
    end
end



