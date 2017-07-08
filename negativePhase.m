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

function DBM = negativePhase( DBM, numData, numCD )

for h=1:numel(DBM)
    DBM{h}.negHProbs = DBM{h}.posHProbs;
end

for iter=1:numCD
    for h=1:numel(DBM)
        if h==1
            % Bernoulli sampling
            DBM{h}.negHStates = DBM{h}.negHProbs > rand( size(DBM{h}.negHProbs) );
            
            DBM{h}.negVProbs = sigmoid( DBM{h}.negHStates*DBM{h}.W' + repmat(DBM{h}.VB, numData, 1) );
            
            if DBM{h}.binaryInputs == 1                
                % Bernoulli sampling
                DBM{h}.negVStates = DBM{h}.negVProbs > rand( size(DBM{h}.negVProbs) );
            else
                DBM{h}.negVStates = DBM{h}.negVProbs;
            end
        else
            bottomUp = DBM{h-1}.negHStates*DBM{h}.W;
            
            if h < numel(DBM)
                topDown = DBM{h+1}.negHStates*DBM{h+1}.W';
            else
                if isfield(DBM{h}, 'OW')
                    topDown = DBM{h}.negOStates*DBM{h}.OW;
                else
                    topDown = 0;
                end
            end
            
            DBM{h}.negHProbs = sigmoid( bottomUp + topDown + repmat(DBM{h}.HB, numData, 1) );    % p(h|x)
            
            % Bernoulli sampling
            DBM{h}.negHStates = DBM{h}.negHProbs > rand(size(DBM{h}.negHProbs));
            
            if h==numel(DBM) && isfield(DBM{h}, 'OW')
                if DBM{h}.binaryTarget == 1
                    % logistic regression
                    totin = DBM{h}.negHStates*DBM{h}.OW' + repmat(DBM{h}.OB, numData, 1);
                    DBM{h}.negOProbs = exp(totin);
                    DBM{h}.negOProbs = DBM{h}.negOProbs ./ (sum(DBM{h}.negOProbs, 2)*ones(1, size(DBM{h}.OW, 1)));
                    
                    xx = cumsum( DBM{h}.negOProbs, 2 );
                    xx1 = rand( numData, 1 );
                    DBM{h}.negOStates = DBM{h}.negOStates*0;
                    for jj=1:numData
                        index = min(find(xx1(jj) <= xx(jj,:)));
                        DBM{end}.negOStates(jj, index) = 1;
                    end
                else    %%% should be revised correctly %%%
                    DBM{h}.negOProbs = sigmoid(DBM{h}.negHStates*DBM{h}.OW' + repmat(DBM{h}.OB, numData, 1));
                    DBM{h}.negOStates = DBM{h}.negOProbs;
                end
            end
        end
    end
    
    % assume that the number of layers is always larger than 1
    DBM{1}.negHProbs = sigmoid( DBM{1}.negVStates*DBM{1}.W + DBM{2}.negHStates*DBM{2}.W' + repmat(DBM{1}.HB, numData, 1) );
end

if isfield(DBM{end}, 'OW')
    % assume that the number of layers is always larger than 1
    DBM{end}.negHProbs = sigmoid( DBM{end-1}.negHProbs*DBM{end}.W + DBM{end}.negOProbs*DBM{end}.OW + repmat(DBM{end}.HB, numData, 1) );
else
    % assume that the number of layers is always larger than 1
    DBM{end}.negHProbs = sigmoid( DBM{end-1}.negHProbs*DBM{end}.W + repmat(DBM{end}.HB, numData, 1) );
end

% Summary statistics
for h=1:numel(DBM)
    if h==1
        DBM{h}.negProds = DBM{h}.negVStates' * DBM{h}.negHProbs;
        DBM{h}.negVAct = sum(DBM{h}.negVStates);
    else
        DBM{h}.negProds = DBM{h-1}.negHProbs' * DBM{h}.negHProbs;
    end
    DBM{h}.negHAct = sum(DBM{h}.negHProbs);
end

if isfield(DBM{end}, 'OW')
    DBM{end}.negOAct = sum(DBM{end}.negOStates);
    DBM{end}.negOProds = DBM{end}.negOStates' * DBM{end}.negHProbs;
end



