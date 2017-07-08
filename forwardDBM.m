function DBM = forwardDBM( DBM, data, targets, varargin )

if isempty(varargin)
    bTest = 0;
else
    bTest = varargin{1};
end

numData = size( data, 1 );

if numel(DBM)==1
    if ~isempty(targets)
        DBM{1}.posHProbs = sigmoid( data*DBM{1}.W + targets*DBM{1}.OW + repmat(DBM{1}.HB, numData, 1) );    % p(h|x)
    else
        DBM{1}.posHProbs = sigmoid( data*DBM{1}.W + repmat(DBM{1}.HB, numData, 1) );    % p(h|x)
    end
else
    for h=1:numel(DBM)
        if h==numel(DBM)
            factor = 1;
        else
            factor = 2;
        end
        
        if h==1
            DBM{h}.posHProbs = sigmoid( data*factor*DBM{h}.W + repmat(DBM{h}.HB, numData, 1) );    % p(h|x)
        elseif h==numel(DBM) && ~isempty(targets)
            if bTest==1
                DBM{h}.posHProbs = DBM{h-1}.posHProbs*DBM{h}.W + targets*DBM{h}.OW + repmat(DBM{h}.HB, numData, 1);    % p(h|x)
            else
                DBM{h}.posHProbs = sigmoid( DBM{h-1}.posHProbs*factor*DBM{h}.W + targets*DBM{h}.OW + repmat(DBM{h}.HB, numData, 1) );    % p(h|x)
            end
        else
            DBM{h}.posHProbs = sigmoid( DBM{h-1}.posHProbs*factor*DBM{h}.W + repmat(DBM{h}.HB, numData, 1) );    % p(h|x)
        end
    end
end