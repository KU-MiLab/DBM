function DBM = updateDBM( DBM, data, epoch, epsilonW, epsilonB, sparsetarget )

numData = size(data, 1);

weightcost      = 0.0002;
initialmomentum = 0.5;
finalmomentum   = 0.9;

sparsecost      = .001;
sparsedamping   = .9;

if epoch > 5
    momentum = finalmomentum;
else
    momentum = initialmomentum;
end

for h=1:numel(DBM)
    DBM{h}.HAvg = sparsedamping*DBM{h}.HAvg + (1-sparsedamping)*DBM{h}.posHAct/numData;
    DBM{h}.sparsegrads = sparsecost*(repmat(DBM{h}.HAvg,numData,1)-sparsetarget);
    
    DBM{h}.dHB = momentum*DBM{h}.dHB + ...
        epsilonB/numData*(DBM{h}.posHAct-DBM{h}.negHAct) - ...
        epsilonB/numData*sum(DBM{h}.sparsegrads);
    DBM{h}.HB = DBM{h}.HB + DBM{h}.dHB;
    
    if h==1
        DBM{h}.dVB = momentum*DBM{h}.dVB + (epsilonB/numData)*(DBM{h}.posVAct-DBM{h}.negVAct);
        DBM{h}.VB = DBM{h}.VB + DBM{h}.dVB;
        
        DBM{h}.dW = momentum*DBM{h}.dW + ...
            epsilonW*((DBM{h}.posProds-DBM{h}.negProds)/numData - ...
            weightcost*DBM{h}.W - data'*DBM{h}.sparsegrads/numData );
    else
        DBM{h}.dW = momentum*DBM{h}.dW + ...
            epsilonW*((DBM{h}.posProds-DBM{h}.negProds)/numData - ...
            weightcost*DBM{h}.W - DBM{h-1}.posHProbs'*DBM{h}.sparsegrads/numData - ...
            (DBM{h}.posHProbs'*DBM{h-1}.sparsegrads)'/numData );
    end
    DBM{h}.W = DBM{h}.W + DBM{h}.dW;
    
    if isfield(DBM{h}, 'OW')
        DBM{h}.dOB = momentum*DBM{h}.dOB + (epsilonB/numData)*(DBM{h}.posOAct-DBM{h}.negOAct);
        DBM{h}.OB = DBM{h}.OB + DBM{h}.dOB;
        
        DBM{h}.dOW = momentum*DBM{h}.dOW + ...
            epsilonW*((DBM{h}.posOProds-DBM{h}.negOProds)/numData - weightcost*DBM{h}.OW);
        DBM{h}.OW = DBM{h}.OW + DBM{h}.dOW;
    end
end