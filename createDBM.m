function DBM = createDBM( batchTrainData, batchTrainTarget, bBinaryInputs, numHidUnits )

fprintf(1,'Creating a Deep Boltzmann Machine. \n');

% numData: # of patches
% numDim: size of a patch
[numData, numDim] = size( batchTrainData{1} );

if numel(numHidUnits) == 1
    if ~isempty(batchTrainTarget)
        numClasses = size(batchTrainTarget{1}, 2);
        DBM{1} = initRBM( numData, numDim, numHidUnits(1), numClasses );
    else
        DBM{1} = initRBM( numData, numDim, numHidUnits(1), [] );
    end
else
    for h=1:numel(numHidUnits)
        if h==numel(numHidUnits) && ~isempty(batchTrainTarget)
            numClasses = size(batchTrainTarget{1}, 2);
            DBM{h} = initRBM( numData, numHidUnits(h-1), numHidUnits(h), numClasses );
        elseif h==1
            DBM{h} = initRBM( numData, numDim, numHidUnits(h), [] );
            DBM{h}.binaryInputs = bBinaryInputs;
        else
            DBM{h} = initRBM( numData, numHidUnits(h-1), numHidUnits(h), [] );
        end
    end
end