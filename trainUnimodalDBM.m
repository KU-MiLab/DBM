function trainUnimodalDBM( type, tissue, wsize, bBinaryInputs, bUseGRBM, numMaxEpoch, bUnimodal, folder, numHidUnits, bUseTarget, bWhitening )

rand( 'state', sum(100*clock) );
randn( 'state', sum(100*clock) );

batchPath = fullfile( '.', 'batchData', ['winSize_', num2str(wsize)], 'batchData_' );
trainedModelPath = fullfile( '.', 'trainedModels', ['winSize_', num2str(wsize)], 'trainedDBM_' );

strHiddenUnits = '_numHiddens';
for h=1:numel(numHidUnits)
    strHiddenUnits = [strHiddenUnits, '-', num2str(numHidUnits(h))];
end

% 10-fold cross-validation
for cv=1%:10
    disp( ['******* CV ', num2str(cv), ' *******'] );
    disp( 'Loading batchData sets...' );
    if bUseGRBM == 0
        if bUnimodal == 1
            eval( ['load ', batchPath, type, '_Unimodal_', tissue, '_CV', num2str(cv), '_', folder, '.mat'] );
        else
            eval( ['load ', batchPath, type, '_Multimodal_', tissue, '_CV', num2str(cv), '_', folder, '.mat'] );
        end
    else
        if bUnimodal == 1
            eval( ['load ', batchPath, type, '_Unimodal_', tissue, '_CV_GRBM', num2str(cv), '_', folder, '_NumUnits', num2str(numHidUnits(1)), '_Whitening', num2str(bWhitening), '.mat'] );
        else
            eval( ['load ', batchPath, type, '_Multimodal_', tissue, '_CV_GRBM', num2str(cv), '_', folder, '_NumUnits', num2str(numHidUnits(1)), '_Whitening', num2str(bWhitening), '.mat'] );
        end
    end
    
    
    bRetraining = 1;
    dbmEpoch = 0;
    epochInc = 10;
    for e=0:50 %[0 10 20 30]
        numEpochs = [ones(1, numel(numHidUnits))*numMaxEpoch epochInc];
        
        if bUseTarget == 1
            if bRetraining == 1
                DBM = createDBM( batchTrainData, batchTrainLabel, bBinaryInputs, numHidUnits );
            else
                DBM = loadTrainedDBM( bUseTarget, bUseGRBM, bUnimodal, trainedModelPath, type, tissue, cv, folder, bBinaryInputs, strHiddenUnits, dbmEpoch-epochInc, bWhitening );
            end
            [DBM, errorChange] = trainDBM( DBM, batchTrainData, batchTrainLabel, numHidUnits, numEpochs, bRetraining, batchTrainData, batchTrainLabel );
            
        else
            if bRetraining == 1
                DBM = createDBM( batchTrainData, [], bBinaryInputs, numHidUnits );
            else
                DBM = loadTrainedDBM( bUseTarget, bUseGRBM, bUnimodal, trainedModelPath, type, tissue, cv, folder, bBinaryInputs, strHiddenUnits, dbmEpoch-epochInc, bWhitening );
            end
            [DBM, errorChange] = trainDBM( DBM, batchTrainData, [], numHidUnits, numEpochs, bRetraining );
        end
        
        if bUseTarget == 1
            if bUseGRBM == 0
                if bUnimodal == 1
                    eval( ['save ', trainedModelPath, type, '_Unimodal_', tissue, '_CV', num2str(cv), '_', folder, '_bBinary', num2str(bBinaryInputs), strHiddenUnits, '_numEpoch_', num2str(dbmEpoch), '.mat DBM errorChange;'] );
                else
                    eval( ['save ', trainedModelPath, type, '_Multimodal', tissue, '_CV', num2str(cv), '_', folder, '_bBinary', num2str(bBinaryInputs), strHiddenUnits, '_numEpoch_', num2str(dbmEpoch), '.mat DBM errorChange;'] );
                end
            else
                if bUnimodal == 1
                    eval( ['save ', trainedModelPath, type, '_Unimodal_', tissue, '_CV_GRBM', num2str(cv), '_', folder, '_bBinary', num2str(bBinaryInputs), '_Whitening', num2str(bWhitening), strHiddenUnits, '_numEpoch_', num2str(dbmEpoch), '.mat DBM errorChange;'] );
                else
                    eval( ['save ', trainedModelPath, type, '_Multimodal', tissue, '_CV_GRBM', num2str(cv), '_', folder, '_bBinary', num2str(bBinaryInputs), '_Whitening', num2str(bWhitening), strHiddenUnits, '_numEpoch_', num2str(dbmEpoch), '.mat DBM errorChange;'] );
                end
            end
        else
            if bUseGRBM == 0
                if bUnimodal == 1
                    eval( ['save ', trainedModelPath, type, '_Unimodal_NoTarget_', tissue, '_CV', num2str(cv), '_', folder, '_bBinary', num2str(bBinaryInputs), strHiddenUnits, '_numEpoch_', num2str(dbmEpoch), '.mat DBM errorChange;'] );
                else
                    eval( ['save ', trainedModelPath, type, '_Multimodal_NoTarget_', tissue, '_CV', num2str(cv), '_', folder, '_bBinary', num2str(bBinaryInputs), strHiddenUnits, '_numEpoch_', num2str(dbmEpoch), '.mat DBM errorChange;'] );
                end
            else
                if bUnimodal == 1
                    eval( ['save ', trainedModelPath, type, '_Unimodal_NoTarget_', tissue, '_CV_GRBM', num2str(cv), '_', folder, '_bBinary', num2str(bBinaryInputs), '_Whitening', num2str(bWhitening), strHiddenUnits, '_numEpoch_', num2str(dbmEpoch), '.mat DBM errorChange;'] );
                else
                    eval( ['save ', trainedModelPath, type, '_Multimodal_NoTarget_', tissue, '_CV_GRBM', num2str(cv), '_', folder, '_bBinary', num2str(bBinaryInputs), '_Whitening', num2str(bWhitening), strHiddenUnits, '_numEpoch_', num2str(dbmEpoch), '.mat DBM errorChange;'] );
                end
            end
        end
        
        bRetraining = 0;
        dbmEpoch = dbmEpoch + epochInc;
    end
end



