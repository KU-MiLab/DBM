function DBM = loadTrainedDBM( bUseTarget, bUseGRBM, bUnimodal, trainedModelPath, type, tissue, cv, folder, bBinaryInputs, strHiddenUnits, dbmEpoch, bWhitening )

if bUseTarget == 1
    if bUseGRBM == 0
        if bUnimodal == 1
            eval( ['load ', trainedModelPath, type, '_Unimodal_', tissue, '_CV', num2str(cv), '_', folder, '_bBinary', num2str(bBinaryInputs), strHiddenUnits, '_numEpoch_', num2str(dbmEpoch), '.mat DBM;'] );
        else
            eval( ['load ', trainedModelPath, type, '_Multimodal', tissue, '_CV', num2str(cv), '_', folder, '_bBinary', num2str(bBinaryInputs), strHiddenUnits, '_numEpoch_', num2str(dbmEpoch), '.mat DBM;'] );
        end
    else
        if bUnimodal == 1
            eval( ['load ', trainedModelPath, type, '_Unimodal_', tissue, '_CV_GRBM', num2str(cv), '_', folder, '_bBinary', num2str(bBinaryInputs), '_Whitening', num2str(bWhitening), strHiddenUnits, '_numEpoch_', num2str(dbmEpoch), '.mat DBM;'] );
        else
            eval( ['load ', trainedModelPath, type, '_Multimodal', tissue, '_CV_GRBM', num2str(cv), '_', folder, '_bBinary', num2str(bBinaryInputs), '_Whitening', num2str(bWhitening), strHiddenUnits, '_numEpoch_', num2str(dbmEpoch), '.mat DBM;'] );
        end
    end
else
    if bUseGRBM == 0
        if bUnimodal == 1
            eval( ['load ', trainedModelPath, type, '_Unimodal_NoTarget_', tissue, '_CV', num2str(cv), '_', folder, '_bBinary', num2str(bBinaryInputs), strHiddenUnits, '_numEpoch_', num2str(dbmEpoch), '.mat DBM;'] );
        else
            eval( ['load ', trainedModelPath, type, '_Multimodal_NoTarget_', tissue, '_CV', num2str(cv), '_', folder, '_bBinary', num2str(bBinaryInputs), strHiddenUnits, '_numEpoch_', num2str(dbmEpoch), '.mat DBM;'] );
        end
    else
        if bUnimodal == 1
            eval( ['load ', trainedModelPath, type, '_Unimodal_NoTarget_', tissue, '_CV_GRBM', num2str(cv), '_', folder, '_bBinary', num2str(bBinaryInputs), '_Whitening', num2str(bWhitening), strHiddenUnits, '_numEpoch_', num2str(dbmEpoch), '.mat DBM;'] );
        else
            eval( ['load ', trainedModelPath, type, '_Multimodal_NoTarget_', tissue, '_CV_GRBM', num2str(cv), '_', folder, '_bBinary', num2str(bBinaryInputs), '_Whitening', num2str(bWhitening), strHiddenUnits, '_numEpoch_', num2str(dbmEpoch), '.mat DBM;'] );
        end
    end
end