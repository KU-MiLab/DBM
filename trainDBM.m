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

function [DBM, errorChange] = trainDBM( DBM, batchTrainData, batchTrainTarget, numHidUnits, numEpochs, bRetraining, varargin )

if length(varargin) == 2
    testBatchData = varargin{1};
    testBatchTarget = varargin{2};
else
    testBatchData = [];
    testBatchTarget = [];
end
fprintf(1,'Pretraining a Deep Boltzmann Machine. \n');

if bRetraining == 1
    visData = batchTrainData;
    for h=1:numel(numHidUnits)
        if h==numel(numHidUnits) && ~isempty(batchTrainTarget)
            % supervised learning
            DBM = pretrainDBM( DBM, h, visData, numHidUnits, numEpochs(h), batchTrainTarget, testBatchData, testBatchTarget );
        else
            % unsupervised learning
            DBM = pretrainDBM( DBM, h, visData, numHidUnits, numEpochs(h), [] );
        end
        
        if h<numel(numHidUnits)
            visData = forwardRBM( DBM, h, visData );
        end
        
        fprintf( 2, 'Done: pre-training %d-th hidden layer...\n', h );
    end
    clear visData;
    
    save( 'pretrainedDBM.mat', 'DBM' );
else
    fprintf( 2, 'Loading the pretrained model...\n' );
    load( 'pretrainedDBM_binary.mat', 'DBM' );
end

errorChange = [];
if numel(DBM)>1 && numEpochs(end) > 0
    [DBM, errorChange] = DBM_ApproxLearning( batchTrainData, batchTrainTarget, DBM, numEpochs(end) );
end



