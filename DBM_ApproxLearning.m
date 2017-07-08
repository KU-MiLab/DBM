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

function [DBM, errorChange] = DBM_ApproxLearning( batchData, batchTargets, DBM, maxepoch )

epsilonW        = 0.001;   % Learning rate for weights
epsilonB        = 0.001;   % Learning rate for biases

sparsetarget    = 0.05;     % sparcity penalty

errorChange = [];
minError = Inf;
minErrorIdx = 1;
tolerate_cnt = 10;

% numData = size(batchData{1}, 1);

DBM = resetParams( DBM, batchData, sparsetarget );

for epoch=1:maxepoch
    numbatches = length(batchData);
    fprintf(1, 'epoch %d \t eps %f\r', epoch, epsilonW);
    
    errorSum = 0;
    
    rr = randperm(numbatches);
    batch = 0;
    counter = 0;
    totalCounter = 0;
    for batch_rr = rr
        batch = batch + 1;
        numData = size(batchData{batch_rr}, 1);
        
        if rem(batch, 10) == 0
            fprintf( 1, '.' );
        end
        
        epsilonW = max(epsilonW/1.000015, 0.00010);
        epsilonB = max(epsilonB/1.000015, 0.00010);
        
        data = batchData{batch_rr};
        if DBM{1}.binaryInputs == 1
            data = double(data > rand(size(data)));
        end
        
        if ~isempty( batchTargets )
            targets = batchTargets{batch_rr};
            if DBM{end}.binaryTarget ==1
                targets = double(targets >= rand(size(targets)));
            end
        else
            targets = [];
        end
        
        DBM = positivePhase( DBM, data, targets );
        
        negDataCD1 = sigmoid( DBM{1}.posHProbs*DBM{1}.W' + repmat(DBM{1}.VB, numData, 1) );
        err= sum( sum( (data-negDataCD1).^2 ) );
        errorSum = err + errorSum;
        
        if ~isempty(targets)
            totIn = DBM{end}.posHProbs*DBM{end}.OW' + repmat(DBM{end}.OB, numData, 1);
            DBM{end}.posOProbs = exp(totIn);
            DBM{end}.posOProbs = DBM{end}.posOProbs ./ (sum(DBM{end}.posOProbs, 2)*ones(1, size(DBM{end}.OW, 1)));
            [I, J] = max(DBM{end}.posOProbs, [], 2);
            [I1, J1] = max(targets, [], 2);
            counter = counter + length(find(J~=J1));
            totalCounter = totalCounter + size(targets, 1);
        end
        
        numCD = 30;
        DBM = negativePhase( DBM, numData, numCD );        
        
        DBM = updateDBM( DBM, data, epoch, epsilonW, epsilonB, sparsetarget );
        
        % In order to check the validity of the codes with MNIST
        if rem(batch,100)==0
            figure(1);
            subplot(1, 2, 1);   dispims(data',28,28);
            subplot(1, 2, 2);   dispims(DBM{1}.negVStates',28,28);
        end
    end
    
    fprintf( 1, '\n\t epoch %4i reconstruction error %6.1f \n', epoch, errorSum );
    if ~isempty(targets)
        fprintf( 1, '\t epoch %4i: Number of misclassified training samples %d (out of %d)\n', epoch, counter, totalCounter );
    end
    
    errorChange = [errorChange errorSum];
    
    %     if errorSum < minError
    %         minError = errorSum;
    %         minErrorIdx = epoch;
    %     else
    %         if epoch > minErrorIdx+tolerate_cnt
    %             fprintf( 2, 'Stopping criterion reached (recon error) %.4f > %.4f...\n', errorSum, minError );
    %             return;
    %         end
    %     end
end


