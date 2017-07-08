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

function DBM = approxMeanField( DBM, data, targets, verbose )

% initialized the hidden units value (similar to pretraining)
DBM = forwardDBM( DBM, data, targets );

numData = size( data, 1 );
for ii=1:30
    for h=1:numel(DBM)
        if h==1
            bottomUp = data*DBM{h}.W;
        else
            bottomUp = DBM{h-1}.posHProbs*DBM{h}.W;
        end
        
        if h < numel(DBM)
            topDown = DBM{h+1}.posHProbs*DBM{h+1}.W';
        else
            if ~isempty(targets)
                topDown = targets*DBM{h}.OW;
            else
                topDown = 0;
            end
        end
        
        DBM{h}.posHProbsOld = DBM{h}.posHProbs;
        DBM{h}.posHProbs = sigmoid( bottomUp + topDown + repmat(DBM{h}.HB, numData, 1) );
    end
    
    
    bBreak = 1;
    if verbose == 1
        fprintf( 1, '\t\t\t\t' );
    end
    for h=1:numel(DBM)
        [numData, numHidUnits] = size(DBM{h}.posHProbs);
        diffSum = sum(sum(abs(DBM{h}.posHProbs-DBM{h}.posHProbsOld), 2))/(numData*numHidUnits);
        
        if verbose == 1
            fprintf( 1,'ii=%d Mean-Field: h%d=%f, ', ii, h, diffSum );
        end
        if diffSum > 0.0000001
            bBreak = 0;
            break;
        end
    end
    if verbose == 1
        fprintf( 1, '\n' );
    end
    
    if bBreak==1
        if verbose == 1
            fprintf( 2, '\t\t\t\t Mean-Field Approximation converged...\n' );
        end
        break;
    end
end

