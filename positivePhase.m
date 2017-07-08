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

function DBM = positivePhase( DBM, data, targets, varargin )

if isempty(varargin)
    verbose = 0;
else
    verbose = varargin{1};
end

DBM = approxMeanField( DBM, data, targets, verbose );

% Summary statistics
for h=1:numel(DBM)
    if h==1
        DBM{h}.posProds = data' * DBM{h}.posHProbs;
        DBM{h}.posVAct = sum(data);
    else
        DBM{h}.posProds = DBM{h-1}.posHProbs' * DBM{h}.posHProbs;
    end
    
    DBM{h}.posHAct = sum(DBM{h}.posHProbs);
    DBM{h}.posProds = DBM{h}.posProds;
end

if ~isempty(targets)
    DBM{end}.posOProds = targets' * DBM{end}.posHProbs;
    DBM{end}.posOAct = sum(targets);
end
