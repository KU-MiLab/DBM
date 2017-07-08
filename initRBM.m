function RBM = initRBM( numData, numVisUnits, numHidUnits, numOutUnits )

RBM.binaryInputs = 1;

% parameter initialization
RBM.W           = 0.01*randn( numVisUnits, numHidUnits );
RBM.HB          = zeros( 1, numHidUnits );
RBM.VB          = zeros( 1, numVisUnits );

RBM.dW          = zeros( numVisUnits, numHidUnits );
RBM.dHB         = zeros( 1, numHidUnits );
RBM.dVB         = zeros( 1, numVisUnits );

% probability
RBM.posHProbs   = zeros( numData, numHidUnits );
RBM.negHProbs   = zeros( numData, numHidUnits );
RBM.negVProbs   = zeros( numData, numVisUnits );

% binary state
RBM.posHStates  = zeros( numData, numHidUnits );
RBM.negHStates  = zeros( numData, numHidUnits );
RBM.negVStates  = zeros( numData, numVisUnits );

RBM.bClamped    = 0;    % indicator whether to fix the states

RBM.posProds    = zeros( numData, numHidUnits );
RBM.negProds    = zeros( numData, numHidUnits );

RBM.posHAct     = zeros( 1, numHidUnits );
RBM.negHAct     = zeros( 1, numHidUnits );
RBM.HAvg        = ones( 1, numHidUnits );

RBM.posVAct     = zeros( 1, numVisUnits );
RBM.negVAct     = zeros( 1, numVisUnits );

if ~isempty(numOutUnits)
    RBM.binaryTarget    = 1;
    RBM.OW              = 0.01*randn( numOutUnits, numHidUnits );
    RBM.OB              = zeros( 1, numOutUnits );
    
    RBM.posOProds       = zeros( numData, numOutUnits );
    RBM.negOProds       = zeros( numData, numOutUnits );
    
    RBM.posOAct         = zeros( 1, numOutUnits );
    RBM.negOAct         = zeros( 1, numOutUnits );
    
    RBM.posOStates      = zeros( numData, numOutUnits );
    RBM.negOStates      = ones( numData, numOutUnits )/numOutUnits;
    
    RBM.dOW             = zeros( numOutUnits, numHidUnits );
    RBM.dOB             = zeros( 1, numOutUnits );
    
    RBM.TOut            = zeros( numData, numOutUnits );
end
