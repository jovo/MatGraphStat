function P = get_ind_edge_params(adjacency_matrices,class_labels,type)
% this funtion gets the parameters necessary for independent edge algorithms

if ~isstruct(class_labels),
    constants=get_constants(adjacency_matrices,class_labels);
else
    constants=class_labels;
end


if nargin==2, type='L-estimator'; end % default to L-estimator
eta = 1/(10*constants.s);             % to deal with 0's and 1's

sum0 = sum(adjacency_matrices(:,:,constants.y0),3);
sum1 = sum(adjacency_matrices(:,:,constants.y1),3);

if strcmp(type,'L-estimator')
    % estimated class 0 edge probabilities
    P.E0            = sum0/constants.s0;
    P.E0(P.E0==0)   = eta;
    P.E0(P.E0==1)   = 1-eta;
    
    % estimated class 0 edge probabilities
    P.E1            = sum1/constants.s1;
    P.E1(P.E1==0)   = eta;
    P.E1(P.E1==1)   = 1-eta;

elseif strcmp(type,'robust')
    P.E0=(sum0+eta)/(constants.s0+eta);
    P.E1=(sum1+eta)/(constants.s1+eta);
   
elseif strcmp(type,'map')
    % set prior to be flat
    alpha = 1.001;
    beta = 1.001;
        
    P.alpha0    = alpha + sum0;
    P.alpha1    = alpha + sum1;
    
    P.beta0     = beta + constants.s0 - sum0;
    P.beta1     = beta + constants.s1 - sum1;
    
    % mode
    P.E0     = (P.alpha0-1)./(P.alpha0+P.beta0-2);
    P.E1     = (P.alpha1-1)./(P.alpha1+P.beta1-2);
    
elseif strcmp(type,'mle')
    P.E0 = sum0/constants.s0;
    P.E1 = sum1/constants.s1;        
end

% pre-compute constants for bernoulli distribution
P.lnE0  = log(P.E0);
P.ln1E0 = log(1-P.E0);
P.lnE1  = log(P.E1);
P.ln1E1 = log(1-P.E1);

% log-priors (using L-estimator)
pi0=constants.s0/constants.s;
if pi0==0, pi0=eta; elseif pi0==1, pi0=1-eta; end
pi1=constants.s1/constants.s;
if pi1==0, pi1=eta; elseif pi1==1, pi1=1-eta; end

P.lnprior0 = log(pi0);
P.lnprior1 = log(pi1);