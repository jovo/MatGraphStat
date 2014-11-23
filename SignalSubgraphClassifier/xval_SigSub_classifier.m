function [Lhat incorrects subspace tElapsed] = xval_SigSub_classifier(As,ys,constraints,cv,Atst,ytst)
% loops over constraints
% estimating the signal subgraph and plug-in classifier for each iteration.
%
% INPUTS:
%   As:         adjacency matrices
%   ys:         class labels
%   cv:         cross-validation type: 'loov','InSample','HoldOut'
%   constraints:structure containing algorithm parameters
%   Atst:       (only when cv='HoldOut'): test graphs
%   ytst:       (only when cv='HoldOut'): test class
%
% OUTPUTS:
%   Lhat:       average misclassification rates
%   incorrects: whether the classifier got the answer correct
%   subspace:   the set of features used to classify

constants = get_constants(As,ys);
if nargin==2 || ~exist('constraints','var'), constraints{1}=NaN; end
if nargin<=3, cv='loo'; end
len_constraints=length(constraints);

% determine whether to estimate the significance matrix
get_SigMat=0;               % default to not estimating SigMat
FullMat=1:constants.n^2;    % default to taking whole matrix
SigMat=0*FullMat;           % initialize SigMat
if len_constraints>1,
    get_SigMat=1;
elseif len_constraints==1,
    % constraints must be cell arrays
    if ~iscell(constraints), c{1}=constraints; clear constraints, constraints=c; clear c; end
    if ~isnan(constraints{1}), get_SigMat=1; end
end
try matlabpool, catch ME, display(' '); end
tElapsed=[]; % currently only used in 'HoldOut'

%% cross-validate over different algorithms
if strcmp(cv,'loo')    % get signal subgraph using training data, then classify test data
    
    incorrects=nan(constants.s,len_constraints);
    for i=1:constants.s, 
        
        if mod(i,10)==0, disp(['loocv iter: ' num2str(i)]), end
        
        [Atrn Gtrn Atst] = xval_prep(As,constants,[],i);    % seperate data into training and testing sets
        phat    = get_ind_edge_params(Atrn,Gtrn);           % get parameters
        
        if get_SigMat, SigMat = get_fisher(Atrn,Gtrn); end
        yi=ys(i);
        parfor j=1:len_constraints
            
            if mod(j,100)==0, disp(['current constraint: ' num2str(constraints{j})]), end
           
            subspace{i,j} = signal_subgraph_estimator(SigMat,constraints{j});
            incorrects(i,j) = plugin_bern_classify(Atst,phat,subspace{i,j},yi);
        end
    end
    
    
elseif strcmp(cv,'InSample') % learn signal subgraph from full data
    
    incorrects=nan(constants.s,len_constraints);

    if get_SigMat, SigMat = get_fisher(As,constants); end
    for j=1:len_constraints
        subspace{j} = signal_subgraph_estimator(SigMat,constraints{j});
        
        parfor i=1:constants.s
            [Atrn Gtrn Atst] = xval_prep(As,constants,[],i); % seperate data into training and testing sets
            phat    = get_ind_edge_params(Atrn,Gtrn);
            incorrects(i,j) = plugin_bern_classify(Atst,phat,subspace{j},ys(i));
        end
    end
    
elseif strcmp(cv,'HoldOut'), % get signal subgraph from training data, test on held-out data
    
    incorrects=nan(length(ytst),len_constraints);

    tic
    phat    = get_ind_edge_params(As,constants);
    if get_SigMat, SigMat = get_fisher(As,constants); end
    minTime=toc;
    
    for j=1:len_constraints
        tic
        subspace{j} = signal_subgraph_estimator(SigMat,constraints{j});
        parfor i=1:length(ytst)
            incorrects(i,j) = plugin_bern_classify(Atst(:,:,i),phat,subspace{j},ytst(i));
        end
        tElapsed(j)=toc+minTime;
    end
    
end % cv

Lhat=mean(incorrects,1);