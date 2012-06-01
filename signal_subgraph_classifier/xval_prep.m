function [Atrn Gtrn Atst Gtst inds] = xval_prep(As,constants,xval,i)
% this function prepares data for cross-validation using either:
% (1) leave-one-out
% (2) leave-k-out
% 
% when doing leave-one-out, this code just selects 1 to leave out, 
% in particular, the i-th one
%
% INPUT:
%   As:         adjacency matrices
%   constants:  number of samples, etc.
%   xval:       structure containing xval parameters including
%       s0_trn: # of training samples for class 0
%       s1_trn: # of training samples for class 1
%       num_iters: (optional) max # of iters to perform
%       s0_tst: (optional) # of testing samples for class 0
%       s1_tst: (optional) # of testing samples for class 1
%   i:          when doing loo, which one to leave out
%
% OUTPUT:
%   Atrn:       adjacency matrices for training
%   Gtrn:       constatnts for training data
%   Atst:       adj. mat.'s for testing
%   Gtst:       constants for testing data
%   inds:       collection of indices (for graph invariant approach)


if ~isfield(xval,'loo') % default to loo
    xval.loo=1;
end
if xval.loo==0;         % if not doing loo, specify number of training samples per class
    if ~isfield(xval,'s0_trn'), error('must specify xval.s0_trn'); end
    if ~isfield(xval,'s1_trn'), error('must specify xval.s1_trn'); end
    if ~isfield(xval,'num_iters'), error('must specify xval.num_iters'); end
end

siz=size(constants.y0); if siz(1)==1, y0=constants.y0; else y0=constants.y0'; end % make vectors into the correct orientation
siz=size(constants.y1); if siz(1)==1, y1=constants.y1; else y1=constants.y1'; end


%% select training and testing data

if xval.loo==true           % when doing loo
    y0trn=y0;               % initialize vector of samples in class 0
    y1trn=y1;               % initialize vector of samples in class 1
    if any(i==y0)           % is the test sample is in class 0
        y0trn(y0==i)=[];    % remove it from the training list
        y0tst=i;            % add it to the test list
        y1tst=[];           % nothing is in the class 1 test list
    else                    % or do this for class 1
        y1trn(y1==i)=[];
        y1tst=i;
        y0tst=[];
    end
else                        % randomly sample s0_trn and s1_trn data points for training, and use the others as testing
    if ~isfield(xval,'s0_tst'), xval.s0_tst=constants.s0-xval.s0_trn; end % specify # of test samples for class 0
    if ~isfield(xval,'s1_tst'), xval.s1_tst=constants.s1-xval.s1_trn; end % specify # of test samples for class 1
    
    ind0  = randperm(constants.s0);                             
    ind1  = randperm(constants.s1);
    
    y0trn = y0(ind0(1:xval.s0_trn));                            % randomly sample trn samples for class 0
    y0tst = y0(ind0(xval.s0_trn+1:xval.s0_trn+xval.s0_tst));    % randomly sample tst samples for class 0
    
    y1trn = y1(ind1(1:xval.s1_trn));                            % and class 1
    y1tst = y1(ind1(xval.s1_trn+1:xval.s1_trn+xval.s1_tst));
end

%% generate output

inds.ytrn=[y0trn y1trn];                                        % list of trn samples
inds.ytst=[y0tst y1tst];                                        % list of tst samples

Atrn = As(:,:,inds.ytrn);                                       % array of trn graphs
Atst = As(:,:,inds.ytst);                                       % array of tst graphs

ytrn = [zeros(1,length(y0trn)) ones(1,length(y1trn))];
ytst = [zeros(1,length(y0tst)) ones(1,length(y1tst))];

Gtrn = get_constants(Atrn,ytrn);                                % get trn constants
Gtst = get_constants(Atst,ytst);                                % get tst constants

if nargout==2                                                   % i forgot when this is useful, apparently, not that frequently
    inds.y0trn = y0trn;
    inds.y1trn = y1trn;
    inds.y0tst = y0tst;
    inds.y1tst = y1tst;
    inds.s_tst = length(inds.ytst);
end