function [incorrect yhat] = plugin_bern_classify(datum,params,subspace,ytrue)
% this script implements the bayes plugin classifier, using a (sub-)space
% specified by user
%
% INPUT:
%   datum:      data to be classified
%   params:     a structure containing parameters for the bernoulli plugin classifier
%   subspace:   indices of features to use
%   ytrue:      (optional) true y
%
% OUTPUT:
%   incorrect:  whether 
%   yhat:       list of estimated class identity for each graph

data_tmp=datum(subspace);

post0=sum(data_tmp.*params.lnE0(subspace)+(1-data_tmp).*params.ln1E0(subspace))+params.lnprior0;
post1=sum(data_tmp.*params.lnE1(subspace)+(1-data_tmp).*params.ln1E1(subspace))+params.lnprior1;

[~, bar] = sort([post0, post1]); % find the bigger one
yhat=bar(2)-1;

if nargin==4, 
    incorrect=yhat~=ytrue; 
else
    incorrect=[];
end
