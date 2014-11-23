function [SigSub, wcounter] = signal_subgraph_estimator(SigMat,constraints,egg)
% estimates the signal subgraph from SigMat using the constraints
%
% INPUT:
%   SigMat: n x n matrix of Significance (only lower triangle perhaps for undirected graphs
%   constraints: if 2x1, then it is [num_vertices num_edges]=constraints,  else it is just num_edges
%   egg: (optional) whether to do egg
% 
% OUTPUT:
%   SigSub:     the signal subgraph
%   wcounter:   necessary for plotting coherogram when using coherent (or egg) 


if nargin==1, constraints=sqrt(numel(SigMat)); end

num_constraints=length(constraints);
if num_constraints==1                   % if using incoherent or naive bayes
    num_edges=constraints;
elseif num_constraints==2               % if using coherent or egg
    num_vertices=constraints(1);
    num_edges=constraints(2);
end

if num_constraints==1                   % use incoherent or naive bayes
    
    if isnan(constraints)               % use naive bayes
        SigSub=1:numel(SigMat);
    else                                % use incoherent
        [~, delind]=sort(SigMat(:),'ascend');
        SigSub  = delind(1:num_edges);
    end
    wcounter=[];
    
elseif num_constraints==2 && nargin<3   % use coherent
    wset=unique([0; sort(SigMat(:))]);  % set of unique pvals from SigMat
    wset(wset>1-1e-3)=[];               % ignore p-values sufficiently close to 1
    wcounter = 1;
    [V ~] = size(SigMat);               % # of vertices
    
    wconv=0;
    while wconv==0                      % while not converged
        
        w=wset(wcounter);               % increment the threshold
        blank=SigMat;                   % generate a matrix with 1's for all elements below threshold
        blank(blank>w)=1;
        blank(blank<=w)=0;
        score=V*2-(sum(blank)+sum(blank,2)'); % score each vertex as a function of how many edges incident to it are significant
        [vscore, vstars] = sort(score,'descend');
        
        if sum(vscore(1:num_vertices))>=num_edges   % if there are enough edges incident to enough vertices
            
            blank=0*SigMat+1;                                           % generate matrix with 1's for all candidate edges
            nstars=min(length(find(vscore>0)),num_vertices);            % count number of signal vertices
            blank(vstars(1:nstars),:) = SigMat(vstars(1:nstars),:);     
            blank(:,vstars(1:nstars))= SigMat(:,vstars(1:nstars));
            [~, indsp] = sort(blank(:));                                % sort the candidate edges
            
            SigSub=indsp(1:num_edges);                                  % keep the num_edge most significant edges from the candidate set
            wconv=1;                                                    % converge
        else                                        % if there are not enough significant edges
            wcounter=wcounter+1;                    % increment the threshold
            if wcounter>length(wset),               % if we've reached the largest threhold
                SigSub=[];                          % then we found no signal subgraph for the specified hyper-parameters
                wconv=1;                            % and we converge
            end
        end % if there were enought edges
        
    end % while we did not converge
    wcounter=wcounter-1;
    
elseif nargin==3 && egg==1              % use egg (similar to coherent, perhaps comments to come)
    
    wset=unique(sort(SigMat(:)));
    wset(wset>1-1e-3)=[];
    wcounter = 1;
    siz = size(SigMat);
    V = siz(1);
    
    wconv=0;
    while wconv==0
        
        w=wset(wcounter);
        inds = find(SigMat<=w);
        [I, J] = ind2sub(siz,inds);
        ncounts = histc([I; J],1:V);
        [B, IX] = sort(ncounts,'descend');
        sumcounts = sum(B(1:num_stars));
        if sumcounts>=num_edges
            blank=0*SigMat+1;
            blank(IX(1:num_stars),IX(1:num_stars))=SigMat(IX(1:num_stars),IX(1:num_stars));
            [~, indsp] = sort(blank(:));
            SigSub=indsp(1:num_edges);
            wconv=1;
        else
            wcounter=wcounter+1;
            if wcounter>numel(wset),
                blank=0*SigMat+1;
                blank(IX(1:num_stars),IX(1:num_stars))=SigMat(IX(1:num_stars),IX(1:num_stars));
                [~, indsp] = sort(blank(:));
                SigSub=indsp(1:num_edges);
                wconv=1;
            end
        end
        
    end % while wconv
    
end