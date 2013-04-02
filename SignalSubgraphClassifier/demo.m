clear, clc

% select some parameters governing the sample data
s=100;                  % # subjects
n=10;                   % # vertices
pi0=0.5;                % prior probability of being in class 0
p0=rand(n);             % prob of edges in class 0
p1=p0;                  % prob of edge in class 1 (a little different from class 0)
SignalSubgraph=2:n;     % the signal subgraph has 1 vertex and n-1 edges    
p1(SignalSubgraph)=p1(SignalSubgraph)+randn(1,n-1)*0.1;   
p1(p1<=0)=1/(10*s);     % make sure probs are away from boundaries
p1(p1>=10)=1-1/(10*s);  % make sure probs are away from boundaries

% plot the parameters
figure(1), clf
subplot(131), imagesc(p0)
subplot(132), imagesc(p1)
subplot(133), imagesc(abs(p0-p1))

% pre-allocate memory
ys=nan(s,1);
As=nan(n,n,s);

s=100;                  % # samples
for i=1:s
   ys(i)=rand>pi0;      % samples classes
   if ys(i)==1          % sample UNDIRECTED graphs
       As(:,:,i)=tril(rand(n)>p1,-1);
   else
       As(:,:,i)=tril(rand(n)>p0,-1);
   end
    
end

% set some constraints on the signal subgraph
constraints{1}=NaN;         % use all edges
constraints{2}=round(n/2);  % use best 5 edges
constraints{3}=[1 n-1];     % use best 5 edges incident to 1 vertex
constraints{4}=[2 n];       % use best 10 edges incident to 2 vertices

% classify
[Lhat incorrects subspace] = xval_SigSub_classifier(As,ys,constraints,'loo');
