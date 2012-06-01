function SigMat = get_fisher(As,constants)
% this function gets the significance values using Fisher's exact test
% and outputs a matrix of appropriate size

Areshaped=reshape(As,[constants.n^2 constants.s]);  % make a data matrix
siz=size(constants.ys); if siz(2)>siz(1), ys=constants.ys'; else ys=constants.ys; end
p=fexact(Areshaped',ys);                            % compute p-values
SigMat=reshape(p,[constants.n constants.n]);        % reshape into SigMat
