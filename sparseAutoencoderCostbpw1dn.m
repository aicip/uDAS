function [cost,grad] = sparseAutoencoderCostbpw1dn(theta, visibleSize, hiddenSize, ...
                                             lambda,data,denoised,W2)

% modified by Ying Qu
% theta: initilized w1                                        
% visibleSize: the number of input units 
% hiddenSize: the number of hidden units 
% lambda: weight decay parameter
% data: input data
% denoised: denoised data
% W2: weight of the last layer 

N = size(data,2);
c = size(W2,2);
deta = 20;
tdata = [denoised; deta*ones(1,N)];
W2 = [W2; deta*ones(1,c)];

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W1grad = zeros(size(W1));

z2 = W1*data;
a2 = relu(z2);

dataout = W2*a2;


Jwhole = (0.5/N)*sum(sum((tdata-dataout).^2));
Jweight = (0.5)*(sum(sqrt(sum(W1.^2,2))));
cost = Jwhole + lambda*Jweight;

dwhole = -(tdata-dataout);
dwholeback = W2'*dwhole;
dwholebackrelu = dwholeback.* reluInv(a2);

W1grad = W1grad + (dwholebackrelu)*data';
W1grad = (1/N)*W1grad + lambda*W1.*(repmat(sum(W1.^2,2),1,size(W1grad,2))).^(-0.5);

grad = W1grad(:);

end

%-------------------------------------------------------------------
%relu function
function sigm = relu(x)
    sigm = max(x,0);
end

%inverse relu
function sigmInv = reluInv(x)
    idx1 = find(x < 0);
    sigmInv = x;
    sigmInv(:) = 1;    
    sigmInv(idx1) = 0;
end
