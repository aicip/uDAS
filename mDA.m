function [hx, W] = mDA(xx,noise,lambda)

% xx : dxn input
% noise: corruption level
% lambda: regularization

% hx: dxn hidden representation
% W: dx(d+1) mapping

[d, n] = size(xx);
% adding scaled bias
% xxb = [xx; ones(1, n)./((1-noise))];
xxb = [xx; ones(1, n)];

% scatter matrix S
S = xxb*xxb';

% corruption vector
q = ones(d+1, 1)*(1-noise);
q(end) = 1;

% Q: (d+1)x(d+1)
Q = S.*(q*q');
Q(1:d+2:end) = q.*diag(S); % change the diagno

% P: dx(d+1)
P = S(1:end-1,:).*repmat(q', d, 1);


% final W = P*Q^-1, dx(d+1);
reg = lambda*eye(d+1);
reg(end,end) = 0;
W = P/(Q+reg);

% hx = W*xxb;
hx = (1-noise)*W*xxb;
% hx = tanh(hx);
% hx = sigmf(hx,[1,0]);
% hx = 1.0 ./ (1.0 + exp(-hx));

% sigmoid


