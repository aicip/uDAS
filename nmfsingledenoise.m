function [W, H,W1,converse] = nmfsingledenoise(V,Winit,Hinit,c,tol,maxiter,Wnoise,W1,gamma,lambda,input)

% NMF by alternative non-negative least squares using projected gradients
% Author: Chih-Jen Lin, National Taiwan University
% Modified by Ying Qu, The University of Tennessee, Knoxville

% W,H: output solution
% Winit,Hinit: initial solution
% c: number of hidden note
% tol: tolerance for a relative stopping condition
% timelimit, maxiter: limit of time and iterations

addpath minFunc/
options.Method = 'CG';    % Here, we use L-BFGS to optimize our cost
options.maxIter = 1;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';
theta = W1(:);
converse = [];

W = Winit; H = Hinit;
% add the denoise constraints
[d, n] = size(V);
Vaug = [input; ones(1, n)];
gradnoise = (W*H-Wnoise*Vaug)*H';
gradW = W*(H*H') - V*H' + gamma*gradnoise;
initgrad = norm(gradW','fro');
fprintf('Init gradient norm %f\n', initgrad); 
tolW = max(0.001,tol)*initgrad; 

for iter=1:maxiter,
  % stopping condition
  projnorm = norm(gradW(gradW<0 | W>0));
  if projnorm < tol*initgrad,
    break;
  end
  [W,gradW,iterW] = nlssubprob(V',H',W',tolW,1000,Wnoise,W1,gamma,input'); W = W'; gradW = gradW';
  if iterW==1,
    tolW = 0.1 * tolW;
  end
  
  % update W1 with l21 constraints
  [cost, grad] = sparseAutoencoderCostbpw1dn(theta, d, c, lambda, input,V, W);
  [opttheta, cost] = minFunc( @(p) sparseAutoencoderCostbpw1dn(p, ...error
                                       d, c, ...
                                       lambda,input,V,W), ...
                                  theta, options);
   W1 = reshape(opttheta(1:c*d), c, d);
   theta = W1(:);
   H = max(W1*input,0);
  if rem(iter,10)==0, fprintf('.'); end
end
fprintf('\nIter = %d Final proj-grad norm %f\n', iter, projnorm);

function [H,grad,iter] = nlssubprob(V,W,Hinit,tol,maxiter,Wnoise,W1,gamma,input)
% H, grad: output solution and gradient
% iter: #iterations used
% V, W: constant matrices
% Hinit: initial solution
% tol: stopping tolerance
% maxiter: limit of iterations
H = Hinit; WtV = W'*V; WtW = W'*W; 
[d, n] = size(input');
Vaug = [input'; ones(1, n)];

alpha = 1; beta = 0.1;
for iter=1:maxiter 
  % add the denoise constraints
  gradnoise = (H'*W'-Wnoise*Vaug)*W; 
  grad = WtW*H - WtV + gamma*gradnoise';
  
  projgrad = norm(grad(grad < 0 | H >0));
  if projgrad < tol,
    break
  end

  % search step size 
  for inner_iter=1:20,
    Hn = max(H - alpha*grad, 0); 
    Hn = min(Hn, 1); 
    d = Hn-H;
    gradd=sum(sum(grad.*d));
    W1V = W';
    dQd = sum(sum(((WtW + gamma*W1V*W1V')*d).*d));
    % check if the funciton satisfy the condition 
    suff_decr = 0.99*gradd + 0.5*dQd < 0;
    % if the function satisfy the condition, then we can use it 
    if inner_iter==1,
      decr_alpha = ~suff_decr; Hp = H;
    end
    if decr_alpha, 
      if suff_decr,
        H = Hn; break;
      else
	alpha = alpha * beta;
      end
    else
      if ~suff_decr | Hp == Hn,
	H = Hp; break;
      else
	alpha = alpha/beta; Hp = Hn;
      end
    end
  end
end

if iter==maxiter,
  fprintf('Max iter in nlssubprob is %d \n', iter);
end
