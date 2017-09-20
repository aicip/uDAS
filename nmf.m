function [W,H] = nmf(V,Winit,Hinit,c,tol,maxiter)

% NMF by alternative non-negative least squares using projected gradients
% Author: Chih-Jen Lin, National Taiwan University

% W,H: output solution
% Winit,Hinit: initial solution
% c: number of hidden note
% tol: tolerance for a relative stopping condition
% timelimit, maxiter: limit of time and iterations

W = Winit; H = Hinit;
gradW = W*(H*H') - V*H'; 
gradH = (W'*W)*H - W'*V;
initgrad = norm([gradW; gradH'],'fro');
fprintf('Init gradient norm %f\n', initgrad); 
tolW = max(0.001,tol)*initgrad; tolH = tolW;
converse = [];
for iter=1:maxiter,
  % stopping condition
  projnorm = norm([gradW(gradW<0 | W>0); gradH(gradH<0 | H>0)]);
  if projnorm < tol*initgrad,
    break;
  end
  
  [W,gradW,iterW] = nlssubprob(V',H',W',tolW,1000); W = W'; gradW = gradW';
  if iterW==1,
    tolW = 0.1 * tolW;
  end

  % to consider the sum-to-one constraint
  N = size(V,2);
  tV = [V; 20*ones(1,N)];
  tW = [W; 20*ones(1,c)];
  [H,gradH,iterH] = nlssubprob(tV,tW,H,tolH,1000);
%   error2 = V - W*H;
%   errorsum = sum(abs(error2(:)));
%   converse = [converse errorsum];

%   [H,gradH,iterH] = nlssubprob(V,W,H,tolH,1000);

  if iterH==1,
    tolH = 0.1 * tolH; 
  end

  if rem(iter,10)==0, fprintf('.'); end
end
fprintf('\nIter = %d Final proj-grad norm %f\n', iter, projnorm);

function [H,grad,iter] = nlssubprob(V,W,Hinit,tol,maxiter)

% H, grad: output solution and gradient
% iter: #iterations used
% V, W: constant matrices
% Hinit: initial solution
% tol: stopping tolerance
% maxiter: limit of iterations

H = Hinit; WtV = W'*V; WtW = W'*W; 

alpha = 1; beta = 0.1;
for iter=1:maxiter,  
  grad = WtW*H - WtV;
  projgrad = norm(grad(grad < 0 | H >0));
  if projgrad < tol,
    break
  end

  % search step size 
  for inner_iter=1:20,
    Hn = max(H - alpha*grad, 0); d = Hn-H;
    gradd=sum(sum(grad.*d)); 
    dQd = sum(sum((WtW*d).*d));
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
