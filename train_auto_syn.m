%%======================================================================
%% Written by Ying Qu <yqu3@vols.utk.edu>, September 2017
%% All Rights Reserved
%% uDAS: An Untied Denoising Autoencoder with Sparsity for Spectral Unmixing
%% Please cite the following paper
%% 
% @article{qu2019udas,
%   title={uDAS: An untied denoising autoencoder with sparsity for spectral unmixing},
%   author={Qu, Ying and Qi, Hairong},
%   journal={IEEE Transactions on Geoscience and Remote Sensing},
%   volume={57},
%   number={3},
%   pages={1698--1712},
%   year={2019},
%   publisher={IEEE}
% }
% @article{qu2017spectral,
%   title={Spectral unmixing through part-based non-negative constraint denoising autoencoder},
%   author={Qu, Ying and Guo, Rui and Qi, Hairong},
%   journal ={2017 IEEE International Geoscience and Remote Sensing Symposium (IGARSS)},
%   pages={209--212},
%   year={2017},
%   organization={IEEE}
% }
%%======================================================================

% load hyperspectral data
load used_data/toy5n20
[M,N,D] = size(mixed);
signal = reshape(mixed,M*N,D);

visibleSize = size(signal,2);
c = 5; % Layer 1 Hidden Size
SNR = 20;

% estimate denoise constraints
[mixed,w12] = mDA(signal',0.00001,1); % for noise level 20   
w12noise = w12(:,1:end-1);
bpre = w12(:,end);
wb = bpre;
iter = 11; % for SNR = 20db iter = 11; for SNR = 10, iter = 11
for level = 1:iter
    bpre = w12(:,end);
    [mixed,w12] = mDA(mixed,0.00001,1); 
    w12tmp = w12(:,1:end-1);
    btmp = w12(:,end);
    w12noise = w12tmp*w12noise;
    wb =  w12tmp*bpre + btmp;
end
% w12noise = [w12noise wb];
w12noise = [w12noise wb].*0.99^(iter);
tmp = w12noise*[mixed;ones(1,size(mixed,2))];
temp = W*abf;
w12noise = w12;

input = signal';
denoised = mixed;
datasize = size(mixed,2);

% initialize with vca 
n = M*N;
Y = mixed;
my = mean(Y,2);
Y = Y-repmat(my,1,n);
[Uq,DE] = svd(Y*Y'/n);
Uq = Uq(:,1:c-1);    
sing_values = diag(DE);

% define affine projection of dimension q-1 and lift
aff_proj = @(Z) [ Uq'*Z; ones(1,size(Z,2))];

% compute the coordinates of Y (centered) wrt Uq and lift
Y = aff_proj(Y);
VCA_runs = 100;
small = 1e-4;
lam_sphe_reg  = 1e-6;
vol_ant = -inf;
verbose = 'no';

C = diag([1./(sqrt(sing_values(1:c-1))+lam_sphe_reg); 1]);
% spherize data
Y = C*Y;

for i=1:VCA_runs
    Aux = vca(Y,'Endmembers',c,'SNR',1,'verbose',verbose);
    %Aux = svmax(Y,'Endmembers',q,'SNR',1,'verbose',verbose);
    vol = sum(log(svd(denoised) + small ));
    if vol > vol_ant
        Avca = Aux;
        vol_ant = vol;
    end
end

W_vca = inv(C)*Avca;
W_vca = Uq*W_vca(1:c-1,:);
W_vca = W_vca + repmat(my,1,c);

% % FCLS
% warning off;
AA = [1e-5*W_vca;ones(1,length(W_vca(1,:)))];
h_fcls = zeros(length(W_vca(1,:)),datasize);
for j=1:datasize
    r = [1e-5*denoised(:,j); 1];
    h_fcls(:,j) = lsqnonneg(AA,r);
end
% legend('corrupted pixel','denoised pixel');

%%======================================================================
%  Initialize parameters w (weight) and h (hidden layer) using NMF
% ======================================================================
[W,H]=nmf(denoised,W_vca,h_fcls,c,1e-6,1);
W_vca = W;
h_fcls = H;

initW1 = H*pinv(input);
% ======================================================================

theta = initW1(:);
W1 = initW1;
converse = [];
step = 10;
gamma = 1e-4;  % denoise level
lambda = 1e-6*sqrt(n/1000);   % l21 weight decay parameter  
breakiter = 0;

% main code
[W, H,converse] =nmfsingledenoise(denoised,W_vca,h_fcls,c,1e-6,step,w12noise,W1,gamma,lambda, input);     


W1sum = sum((abs(W1)),2)./(M*N)
if min(W1sum)==0
    disp('please deduct the number of endmember')
end

% evaluate the results
load used_data/A;
load used_data/BANDS;
A = A(BANDS,1:c);
% % permute results
CRD = corrcoef([A W]);
DD = abs(CRD(c+1:2*c,1:c));  
perm_mtx = zeros(c,c);
aux=zeros(c,1);
for i=1:c
    [ld cd]=find(max(DD(:))==DD); 
    ld=ld(1);cd=cd(1); % in the case of more than one maximum
    perm_mtx(ld,cd)=1; 
    DD(:,cd)=aux; DD(ld,:)=aux';
end
W = W*perm_mtx;
H = H'*perm_mtx;
H = H';

% % % rmse error of abundances
E_rmse = sqrt(sum(sum(((abf-H).*(abf-H)).^2))/(M*N*c))

% the angle between abundances
nabf = diag(abf*abf');
nsest = diag(H*H');
ang_beta = 180/pi*acos( diag(abf*H')./sqrt(nabf.*nsest));
E_aad = mean(ang_beta.^2)^.5

% cross entropy between abundance
E_entropy = sum(abf.*log((abf+1e-9)./(H+1e-9))) + sum(H.*log((H+1e-9)./(abf+1e-9)));
E_aid = mean(E_entropy.^2)^.5

% the angle between material signatures
nA = diag(A'*A);
nAest = diag(W'*W);
ang_theta = 180/pi*acos( diag(A'*W)./sqrt(nA.*nAest) );
E_sad = mean(ang_theta.^2)^.5

% the spectral information divergence
pA = A./(repmat(sum(A),[length(A(:,1)) 1]));
qA = W./(repmat(sum(W),[length(A(:,1)) 1])); 
qA = abs(qA);
SID = sum(pA.*log((pA+1e-9)./(qA+1e-9))) + sum(qA.*log((qA+1e-9)./(pA+1e-9)));
E_sid = mean(SID.^2)^.5
                  

