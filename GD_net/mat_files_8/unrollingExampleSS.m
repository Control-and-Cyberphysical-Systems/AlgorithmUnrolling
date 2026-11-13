% clear; close all; clc;
% Using this seed results in the mpQP parameters used in the paper
% Removing it yields similar results with different noise realizations.
rng(2)

% DPC in state-space setting
% Discrete-time double integrator dynamics with dt = 0.1
A = [1, 1; 0, 1]; B = [0.5; 1];
n = 2; m = 1;

% Parameters of white additive Gaussian measurement noise
mu_x = zeros(n,1);
sigma_x = sqrt(0.001);

% DPC tuning parameters
lambdaPi = 0.01;
lambdaPerp = 1000;

% Constraints
xlb = [-25; -5];
xub = [25; 5];
ulb = -1;
uub = 1;

% Control objective design parameters
Q = eye(n);
R = 0.1;
N = 8;

% LQR solution 
[P, minusK, ~] = idare(A, B, Q, R, [], []);
K_LQR = -minusK;

% Construct multi-step versions of weight matrix
% Note that P is usually not available in a data-driven setting and we only
% used it for potential comparison with a nominal MPC scheme.
% Nonetheless, using a different weight (e.g., P = c*Q for some c >= 1) 
% does not change the structure of the learned mpQP though.
QQ = kron(eye(N), Q);
QQ(end-n+1:end, end-n+1:end) = P;
RR = kron(eye(N), R);

% Specify length for data collection
Ndsquare = N+n*(N+1)+m*N-1; % minimum time-steps to get square data matrix
Nd = Ndsquare+5; % use a few more time-steps to get slightly wider data matrix

% Generate data
x0 = zeros(n,1);
ud = zeros(m,Nd);
xd = [x0,zeros(n,Nd)];
for i = 1:Nd
    % Recall that the double integrator is only marginally stable. We want
    % to avoid the state trajectory diverging too far during data 
    % collection, since that creates "too good" state data, i.e., data with
    % very high SNR. We therefore added some pre-stabilization, where the
    % choice 0.5*K_LQR is rather arbitrary. 
    ud(:,i) = rand(m,1)*(uub-ulb)+ulb+0.5*K_LQR*xd(:,i);
    % Similarly, clip the inputs to avoid "too good" input data.
    if ud(:,i) > uub
        ud(:,i) = uub;
    elseif ud(:,i) < ulb
        ud(:,i) = ulb;
    end
    xd(:,i+1) = A*xd(:,i)+B*ud(:,i);
end

% Add measurement noise
xd = xd + randn(n,Nd+1)*sigma_x+mu_x;
% Construct block-hankel matrices from data
Hu = blockHankelMat(ud,N);
Hx = blockHankelMat(xd,N+1);

% Retrieve data matrices used for image representation
Uf = Hu;
X0 = Hx(1:n,:);
Xf = Hx(n+1:end, :);
Phi = [X0; Uf];

% Define LS Predictors / Controllers / Residuals
% used in regularization
K = Xf*pinv(Phi);
Kx = K(:, 1:n);
Ku = K(:, n+1:end);
Lx = Uf*pinv(X0);

dX = Xf-K*Phi;
dU = Uf-Lx*X0;
Qreg = inv(dX*dX');
Rreg = inv(dU*dU');

% Construct mpQP parameters
S = [-Ku, eye(n*N)];
T = [eye(m*N), zeros(m*N, n*N)];

H = 2*blkdiag(RR+lambdaPi, QQ) + 2*lambdaPerp*S'*Qreg*S;
F = -2*(lambdaPerp*S'*Qreg*Kx+lambdaPi*T'*Rreg*Lx);
G = [eye((m+n)*N); -eye((m+n)*N)];
E = zeros(2*(m+n)*N, n);
d = [kron(ones(N,1), uub);
    kron(ones(N,1), xub);
    -kron(ones(N,1), ulb);
    -kron(ones(N,1), xlb)];

% Export mpQP parameters to .mat file
% save('QP_parameters_SS', "H", "F", "G", "E", "d", "Kx", "Ku")

