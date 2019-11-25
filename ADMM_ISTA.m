function [xadmm, stats] = admm_ista(A,b,xc,gamma,opts,m,n)
rho = opts.rho;

% Identity matrix
I = eye(n);

% Initial conditions
maxiter = 1000;
y = zeros(n,1); % Lagrange multiplier
z = zeros(n,1); % copy of x
x = xc;
b = A*xc;

eps_abs = 1e-4;
eps_rel = 1e-2;
stats.res = zeros(maxiter);
%stats.time = zeros(maxiter,1);
AtA = A'*A;
Atb = A'*b;
% Use ADMM to solve the lambda-parameterized problem
tic  % start timer
for ADMMstep = 1 : maxiter
    
    % x-minimization step
    u = A'*b + rho*(z - y);    % temporary value
    xnew = (A'*A + rho*I) \ u; 
    
    % z-minimization step       
    a = (gamma/rho)*ones(n,1);
    v = xnew + y;
   
    % Soft-thresholding of v
    znew = ( (1 - a ./ abs(v)) .* v ) .* (abs(v) > a); 
    stats.xhist(:,ADMMstep) = xnew;
    % Primal and dual residuals
    res_prim = norm(xnew - znew);
    res_dual = rho * norm(znew - z);
    
    % Lagrange multiplier update step
    y = y + rho*(xnew - znew);
        
    % Stopping criteria
    eps_prim = sqrt(n) * eps_abs + eps_rel * max([norm(xnew),norm(znew)]);
    eps_dual = sqrt(n) * eps_abs + eps_rel * norm(y);
    
    if (res_prim < eps_prim) && (res_dual < eps_dual)
        break;
    else
        z = znew;
    end
    stats.time(ADMMstep) = toc;
end
stats.time(ADMMstep) = toc;
xadmm=xnew;
stats.xhist(:,ADMMstep) = xnew;
stats.objval(ADMMstep+1) = (0.5*sum_square(A*xnew - b) + gamma*norm(znew,1));
stats.steps = ADMMstep;


