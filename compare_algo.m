%Comparison for different problem data (matrix A of different size) and different values of gamma 

clc, clear,close all

%Number of trials to run
ntrial = 1;

%Problem dimension
Ns     = ceil(linspace(100,2e3,15));

% Time to compute solution
Tsomm  = zeros(length(Ns),ntrial);
Tsommv = zeros(length(Ns),ntrial);
Tsparsa= zeros(length(Ns),ntrial);
Tista  = zeros(length(Ns),ntrial);
Tforbes  = zeros(length(Ns),ntrial);

w=0.15:0.10:0.85;

%%
for d=1:length(w)
    for i = 1:ntrial
        for j = 1: 1: length(Ns)
        
            % Problem dimensions
            n = Ns(j);
            m = 2*n;
            
            % Generate problem data
            A = randn(m,n);
            b = randn(m,1);
            
            % Penalty parameter
            gamma_max = max(A'*b);
            gamma = w(d)*gamma_max;
            
            % Solver options
            options = struct('rho',10,'maxiter',500,...
                'eps_MM',1.e-8,'eps_dy',1.e-3,'ssone',1);
            rel_tol = options.eps_MM;
            
            % Code for changing the condition number of A'*
            [U,S,V] = svd(A);
            l = diag(S);
            ml = min(l);
            scale = 1;
            l = (l - ml)*scale + ml;
            A = U*[diag(l); zeros(size(S,1)-length(l),length(l))]*V';
            
            AtA = A'*A;
            Atb = A'*b;
            xc = AtA\Atb;
            
            % Compute centralized solution
            Atb = A'*b;
            xc = (A'*A)\(A'*b);
            
            % SOMM
            tic
            ansf = lassoMM2nd_fn_altmerit_eff(A,AtA,b,Atb,gamma,options,xc);
            Tsomm(j,i) = toc;
            xsp = ansf.xsp;
            Fsomm(j,i) = .5*norm(A*xsp - b)^2 + gamma*norm(xsp,1);
            
            % SOMM V
            options.rho = 10;
            tic
            ansfv = lassoMM2nd_V_eff(A,AtA,b,Atb,gamma,options,xc);
            Tsommv(j,i) = toc;
            xspv = ansfv.xsp;
            Fsommv(j,i) = .5*norm(A*xspv - b)^2 + gamma*norm(xspv,1);
            
            % ISTA
            tic
            [xista, istastats] = lasso_ista(AtA,Atb,gamma,xc,options);
            Tista(j,i) = toc;
            Fista(j,i) = .5*norm(A*xista - b)^2 + gamma*norm(xista,1);
            
            % FISTA
            tic
            [xfista, fistastats] = lasso_fista(AtA,Atb,gamma,xc,options);
            Tfist(j,i) = toc;
            Ffist(j,i) = .5*norm(A*xfista - b)^2 + gamma*norm(xfista,1);
            
            % ISTA
            options.rho = 1;
            tic
            [xista, istastats] = admm_ista(A,b,gamma,options,n);
            Tista(j,i) = toc;
            Fista(j,i) = .5*sum_square(A*xista - b) + gamma*norm(xista,1);
            
            %CVX Verification
            tic;
            cvx_begin quiet
            cvx_precision best
            variable xcvx(n);
            minimize (0.5*sum_square(A*xcvx - b) + gamma*norm(xcvx,1));
            cvx_end
            cvx_toc=toc;
            
            %ForBES
            tic
            f = quadLoss(1, zeros(m,1));
            aff = {A, -b};
            g = l1Norm(gamma);
            opt.maxit = options.maxiter;
            opt.tol = 1e-5;;
            opt.toRecord='1';
            opt = opt;
            v=forbes(f, g, xc, aff, [],opt);
            Tforbes(j,i) = toc;
            Fforbes(j,i) =.5*sum_square(A*v.solver.x - b) + gamma*norm(v.solver.x,1);
            
        end
    end
    %  Computation time for each method
    fprintf('Computation times (s)\n')
    fprintf('SOMM \t SOMMv \t ISTA \t Forbes \n%2.2f \t %2.2f \t %2.2f \t %2.2f \n\n',...
        [Tsomm Tsommv Tista Tforbes]);
    
    % take SpaRSA solution as 'ground truth'
    xsol = xsp;
    
    % calculate distance from ground truth at every iteration
    xsommres = norms(ansf.xhist - xsol*ones(1,size(ansf.xhist,2)));
    xforbesres = norms(v.solver.xhist - v.solver.x*ones(1,size(v.solver.xhist,2)));
    xfistares = norms(fistastats.xhist - xsol*ones(1,size(fistastats.xhist,2)));
    xistares = norms(istastats.xhist - xsol*ones(1,size(istastats.xhist,2)));
    
    xsommres = xsommres(1:ansf.MMstep);
    xforbesres=xforbesres(1:v.solver.iterations);
    xfistares = xfistares(1:fistastats.steps);
    xistares = xistares(1:istastats.steps);
    
    %Plot
    semilogy(1:ansf.MMstep,xsommres,...
        1:v.solver.iterations,xforbesres,...
        1:fistastats.steps,xfistares,1:istastats.steps,xistares)
    legend('SOMM','ForBES','FISTA','ISTA')
    axis tight
    xlabel('Number of iterations')
    ylabel('|| x - x* ||')
    title('gamma=gamma max*0.15')
    figure

end

