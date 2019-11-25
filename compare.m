%Compare algorithms for given problem data (matrix A and max gamma), and different values of gamma

clc, clear, close all

Solver options
options = struct('rho',10,'maxiter',500,...
   'eps_MM',1.e-8,'eps_dy',1.e-3,'ssone',1);

w=0.15:0.10:0.85;

for d=1:length(w)
    gamma = w(d)*gamma_max;
    
    % SOMM
    ansf = lassoMM2nd_fn_altmerit_eff(A,AtA,b,Atb,gamma,options,xc);
    xsp = ansf.xsp;
   
    %ForBES
    f = quadLoss(1, zeros(m,1));
    aff = {A, -b};
    g = l1Norm(gamma);
    opt.maxit = options.maxiter;
    opt.tol = options.eps_MM;
    opt.toRecord='1';
    opt = opt;
    v=forbes(f, g, xc, aff, [],opt);
    
    %ISTA
    [xista, istastats] = lasso_ista(AtA,Atb,gamma,xc,options);
    
    %FISTA
    [xfista, fistastats] = lasso_fista(AtA,Atb,gamma,xc,options);
    
    %Calculate ||x-x*||
    xsol =xsp;
    
    xsommres = norms(ansf.xhist - xsol*ones(1,size(ansf.xhist,2)));
    xforbesres = norms(v.solver.xhist - xsol*ones(1,size(v.solver.xhist,2)));
    xfistares = norms(fistastats.xhist - xsol*ones(1,size(fistastats.xhist,2)));
    xistares = norms(istastats.xhist - xsol*ones(1,size(istastats.xhist,2)));
    
    xsommres = xsommres(1:ansf.MMstep);
    xforbesres=xforbesres(1:v.solver.iterations);
    xfistares = xfistares(1:fistastats.steps);
    xistares = xistares(1:istastats.steps);
    
    figure
    %Plot
    semilogy(1:ansf.MMstep,xsommres,'r',...
        1:v.solver.iterations,xforbesres,'g',...
        1:fistastats.steps,xfistares,'b',1:istastats.steps,xistares,'m--')
    legend('SOMM','ForBES','FISTA','ISTA')
    axis tight
    xlabel('Number of iterations')
    ylabel('|| x - x* ||')
    title("gamma="+w(d)+"*gamma max")
    ylim([1e-8 1])
    figure
        %Plot
    semilogy(ansf.time,xsommres,'r',...
       v.solver.ts,xforbesres,'g',...
       fistastats.time,xfistares,'b',istastats.time,xistares,'m--') 
    legend('SOMM','ForBES','FISTA','ISTA')
    axis tight
    xlabel('Solver time')
    ylabel('|| x - x* ||')
    title("gamma="+w(d)+"*gamma max")
    ylim([1e-8 1])
end
