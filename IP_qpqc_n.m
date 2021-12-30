%##########################################################################
% This file constaints a Interior Point solver for quadratic constrait
% optimization problems with a quadratic cost function
% 
% ToDo:
%      - Debug Informationen einfügen, je nachdem wie hoch die Verbosität ist
%      - Ausgabe des Lösungsvorgangs um zu erkennen wo und warum das
%      Optimierungssproblem sich nicht lösen lässt
%      - Visualisierung des Optimierungsproblems
%
% Markus Bell 27.12.2019
%
%--------------------------METHOD DESCRIPTION START------------------------
%
% The method presented here is mainly based on the PhD thesis of Alexander
% Domahidi from ETH Zuerich. Some of the code is also adapted from an 
% optimal power flow paper of Florin Capinescu from the University of Lille 
%
% It resembles an Interior Point algorithm for optimization problems of 
% the form:
%
%                          min_{x} 1/2*x'H*x + f*x
%
%                           st.:
%                               Aequ*x = Bequ
%                               Aineq*x <= u
%                               x'*Aquad*x + lquad'*x <= rquad
%
% The Solution of the perturbed KKT equations is performed with a standard
% newton method
%
%--------------------------METHOD DESCRIPTION END--------------------------
%
%##########################################################################

function ip_result = IP_qpqc_n(optimmodel,options)
    
    %----------------------------------------------------------------------
    %Determine the sizes and amount of constraints
    %----------------------------------------------------------------------
    fprintf('====================================================================\n');
    fprintf('INFO: Optimization preprocessing start...\n');
    fprintf('====================================================================\n');
   
    %Number of linear inequality constraints
    minequ = size(optimmodel.Binequ,1);
	  mquad = size(optimmodel.rquad,1);
    %Number of linear equality constraints
    mequ = size(optimmodel.Bequ,1);
    
    fprintf(' \n');
    fprintf('INFO: Optimization Problem has %i quadratic inequality constraints\n', size(optimmodel.rquad,1));
    fprintf('INFO: Optimization Problem has %i linear inequality constraints\n', size(optimmodel.Binequ,1));
    fprintf('INFO: Optimization Problem has %i linear equality constraints\n', size(optimmodel.Bequ,1));
    fprintf(' \n');
    
    fprintf('====================================================================\n');
    fprintf('INFO: Checking initial state vector\n');
    fprintf('====================================================================\n');
    
    %Calcualt if the initial point is feasible with regards to the linear inequality constraints
    LinInequFeas = length(nonzeros(optimmodel.Ainequ*options.x0-optimmodel.Binequ>0));
     
    QuadInequFeas = 0;
    %Calculate if the initial point is feasible with regards to the quadratic inequality constraint
    for i=1:1:size(optimmodel.Aquad,3)
      QuadInequFeas = QuadInequFeas + length(nonzeros((options.x0'*optimmodel.Aquad(:,:,i)*options.x0+optimmodel.lquad'*options.x0-optimmodel.rquad)>0));
    end
   
    %If there are infeasible constraints from the beginning tell the user
    if LinInequFeas > 0 || QuadInequFeas > 0
      fprintf(' \n');
      fprintf('WARN: Optimization Problem has %i infeasible quadratic inequality constraints in regards to x0\n', QuadInequFeas);
      fprintf('WARN: Optimization Problem has %i infeasible linear inequality constraints  in regards to x0\n', LinInequFeas);
      fprintf(' \n');
    else
      fprintf(' \n');
      fprintf('INFO: x0 is a feaible starting vector\n', QuadInequFeas);
      fprintf(' \n');
    end
    %----------------------------------------------------------------------
    
    %----------------------------------------------------------------------
    %Initialize the Parameter Matrices (Second order)
    %----------------------------------------------------------------------
    
    %The Hessian Matrix (second Order derivative)
    Hess = zeros(size(optimmodel.f,2),size(optimmodel.f,2));
    %The first order derivative of the quadratic and linear inequalities
    G = zeros(minequ+mquad,size(optimmodel.f,2));
    %The Langrang multiplier Matrix for the inequality constraints
    M = zeros(minequ+mquad,minequ+mquad);
    %The slack variable matrix for the inequality constraints
    S = zeros(minequ+mquad,minequ+mquad);
    %Output vector (evaluation of the inequality constraints at the current iterate xk)
    g = zeros(minequ+mquad,1);
    %Identity matrix
    I = eye(minequ+mquad);
    %Parameter Matrix for the Newton step
    A_IP = zeros(size(optimmodel.f,2)+mequ+2*(minequ+mquad),size(optimmodel.f,2)+mequ+2*(minequ+mquad));
    %Output vector for the Newton step
    B_IP = zeros(size(optimmodel.f,2)+mequ+2*(minequ+mquad),1);
    %Dual residual
    rS = zeros(size(optimmodel.f,2),1);
    %Primal residual
    rE = zeros(mequ,1);
    %Inequality residual
    rI = zeros((minequ+mquad),1);
    %Slack variable residual
    rC = zeros((minequ+mquad),1);
    %----------------------------------------------------------------------
    
    
    %----------------------------------------------------------------------
    %Intialize the variable vectors
    %----------------------------------------------------------------------
    
    %Vector of the optimization variables
    xk = zeros(size(optimmodel.f,2),1);
    %Langrange multipliers for the equality constraints
    lambdak = zeros(mequ,1);
    %Langrange mulitpliers for the inequality constraints
    muk = zeros(minequ+mquad,1);
    %Slack variables for the inequality constraints
    sk = zeros(minequ+mquad,1);
    
    %Intialize delta variables
    %Vector of the optimization variable
    Delta_xk = zeros(size(optimmodel.f,2),1);
    %Langrange multipliers for the equality constraints
    Delta_lambdak = zeros(mequ,1);
    %Langrange mulitpliers for the inequality constraints
    Delta_muk = zeros(minequ+mquad,1);
    %Slack variables for the inequality constraints   
    Delta_sk = zeros(minequ+mquad,1);
    
    %Generate a result structure for visualization pusposes
    ip_result.xk = zeros(size(optimmodel.f,2),options.iter_max);
	ip_result.lambdak = zeros(mequ,options.iter_max);
	ip_result.muk = zeros(minequ+mquad,options.iter_max);
	ip_result.sk = zeros(minequ+mquad,options.iter_max);
	ip_result.Delta_xk = zeros(size(optimmodel.f,2),options.iter_max);
	ip_result.Delta_lambdak = zeros(mequ,options.iter_max);
	ip_result.Delta_muk = zeros(minequ+mquad,options.iter_max);
	ip_result.Delta_sk = zeros(minequ+mquad,options.iter_max);
    ip_result.fval = zeros(1,options.iter_max);
    ip_result.iterations = 0;
    ip_result.exitflag = 0;
    ip_result.exitstatus = '';
    %----------------------------------------------------------------------
    
    %----------------------------------------------------------------------
    %Initialize the single variables
    %----------------------------------------------------------------------
    %Duality measure
    rho = 1;
    %Centering parameter sigma = {0,1}
    sigma = options.centering; 
    %Step length reduction parameter t = {0,1}
    t  = 0.8;
    %inital step length for line search
    hk = 1;
    %Step length for line search
    hk_sk_max = 1;
    hk_muk_max  = 1;
    
    %Security factor (stricly positiveness of slack and lagrange multiplier)
    gamma = 0.995;
    
    fprintf('====================================================================\n');
    fprintf('INFO: Checking solver configurations...\n');
    fprintf('====================================================================\n');
    
    fprintf(' \n');
    fprintf('INFO: Initialization of duality measure %f...\n', rho);
    fprintf('INFO: Centering parameter %f...\n', sigma);
    fprintf('INFO: Step length reduction parameter %f...\n', t);
    fprintf('INFO: Initial step length for backtracking line search %f...\n', hk);
    fprintf('INFO: Maximum step length for backtracking line search %f...\n', hk_sk_max);
    fprintf('INFO: Security factor to guarantee strict posiveness %f...\n', gamma);
    fprintf(' \n');
   
    %======================================================================
    % ALGORITHM START
    %======================================================================
    
    %----------------------------------------------------------------------
    %Initialize the routine variables
    iter = 1;
    xk = options.x0;
    lambdak = ones(mequ,1);
    muk = ones(minequ+mquad,1);
    sk = ones(minequ+mquad,1);
    nuk = zeros(minequ+mquad,1);
    ip_result.fval(1,iter) = 0.5*xk'*optimmodel.H*xk+optimmodel.f*xk;
    
    %Intitialize the residuals
    %Dual Residual
    rS = -(optimmodel.H*xk+optimmodel.f' + optimmodel.Aequ'*lambdak+G'*muk);
    %Primal residual
    rE = -(optimmodel.Aequ*xk-optimmodel.Bequ);
    %----------------------------------------------------------------------
    
    %Don't stop with the iterative solving till the we have a certain value
    %of primal and dual feasibility and the surrogate duality gap is small
    %enough
    fprintf('====================================================================\n');
    fprintf('INFO: Optimization preprocessing finished...\n');
    fprintf('====================================================================\n');
    fprintf('\n');
    fprintf('====================================================================\n');
    fprintf('INFO: Start interior point solver...\n');
    fprintf('====================================================================\n');
    fprintf('objective cost | primal feasibility gap | dual feasibility gap | complementary slackness | iteration\n');
    
    %Try to solve this until certain criteria are fulfilled
     while ((norm(rE,2) >= options.eps_prim) || (norm(rS,2) >= options.eps_dual) || (-g'*muk >= options.eps_gap)) && (iter <= options.iter_max)
       
        %fprintf('Solver preprocessing finished...\n'); 
        fprintf('%e      |%e       | %e     |%e      | %i\n', ip_result.fval(1,max([1;iter-1])), norm(rE,2), norm(rS,2), (-g'*muk), iter);
  
        %First we build up the parameter Matrices of the left side
        
        %Reset to zero for the next iteration
        Hess = zeros(size(optimmodel.f,2),size(optimmodel.f,2));
        %Add the cost function value
        Hess = Hess + optimmodel.H;
        
        %Determine the derivations of the current constraints satisfaction
        for i=1:mquad+1
            if i<=mquad
                Hess = Hess + muk(i,1)*2*optimmodel.Aquad(:,:,i);
                G(i,:) = (2*(optimmodel.Aquad(:,:,i)*xk)+optimmodel.lquad(:,i))';
                g(i) = xk'*optimmodel.Aquad(:,:,i)*xk+optimmodel.lquad(:,i)'*xk-optimmodel.rquad(i);
            else
                G(mquad+1:minequ+mquad,:) = optimmodel.Ainequ;
                g(mquad+1:minequ+mquad) = optimmodel.Ainequ*xk-optimmodel.Binequ;
            end     
        end
        
        %Compute the duality measure for the inequality contraint with the
        %slack variables and the Langrange multipliers devided by the sum
        %of inequality constraints
        nuk = (sk'*muk)/(minequ+mquad);
                
        %==================================================================
        %Solution of the KKT throug Newton
        %==================================================================
        
        %Build up the matrice for slack variables and Lagrange multipliers
        %for the inequality constraints
        S = diag(sk);
        Mu = diag(muk);
             
        %------------------------------------------------------------------
        %Determine the residuals for the right hand side
        %Dual residual
        rS = -(optimmodel.H*xk+optimmodel.f' + optimmodel.Aequ'*lambdak+G'*muk);
        %Primal residual
        rE = -(optimmodel.Aequ*xk-optimmodel.Bequ);
        %Inequality residual
        rI = -(g+sk);
        %Slack variable residual
        rC = -(S*muk-nuk*sigma*ones(minequ+mquad,1));
        %------------------------------------------------------------------
        
        %Solve the reduced system
        A_IP = [Hess optimmodel.Aequ' G' zeros(size(G',1),size(Mu,2));...
                optimmodel.Aequ zeros(size(optimmodel.Aequ,1),size(optimmodel.Aequ',2)+size(G',2)+size(Mu,2));...
                G zeros(size(G,1),size(optimmodel.Aequ',2)+size(G',2)) eye(size(G,1));...
                zeros(size(S,1),size(optimmodel.H,2)+size(optimmodel.Aequ',2)) S Mu];
            
        B_IP = [rS;rE;rI;rC];
                
        %Solve the linear equation
        Delta_Var = A_IP\B_IP;  
        
        %Get the delta variables from the predictor step
        Delta_xk = Delta_Var(1:size(optimmodel.f,2));
        Delta_lambdak = Delta_Var(size(optimmodel.f,2)+1:size(optimmodel.f,2)+mequ);
        
        %Reconstruct the Lagrange multipliers for the inequality
        %constraints
        Delta_muk = Delta_Var(size(optimmodel.f,2)+mequ+1:size(optimmodel.f,2)+mequ+minequ+mquad);
        
        %#############
        %To Do:
        % Check if the Delta_sk value is correct?
        %##############
        
        %Reconstruct the Slack Variables for the inequality
        %constraints
        Delta_sk = Delta_Var(size(optimmodel.f,2)+mequ+minequ+mquad+1:end);
        %==================================================================
        %==================================================================
        
        %------------------------------------------------------------------
        %Line Search and determination of affine duality measure 
        %------------------------------------------------------------------
           
        %determine the starting step length, the maximum should be 1 or
        %smaller
        hk = 0.99*min([1;hk_sk_max;hk_muk_max]);
        
        % Backtracking line search, reduce the step length of the Newton
        % step until we are inside the the interoir again
        while min(sk + hk*Delta_sk) < 0 || min(muk + hk*Delta_muk) < 0
            hk = hk*t; 
        end 
		    % Save some information for debuging purposes
		    % With this information which serves as an output, the solution
		    % process can be visualized
		ip_result.xk(:,iter) = xk;
        ip_result.lambdak(:,iter) = lambdak;
        ip_result.muk(:,iter) = muk;
        ip_result.nuk(:,iter) = nuk;
        ip_result.sk(:,iter) = sk;
        ip_result.hk(:,iter) = hk;
        ip_result.Delta_xk(:,iter) = Delta_xk;
        ip_result.Delta_lambdak(:,iter) = Delta_lambdak;
        ip_result.Delta_muk(:,iter) = Delta_muk;
        ip_result.Delta_sk(:,iter) = Delta_sk;
		ip_result.fval(1,iter) = 0.5*xk'*optimmodel.H*xk+optimmodel.f*xk;
        
        %=======================================================================
        %Debug output in the command window
        %=======================================================================
        %fprintf('There are %d horses\n', A);
        %fprintf('There are %d horses\n', A);
        
        %Determine the value of the final step
        xk = xk + gamma*hk*Delta_xk;
        lambdak = lambdak + gamma*hk*Delta_lambdak;
        muk = muk + gamma*hk*Delta_muk;
        sk = sk + gamma*hk*Delta_sk;
        
		%Increase iteration index
        iter = iter +1;
        %==================================================================
        %==================================================================                 
     end

    %The reason for the algotihm to stop was convergence
    if (norm(rE,2) <= options.eps_prim) && (norm(rS,2) <= options.eps_dual) && (-g'*muk <= options.eps_gap)
        ip_result.exitflag = 1;
        ip_result.exitstatus = 'Problem solved, optimal solution reached';
    %The reason for the algorithm to stop was the maximum number of allowed
    %iterations
    elseif iter >= options.iter_max 
        ip_result.exitflag = 5;
        ip_result.exitstatus = 'Maximum iterations reached';
    %Something else
    else
        ip_result.exitflag = -2;
        ip_result.exitstatus = 'Problem infeasible from presolve';
    end
    %Output variables
    ip_result.iterations = iter;
end
