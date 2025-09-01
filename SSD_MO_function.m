function [ Z, H, dnorm, H_final] = SSD_MO_function(maxiteration, Aopt, Dopt, XX, layers, gnd, beta, eta, delta, graph_k)
numOfView = numel(XX);
nviews = numOfView;
nClass = length(unique(gnd));
nlayers = numel(layers);
% Z = cell(numOfView, nlayers);
% H = cell(numOfView, nlayers);
bUpdateH = 1;
bUpdateLastH = 1;
maxiter = maxiteration;
verbose = 1;
bUpdateZ = 1;
options = [];
options.k = graph_k;
options.WeightMode = 'HeatKernel';    
if verbose
    disp('Skipping initialization, using provided init matrices...');
end

[Z,H] = initialize_DeepNMF_MV(XX, layers);
%% Error Propagation
if verbose
    disp('Finetuning...');
end
H_err = cell(numOfView, nlayers);
derror = [];

labPer = 0.3;
nSamp = size(XX{1},2);
nl= round(labPer*nSamp); 
Hl = cell(numOfView, 1);
U = cell(numOfView, 1);
for v = 1:nviews
    U{v,1} = rand(nClass, layers(nlayers));
end

%Label matrix
nClass = length(unique(gnd));
B = (gnd==1:nClass);
Y = B(1:nl,:)';

maxDnorm = 10000;
for iter = 1:maxiter
    if (isnan(maxDnorm) || isinf(maxDnorm))
        break;
        isbreak = 1;
    end
    for v_ind = 1:numOfView
        X = XX{v_ind};       
        H_err{v_ind,numel(layers)} = H{v_ind,numel(layers)};
        for i_layer = numel(layers)-1:-1:1
            H_err{v_ind,i_layer} = Z{v_ind,i_layer+1} * H_err{v_ind,i_layer+1};
        end
        
        for i = 1:numel(layers)
            if bUpdateZ
                if i == 1
                    tempup = X*H_err{v_ind,1}';
                    tempun = Z{v_ind,1}*H_err{v_ind,1}*H_err{v_ind,1}';
                    Z{v_ind,i} = Z{v_ind,i}.*sqrt(tempup./max(tempun,1e-10));
                else
                    SZ = Z{v_ind,1};
                    SZT = Z{v_ind,i-1}';
                    if i > 2
                        for u = i-2:1
                            SZT = SZT*Z{v_ind,u}';
                        end
                        for u = 2:i-1
                            SZ = SZ*Z{v_ind,u};
                        end
                    end
                    tempup = SZT*X*H_err{v_ind,i}';
                    tempun = SZT*SZ*Z{v_ind,i}*H_err{v_ind,i}*H_err{v_ind,i}';
                    Z{v_ind,i} = Z{v_ind,i}.*sqrt(tempup./max(tempun,1e-10));
                end
            end
            
            if i == 1
                D = Z{v_ind,1}';
            else
                D = Z{v_ind,i}' * D;
            end
            
            if bUpdateH && (i < numel(layers) || (i == numel(layers) && bUpdateLastH))
                A = D * X;
                Ap = (abs(A)+A)./2;
                An = (abs(A)-A)./2;

                B = D * D';
                Bp = (abs(B)+B)./2;
                Bn = (abs(B)-B)./2;  
                % Calculate Lagrange
                PP = A*H{v_ind,i}' - B*H{v_ind,i}*H{v_ind,i}'+ eta*H{v_ind,i}*Aopt*H{v_ind,i}';
                PP1 = (abs(PP)+PP)./2;
                PP0 = (abs(PP)-PP)./2;
                tempup = Ap + Bn*H{v_ind,i} + eta*H{v_ind,i}*Aopt + eta*PP0*H{v_ind,i}*Dopt;
                tempun = An + Bp* H{v_ind,i} + eta*PP1*H{v_ind,i}*Dopt;
                
                if i == numel(layers)
                    sumHs = zeros(size(H{v_ind, numel(layers)}));
                    for s = 1:numOfView
                        if s~=v_ind
                            sumHs = sumHs + H{s, numel(layers)};
                        end
                    end
                    tempun = tempun + beta*sumHs;
                    %% Update Hl which happens only on the top layer
                    Hl{v_ind,1} = H{v_ind,i}(:,1:nl);
                    tempup_Hl = (U{v_ind,1}'*Y);
                    tempun_Hl = U{v_ind,1}'*U{v_ind,1}*Hl{v_ind,1};
                    
                    tempup(:,1:nl) = tempup(:,1:nl) + delta*tempup_Hl;
                    tempun(:,1:nl) = tempun(:,1:nl) + delta*tempun_Hl;
                end
                H{v_ind,i} = H{v_ind,i} .* sqrt(tempup ./max(tempun, 1e-10));
                
                %Re-assign Hl after updating
                Hl{v_ind,i} = H{v_ind,i}(:,1:nl);                
            end
        end
        %% Update U
        tempup = Y*Hl{v_ind,1}';
        tempun = U{v_ind,1}*Hl{v_ind,i}*Hl{v_ind,1}';
        U{v_ind,1} = U{v_ind,1} .* sqrt(tempup ./max(tempun, 1e-10));
        % end update U
        assert(i == numel(layers));
        
    end

    for v_ind = 1:numOfView,
        X = XX{v_ind};
        sumHv = H{1,nlayers};
        
        for v =2:numOfView
            sumHv = sumHv + H{v, nlayers};
        end
        for v = 1:numOfView
            sumnotHv{v,1} = sumHv - H{v,nlayers};
        end
        % get the error for each view
       
        %Q = Z(v_ind,:);
        dnorm(v_ind) = cost_function_graph_K(X, Z(v_ind,:), H(v_ind,:),Hl{v_ind,1},U{v_ind,1},Y, sumnotHv{v_ind,1}, Aopt, beta, eta, delta, nlayers);
                                            
    end 
    % finish update Z H and other variables in each view disp result
    
    maxDnorm = sum(dnorm);
    if verbose
        derror(iter) = maxDnorm;
    end
    if verbose && length(gnd) > 1
        if mod(iter, 1) == 0|| iter ==1
            H_final = (1/numOfView)*H{1,nlayers};
            for v = 2:numOfView
                H_final = H_final + (1/numOfView)*H{v,nlayers};
            end
            
        end
    end
end

end

% function error = cost_function_graph_K(X, Z, H, sumHs, Aopt, beta, eta, nlayers) %H here is a cell of different layers on each view
% error1 = norm(X - reconstruction(Z, H), 'fro');
% error2 = 0;
% for i = 1:nlayers
%     error2 = error2 + eta*trace(H{i}*Aopt*H{i}');
% end
% error3 = beta*trace(H{i}*sumHs');
% error = error1 - error2 + error3;
% end
function error = cost_function_graph_K(X, Z, H, Hl, U, Y, sumHs, Aopt, beta, eta, delta, nlayers) %H here is a cell of different layers on each view
    error1 = norm(X - reconstruction(Z, H), 'fro');
    error2 = 0;
    for i = 1:nlayers
        error2 = error2 + eta*trace(H{i}*Aopt*H{i}');
    end
    error3 = beta*trace(H{i}*sumHs');
    error4 = norm(Y - U*Hl, 'fro');
   
    error = error1 - error2 + error3 + delta*error4;
end

function [ out ] = reconstruction( Z, H )

out = H{numel(H)};

for k = numel(H) : -1 : 1;
    out =  Z{k} * out;
end
end
