function MainFunc_SSD_MO(datasetname, mode, normOnFea, maxiter)
% RUN SSD-MO with fixed params; prints K-means & spectral metrics.
%
% Usage:
%   run_ssd_mo_fixed('BRCA_MV.mat', 'cosine', 2, 200)
%
% Inputs:
%   datasetname : MAT file containing variables {fea, gnd, nviews}
%                 - fea: 1xV cell, each fea{v} is (features x samples) or (samples x features). 
%                 - This code expects (features x samples) and transposes to (samples x features).
%   mode        : graph WeightMode for constructA (e.g., 'cosine', 'binary', 'heat')
%   normOnFea   : 0 = no normalization, 2 = l2-row normalization via NormalizeFea
%   maxiter     : max iterations for SSD_MO_function
%
% Prints:
%   ACC, F, Precision, Recall, NMI, ARI for K-means and Spectral.
    % ---------------- Fixed hyperparameters ----------------
    eta       = 1;       % graph regularization
    beta      = 0.01;    % complementarity/diversity
    delta     = 1;       % semi-supervised term
    graph_k   = 50;      % k-NN
    layertype = 10050;   % 100->[100 C], 50->[50 C], 10050->[150 50]
    % ---------------- Load data ----------------
    S = load(datasetname);  % expects fea, gnd, nviews
    assert(isfield(S,'fea') && isfield(S,'gnd') && isfield(S,'nviews'), ...
        'dataset must contain {fea, gnd, nviews}');
    fea    = S.fea;
    gnd    = S.gnd(:);
    nviews = S.nviews;
    nClass = numel(unique(gnd));
    % Ensure each view is (samples x features)
    X = cell(1, nviews);
    for v = 1:nviews
        X{v} = fea{v,1}';
    end
    % Optional normalization
    if normOnFea == 2
        for v = 1:nviews
            X{v} = NormalizeFea(X{v}, 0);  % l2-normalize rows
        end
    end
    fea = X;
    % Define layer sizes
    if     layertype == 100
        layers = [100, nClass];
    elseif layertype == 50
        layers = [50,  nClass];
    elseif layertype == 10050
        layers = [150, 50];     % your original mapping
    else
        error('Unsupported layertype=%d', layertype);
    end
    % ---------------- Build graph/Laplacian inputs ----------------
    options = struct('k', graph_k, 'WeightMode', mode);
    A_graph = cell(1, nviews);
    for v = 1:nviews
        A_graph{v} = constructA(fea{v}', options);
    end
    Aopt = OptimalManifold(A_graph', nviews);
    Dopt = constructD(Aopt);
    % ---------------- Run SSD-MO ----------------
    rng(5489, 'twister');
    tic;
    [Z, H, dnorm, H_final] = SSD_MO_function(maxiter, Aopt, Dopt, fea, layers, gnd, beta, eta, delta, graph_k); %#ok<ASGLU>
    t_elapsed = toc;
    % ---------------- Evaluate: K-means on H_final ----------------
    km_ok = ~(any(isnan(H_final(:))) || any(isinf(H_final(:))));
    if km_ok
        % The updated function performance_kmeans also returns [value, std]
        [CA_km_full, F_km_full, P_km_full, R_km_full, NMI_km_full, ARI_km_full] = performance_kmeans(H_final', nClass, gnd);
        CA_km = CA_km_full(1);
        F_km = F_km_full(1);
        P_km = P_km_full(1);
        R_km = R_km_full(1);
        NMI_km = NMI_km_full(1);
        ARI_km = ARI_km_full(1);
    else
        warning('H_final has NaN/Inf; skipping K-means evaluation.');
        [CA_km, F_km, P_km, R_km, NMI_km, ARI_km] = deal(NaN);
    end
    % ---------------- Evaluate: Spectral on H_final ----------------
    if km_ok
        % The updated function evalResults_multiview_K returns [value, std]
        [CA_sp_full, F_sp_full, P_sp_full, R_sp_full, NMI_sp_full, ARI_sp_full] = evalResults_multiview_K(H_final, gnd);
        CA_sp  = CA_sp_full(1);
        F_sp   = F_sp_full(1);
        P_sp   = P_sp_full(1);
        R_sp   = R_sp_full(1);
        NMI_sp = NMI_sp_full(1);
        ARI_sp = ARI_sp_full(1);
    else
        [CA_sp, F_sp, P_sp, R_sp, NMI_sp, ARI_sp] = deal(NaN);
    end
    % ---------------- Print summary ----------------
    fprintf('\nSSD-MO (fixed params)\n');
    fprintf('  eta=%.4g, beta=%.4g, delta=%.4g, k=%d, layers=[%s], iter=%d\n', ...
        eta, beta, delta, graph_k, num2str(layers), maxiter);
    fprintf('  dataset=%s | WeightMode=%s | norm=%d | time=%.3fs\n\n', ...
        datasetname, mode, normOnFea, t_elapsed);
    print_block('K-means on H_{final}', CA_km, F_km, P_km, R_km, NMI_km, ARI_km);
    print_block('Spectral on H_{final}', CA_sp, F_sp, P_sp, R_sp, NMI_sp, ARI_sp);
end
% ---------- pretty printer ----------
function print_block(title, ACC, F, P, R, NMI, ARI)
    fprintf('%s\n', title);
    fprintf('  ACC: %.2f | F: %.2f | P: %.2f | R: %.2f | NMI: %.2f | ARI: %.2f\n\n', ...
        ACC, F, P, R, NMI, ARI);
end