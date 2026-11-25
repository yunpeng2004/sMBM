clc;
clear;
close all;

%% 1. Reproducible random seed
rng(0);

%% 2. Simulation parameters 
delta_t = 1;
L       = 2100;
N       = round(L/delta_t);
sigma_ctrl = 0.01;   
sigma_case = 0.05;   

%% 3. List of s parameters
s = [-0.5, -0.475, -0.45, -0.4, -0.375, -0.35, -0.325, -0.3, -0.275, ...
     -0.25, -0.225, -0.2, -0.175, -0.15, -0.125, -0.1, -0.08, -0.05, ...
     -0.02, -0.001, 0.02, 0.05, 0.08, 0.1, 0.15];
total_s = numel(s);

%% 4. Network topology 
D = [ -1  1  0  0  0  0  0  0;
      -1 -1  0  0  0  0  0  0;
       1  0 -1  0  0  0  0  0;
       1  0  0 -1  0  0  0  0;
       1  0  0  0 -1  0  0  0;
       0  0  0  0  1 -1  0  1;
       0  0  0  0  0  0 -1  1;
       0  0  0  0  0  0 -1 -1 ];
total_nodes = size(D,1);
invD = inv(D);

% 4.1 Precompute first-order neighbors
neighbors = cell(total_nodes,1);
for na = 1:total_nodes
    nbrs = unique([find(D(na,:)~=0), find(D(:,na)~=0)']);
    nbrs(nbrs == na) = [];  
    neighbors{na} = nbrs;
end

%% 5. Simulate control samples (20 samples)
num_ctrl = 20;
pre_TC   = zeros(total_nodes, num_ctrl);
ss_ctrl  = linspace(-0.5,-0.45,num_ctrl);

for ll = 1:num_ctrl
    q_raw = 0.96^(1/abs(ss_ctrl(ll)));
    q     = max(q_raw,0.035);
    E     = diag([-3/5*q,-4/5,-5/5,-6/5,-7/5,-8/5,-9/5,-10/5]);

    Jmat  = D * E * invD;

    X      = zeros(total_nodes, N);
    X(:,1) = rand(total_nodes,1)*4 + 1;
    eJ     = expm(Jmat * delta_t);

    for t = 1:N-1
        X(:,t+1) = eJ*X(:,t) + sigma_ctrl*randn(total_nodes,1)*sqrt(delta_t);
    end

    pre_TC(:,ll) = X(:,2000);
end

%% Step 6. Computation
TT           = 100;
kld_all      = zeros(TT, total_s-1);
nodeKL_all   = zeros(TT, total_nodes, total_s-1);
miDE_all     = zeros(TT, total_s-1);   % MIWE
sjcd_all     = zeros(TT, total_s-1);   % sJCD
sle_all      = zeros(TT, total_s-1);   % SLE
eps_small = 1e-12;

% SJCD numerical grid
GRID_N = 200;
for f = 1:TT
    for idx = 2:total_s

        % Generate case/perturbed sample TC
        q_raw = 0.96^(1/abs(s(idx)));
        q     = max(q_raw,0.0375);
        E     = diag([-3/5*q,-4/5,-5/5,-6/5,-7/5,-8/5,-9/5,-10/5]);
        Jmat  = D * E * invD;
        X      = zeros(total_nodes, N);
        X(:,1) = rand(total_nodes,1)*4 + 1;
        eJ     = expm(Jmat * delta_t);
        for t = 1:N-1
            X(:,t+1) = eJ*X(:,t) + sigma_case*randn(total_nodes,1)*sqrt(delta_t);
        end
        TC = X(:,2000);

        % Local network symmetric KL
        symKL_vec = zeros(total_nodes,1);
        for na = 1:total_nodes
            nbrs  = unique([find(D(na,:)~=0), find(D(:,na)~=0)']);
            nodes = [na, nbrs];
            k     = numel(nodes);

            dataP   = pre_TC(nodes,:)';
            C0      = cov(dataP);
            jitter0 = mean(diag(C0)) * 1e-3;
            S0      = C0 + jitter0 * eye(k);
            invS0   = pinv(S0);
            logdet0 = sum(log(eig(S0)+eps_small));
            mu0     = mean(dataP,1);

            dataQ   = [dataP; TC(nodes)'];
            C1      = cov(dataQ);
            jitter1 = mean(diag(C1)) * 1e-3;
            S1      = C1 + jitter1 * eye(k);
            invS1   = pinv(S1);
            logdet1 = sum(log(eig(S1)+eps_small));
            mu1     = mean(dataQ,1);

            d01 = (mu0 - mu1)';
            KL0 = 0.5*( trace(invS1*S0) + d01'*(invS1*d01) ...
                   - k + (logdet1 - logdet0) );

            d10 = -d01;
            KL1 = 0.5*( trace(invS0*S1) + d10'*(invS0*d10) ...
                   - k + (logdet0 - logdet1) );

            symKL_vec(na) = KL0 + KL1;
        end

        nodeKL_all(f,:,idx-1) = symKL_vec';
        kld_all(f,idx-1)      = mean(symKL_vec);

        % MI network + weighted differential entropy
        stage_data = abs([pre_TC, TC]);
        stage_data = log1p(stage_data);

        sd_vec = std(stage_data, 0, 2) + eps_small;
        DE_vec = 0.5 * log(2*pi*exp(1) * (sd_vec.^2));

        R = corrcoef(stage_data');
        R(isnan(R)) = 0;
        MI_mat = zeros(total_nodes);
        for aa = 1:total_nodes-1
            for bb = aa+1:total_nodes
                rho = R(aa,bb);
                rho = max(min(rho, 1-1e-10), -1+1e-10);
                MI_ij = -0.5 * log(1 - rho^2 + eps_small);
                if ~isfinite(MI_ij); MI_ij = 0; end
                MI_mat(aa,bb) = MI_ij;
                MI_mat(bb,aa) = MI_ij;
            end
        end

        node_HT = zeros(total_nodes,1);
        for aa = 1:total_nodes
            acc = 0;
            for bb = 1:total_nodes
                if bb == aa; continue; end
                acc = acc + MI_mat(aa,bb) * DE_vec(bb);
            end
            node_HT(aa) = -acc;
        end

        miDE_all(f, idx-1) = mean(node_HT);

        % SJCD
        js_vec = zeros(total_nodes,1);

        for aa = 1:total_nodes
            dataP = pre_TC(aa,:)';
            dataQ = [dataP; TC(aa)];

            mu0 = mean(dataP);
            sd0 = std(dataP) + eps_small;

            mu1 = mean(dataQ);
            sd1 = std(dataQ) + eps_small;

            x_min = min([mu0-4*sd0, mu1-4*sd1, min(dataQ)]);
            x_max = max([mu0+4*sd0, mu1+4*sd1, max(dataQ)]);
            if x_max <= x_min
                x_max = x_min + 1;
            end

            x = linspace(x_min, x_max, GRID_N);
            p = (1./(sd0*sqrt(2*pi))) .* exp(-0.5*((x-mu0)./sd0).^2);
            q = (1./(sd1*sqrt(2*pi))) .* exp(-0.5*((x-mu1)./sd1).^2);
            p = p + eps_small; q = q + eps_small;
            p = p ./ sum(p);  q = q ./ sum(q);
            m = 0.5*(p+q);

            KLpm = sum( p .* log(p./m) );
            KLqm = sum( q .* log(q./m) );

            js_vec(aa) = 0.5*(KLpm + KLqm);
        end
        sjcd_all(f, idx-1) = mean(js_vec);

        % SLE
        dH_vec = zeros(total_nodes,1);
        for aa = 1:total_nodes
            nbrs = neighbors{aa};
            M = numel(nbrs);
            if M <= 1
                dH_vec(aa) = 0;
                continue;
            end

            c_ref = pre_TC(aa,:)';
            c_mix = [pre_TC(aa,:), TC(aa)]';

            abs_r_ref = zeros(M,1);
            abs_r_mix = zeros(M,1);

            for mIdx = 1:M
                nb = nbrs(mIdx);

                n_ref = pre_TC(nb,:)';
                n_mix = [pre_TC(nb,:), TC(nb)]';

                r0 = pearson_corr(c_ref, n_ref, eps_small);
                r1 = pearson_corr(c_mix, n_mix, eps_small);

                abs_r_ref(mIdx) = abs(r0);
                abs_r_mix(mIdx) = abs(r1);
            end

            s_ref = sum(abs_r_ref);
            s_mix = sum(abs_r_mix);
            if s_ref <= eps_small || s_mix <= eps_small
                dH_vec(aa) = 0;
                continue;
            end

            p_ref = abs_r_ref ./ (s_ref + eps_small);
            p_mix = abs_r_mix ./ (s_mix + eps_small);

            H_ref = -(1/log(M)) * sum( p_ref .* log(p_ref + eps_small) );
            H_mix = -(1/log(M)) * sum( p_mix .* log(p_mix + eps_small) );

            SD_ref = std(pre_TC(aa,:)) + eps_small;
            SD_mix = std([pre_TC(aa,:), TC(aa)]) + eps_small;
            ASD = abs(SD_mix - SD_ref);
            dH_vec(aa) = ASD * abs(H_mix - H_ref);
        end
        sle_all(f, idx-1) = sum(dH_vec);
    end
end

kld       = mean(kld_all, 1);
avgNodeKL = squeeze(mean(nodeKL_all, 1));
miDE      = abs(mean(miDE_all, 1));
sjcd      = mean(sjcd_all, 1);
sle       = mean(sle_all, 1);

%% Visualization
figure('Color','white'); hold on;
x_shade = [s(2), s(end)];
y_lim   = [0, max(kld_all(:))*1.1];
fill([x_shade, fliplr(x_shade)], [y_lim(1:2), y_lim(2:-1:1)], ...
     [1 0.8 0.5], 'EdgeColor','none', 'FaceAlpha',0.3);

% Boxplot
yyaxis left;
boxplot(kld_all, 'positions', s(2:end), ...
        'Widths', 0.015, 'Colors', 'k', 'Symbol', 'r+');
ylabel('KL Distribution');
ylim(y_lim);

% Mean curves
yyaxis right;
h1 = plot(s(2:end), kld,  'b-o', 'LineWidth', 2, 'MarkerFaceColor', 'b');
hold on;
h2 = plot(s(2:end), miDE, 'g-s', 'LineWidth', 2, 'MarkerFaceColor', 'g');
h3 = plot(s(2:end), sjcd, 'm-^', 'LineWidth', 2, 'MarkerFaceColor', 'm');
h4 = plot(s(2:end), sle,  'r-d', 'LineWidth', 2, 'MarkerFaceColor', 'r');
ylabel('Mean Scores (KL / MI-DE / SJCD / SLE)');
xticks(s(2:end));
xtickangle(45);
xlabel('Parameter s');
grid on; box on;
title('');
legend([h1, h2, h3, h4], ...
       {'Mean Symmetric KL', 'Mean MI-Weighted DE', ...
        'Mean SJCD (JSD-based)', 'Mean SLE (Local Entropy)'}, ...
       'Location', 'best');

%% 3D visualization
figure('Color','white');
h = bar3(avgNodeKL);
for kk = 1:numel(h)
    zdata = get(h(kk), 'ZData');
    set(h(kk), 'CData', zdata, 'FaceColor', 'interp');
end
ncol = 256;
r = [linspace(0,1,ncol/2), ones(1,ncol/2)];
g = [ones(1,ncol/2), linspace(1,0,ncol/2)];
b = zeros(1,ncol);
colormap([r',g',b']);
colorbar;
xlabel('parameter s');
ylabel('Node');
zlabel('mean Symmetric KL');
view(-30,30);

%% Pearson
function r = pearson_corr(x, y, eps_small)
    x = x(:); y = y(:);
    x = x - mean(x);
    y = y - mean(y);
    denom = sqrt(sum(x.^2) * sum(y.^2));
    if denom <= eps_small
        r = 0;
    else
        r = sum(x .* y) / denom;
    end
end
