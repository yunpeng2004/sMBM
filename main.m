clc;
clear;
close all;

%% 1. Read network topology
fid = fopen('adj_edges_all.txt');
adjacent_network = {};
j = 0;
while ~feof(fid)
    tline = fgetl(fid);
    j = j + 1;
    adjacent_network{j} = regexp(tline, '\t', 'split');
end
fclose(fid);
total_node_num = j;

%% 2. Read control-group expression data (58 samples)
fpi = fopen('pruning_normal_expression_step2.txt');
textscan(fpi, '%s', 1, 'delimiter','\n');  
fmt = '%s';
for i = 2:59
    fmt = [fmt ' %f'];
end
lines    = textscan(fpi, fmt, 1e6, 'delimiter','\t');
pipi     = lines{1};  
pprofile = [];
for i = 2:59
    pprofile = [pprofile, lines{i}];
end
fclose(fpi);
tempcontrol = pprofile;  

%% 3. Read tumor-group expression data (424 samples)
fpi = fopen('pruning_tumor_expression_step2.txt');
textscan(fpi, '%s', 1, 'delimiter','\n');    
textscan(fpi, '%s', 1, 'delimiter','\n');    
fmt = '%s';
for i = 2:426
    fmt = [fmt ' %f'];
end
lines    = textscan(fpi, fmt, 1e6, 'delimiter','\t');
mprofile = [];
for i = 2:426
    mprofile = [mprofile, lines{i}];
end
fclose(fpi);

%% 4. Define sample numbers per stage and merge I and IA
orig_patients_num = [3,106,124,39,59,62,10,21];   
patients_num = [sum(orig_patients_num(1:2)), orig_patients_num(3:end)];
stage_names  = {'I+IA','IB','IIA','IIB','IIIA','IIIB','IV'};
nStages      = numel(patients_num);
maxNS        = max(patients_num);

%% 5. Organize tumor data according to the new stages
nGenes   = size(pprofile,1);
tempcase = zeros(nGenes, nStages, maxNS);
idx = 1;
for l = 1:nStages
    nS = patients_num(l);
    tempcase(:,l,1:nS) = mprofile(:, idx:idx+nS-1);
    idx = idx + nS;
end

%% 6. Compute local network symmetric KL divergence
symKL   = zeros(total_node_num, nStages, maxNS);
pretime = clock;
for l = 1:nStages
    fprintf('--- Stage %s: start symKL computation ---\n', stage_names{l});
    for na = 1:total_node_num
        center = str2double(adjacent_network{na}{1});
        if numel(adjacent_network{na}) < 2
            continue;
        end
        nbrs  = cellfun(@str2double, adjacent_network{na}(2:end));
        nodes = [center, nbrs];
        k     = numel(nodes);

        % Control-group fitting
        data_ctrl = tempcontrol(nodes,:)';
        C0        = cov(data_ctrl);
        jitter0   = mean(diag(C0)) * 1e-3;
        S0        = C0 + jitter0*eye(k);
        invS0     = pinv(S0);
        logdet0   = sum(log(eig(S0)));
        mu0       = mean(data_ctrl,1);

        % Fit after adding case sample and compute symKL
        for s = 1:patients_num(l)
            vals_case = arrayfun(@(x) tempcase(x,l,s), nodes);
            data_case = [data_ctrl; vals_case];
            C1        = cov(data_case);
            jitter1   = mean(diag(C1)) * 1e-3;
            S1        = C1 + jitter1*eye(k);
            invS1     = pinv(S1);
            logdet1   = sum(log(eig(S1)));
            mu1       = mean(data_case,1);

            d01 = (mu0 - mu1)';
            KL0 = 0.5 * ( trace(invS1*S0) + d01'*(invS1*d01) ...
                       - k + (logdet1 - logdet0) );

            d10 = -d01;
            KL1 = 0.5 * ( trace(invS0*S1) + d10'*(invS0*d10) ...
                       - k + (logdet0 - logdet1) );

            symKL(na,l,s) = KL0 + KL1;
        end

        if mod(na,1000) == 0
            fprintf('Stage %s: processed %d/%d nodes\n', stage_names{l}, na, total_node_num);
        end
    end
    fprintf('Stage %s completed in %.2f s\n', stage_names{l}, etime(clock,pretime));
    pretime = clock;
end

%% 7. IQR filtering
symKL_clean = symKL;
for l = 1:nStages
    for s = 1:patients_num(l)
        v   = symKL(:,l,s);
        Q1  = quantile(v,0.25);
        Q3  = quantile(v,0.75);
        IQR = Q3 - Q1;
        lower = Q1 - 1.5*IQR;
        upper = Q3 + 1.5*IQR;
        out = (v < lower) | (v > upper);
        symKL_clean(out, l, s) = NaN;
    end
end

%% 8. Save cleaned results
save('symKL_clean.mat','symKL_clean');

%% 9. Visualization: mean symKL of top 5% nodes per stage
close all;
topN   = floor(total_node_num * 0.05);
averKL = zeros(nStages, maxNS);
for l = 1:nStages
    for s = 1:patients_num(l)
        v = symKL_clean(:,l,s);
        v = v(~isnan(v));
        if isempty(v), continue; end
        tmp = sort(v,'descend');
        cN  = min(topN, numel(tmp));
        averKL(l,s) = mean(tmp(1:cN));
    end
    plot(l*ones(1,patients_num(l)), averKL(l,1:patients_num(l)), '*'); hold on;
    aveAll(l) = mean(averKL(l,1:patients_num(l)));
end
figure;
plot(1:nStages, aveAll, '-o');
set(gca,'XTick',1:nStages,'XTickLabel',stage_names);
xlabel('Stage');
ylabel('Mean top 5% symKL');
title('');

%% 10. Text output
% Stage IIIB 
fw = fopen('expression_to_deltaKL_IIIB.txt','w');
for i = 1:total_node_num
    fprintf(fw, '%s', pipi{i});
    for s = 1:patients_num(6)
        val = symKL_clean(i,6,s);
        if isnan(val), val = 0; end
        fprintf(fw, '\t%f', val);
    end
    fprintf(fw, '\n');
end
fclose(fw);

% Output all stages
fw = fopen('expression_to_deltaKL_allStages.txt','w');
for i = 1:total_node_num
    fprintf(fw, '%s', pipi{i});
    for l = 1:nStages
        for s = 1:patients_num(l)
            val = symKL_clean(i,l,s);
            if isnan(val), val = 0; end
            fprintf(fw, '\t%f', val);
        end
    end
    fprintf(fw, '\n');
end
fclose(fw);
