clear all;
load('TrainingSamplesDCT_8_new.mat');
c = 8;
dim = 64;

%%
scale = 0.0001;
for i = 1:5
    p_FG_tmp = generate_rd_parameter(c,dim,scale);
    p_FG{i} = EM_algo(TrainsampleDCT_FG,p_FG_tmp);
    
    p_BG_tmp = generate_rd_parameter(c,dim,scale);
    p_BG{i} = EM_algo(TrainsampleDCT_BG,p_BG_tmp); 
end

%% load eval data
gt = imread('cheetah_mask.bmp');
img = imread('cheetah.bmp');
img_p = im2double(padarray(img,[4 4],'symmetric','both'));
test_data = read_image(img,img_p);

%% inference
dim_eval = [1,2,4,8,16,24,32,40,48,56,64];
res = zeros([5,5,size(dim_eval,2),size(img)]);
p_fg = size(TrainsampleDCT_FG,1)/(size(TrainsampleDCT_FG,1)+size(TrainsampleDCT_BG,1));
p_bg = size(TrainsampleDCT_BG,1)/(size(TrainsampleDCT_FG,1)+size(TrainsampleDCT_BG,1));

for k = 1:size(dim_eval,2)
    disp(k);
    for i = 1:5
        for j = 1:5
            likelihood_bg = EM_eval(test_data, p_BG{i}, dim_eval(k));
            likelihood_fg = EM_eval(test_data, p_FG{j}, dim_eval(k));

        p_fg_x = likelihood_fg * p_fg;
        p_bg_x = likelihood_bg * p_bg;

        res_tmp = zeros(size(test_data,1),1);
        res_tmp(p_fg_x>p_bg_x) = 1;
        res(i,j,k,:,:) = reshape(res_tmp, size(img));
        end
    end
end
%% error
rate = zeros(5,5,size(dim_eval,2));
for i = 1:5
    figure
    for j = 1:5
        for k = 1:size(dim_eval,2)
            diff = abs(squeeze(res(i,j,k,:,:))-im2double(gt));
            fg_num = sum(sum(im2double(gt)));
            bg_num = (size(img,1)*size(img,2)) - fg_num;
            error_fg = sum(sum(diff.*(im2double(gt))));
            error = sum(sum(diff));
            error_bg = (error-error_fg);
            rate(i,j,k) = (2*error)/(size(img,1)*size(img,2));
            rate_fg = error_fg/fg_num;
            rate_bg = error_bg/bg_num;
        end
        plot(dim_eval, squeeze(rate(i,j,:)), 'o-', 'linewidth', 1, 'markersize', 5); hold on;
    end
    legend('BG1', 'BG2', 'BG3', 'BG4', 'BG5');
end

function [p] = generate_rd_parameter(c,dim,scale)
% Generate random parameters {(weight_i,mu_i,sigma_i)}, i is number of mixture
%model.

% weight should be summed to 1.
p_weight = rand(c, 1);
p_weight = p_weight/sum(p_weight);

% scaled mu will help divide samples equally.
% this should be a small number;
p_mu = scale*randn(c, dim);

% variance matrix also random and diagnoal.
p_var = zeros(c, dim, dim);
for i = 1:c
    p_var(i, :, :) = rand(dim).*eye(dim);
end
p.weight = p_weight;
p.var = p_var;
p.mu = p_mu;
end

function [p, likelihood_eval] = EM_algo(data,p)
% EM algorithm using
%   Detailed explanation goes here
p_mu = p.mu;
p_weight = p.weight;
p_var = p.var;

c = size(p_mu, 1);
dim = size(p_mu, 2);

likelihood_sum_old = 0;
likelihood_sum = 100;
disp(['start ']);

iter = 0;

while (likelihood_sum-likelihood_sum_old) > 1
    iter = iter + 1;
%% Likelihood
    l_i = zeros(size(data, 1),1);
    h_ij = zeros(size(data, 1), c);
    
    for j = 1:c
        [likelihood] = gaussian_likelihood(data, p_mu(j,:), p_var(j,:,:));
        h_ij(:,j) = likelihood.* p_weight(j);
        l_i = l_i + likelihood.* p_weight(j);
    end
    
    if c==1
        h_ij = ones(size(data,1),1);
    else
        h_ij = (h_ij'./(sum(h_ij')))';
    end
    
    likelihood_sum_old = likelihood_sum;
    likelihood_sum = sum(log(l_i));
    disp(likelihood_sum);
    %% M
    p_mu_next = p_mu;
    p_weight_next = p_weight;
    p_var_next = p_var;

    for j = 1:c
        p_mu_next(j,:) = sum(data.*h_ij(:,j))/sum(h_ij(:,j));
        p_weight_next(j) = sum(h_ij(:,j))/size(data,1);
        tmp_var = zeros(dim,dim);
        for i = 1:size(data,1)
            tmp_var = tmp_var + h_ij(i,j)*eye(dim).*((data(i,:)-p_mu_next(j,:))'*(data(i,:)-p_mu_next(j,:)));
        end
        p_var_next(j,:,:) = tmp_var/sum(h_ij(:,j))+1e-5*eye(size(tmp_var));
    end

    %% update
    p_mu = p_mu_next;
    p_weight = p_weight_next;
    p_var = p_var_next;
end
p.mu = p_mu;
p.weight = p_weight;
p.var = p_var;
end


function [likelihood] = gaussian_likelihood(data, mu, var)
% calculate likelihood of gaussian model given mu and variance matrix
%   data: n by dim
%   mu  : 1 by dim
%   var : dim by dim
    likelihood = zeros(size(data,1),1);
    tmp_var = squeeze(var);
    tmp_var_inv = inv(tmp_var);
    tmp_var_det = (det(tmp_var))^(0.5);
    for i = 1:size(data, 1)
        tmp_x = data(i,:)-mu;
        likelihood(i) = 1/tmp_var_det * exp(-0.5*tmp_x*tmp_var_inv*tmp_x');
    end
end


function data = read_image(img,img_p)
% read test image
index = textread('Zig-Zag Pattern.txt', '%d');
index = reshape(index+1, [8, 8])';

data = zeros([size(img),64]);
for i = 1:size(img_p,1)-8
    for j = 1:size(img_p,2)-8
        crop = img_p(i:i+7,j:j+7);
        crop_dct = dct2(crop);
        for ii = 1:8
            for jj = 1:8
                data(i,j,index(ii,jj)) = crop_dct(ii, jj);
            end
        end
    end
end
data = reshape(data, [size(data,1)*size(data,2), size(data,3)]);
end


function [likelihood] = EM_eval(data,p,dim)
%UNTITLED14 Summary of this function goes here
%   Detailed explanation goes here
p_mu = p.mu;
p_weight = p.weight;
p_var = p.var;

c = size(p_mu, 1);
likelihood = zeros(size(data, 1),1);

data = data(:,1:dim);
p_mu = p_mu(:,1:dim);
p_var = p_var(:,1:dim,1:dim);

for j = 1:c
    likelihood_ = gaussian_likelihood(data, p_mu(j,:), p_var(j,:,:));
    likelihood = likelihood + likelihood_.* p_weight(j);
end
end