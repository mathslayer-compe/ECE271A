%a)
m = load('TrainingSamplesDCT_8_new.mat');
fg = m.TrainsampleDCT_FG;
bg = m.TrainsampleDCT_BG;
fg_rows = size(fg, 1);
bg_rows = size(bg, 1);
prior_cheetah = fg_rows / (fg_rows + bg_rows);
prior_grass = bg_rows / (fg_rows + bg_rows);
fprintf('P(Cheetah) = %.4f\n', prior_cheetah);
fprintf('P(Grass) = %.4f\n', prior_grass);
disp("The priors are the same as before.")

%b)
mean_fg = sum(fg) / fg_rows;
mean_bg = sum(bg) / bg_rows;
std_fg = std(fg);
std_bg = std(bg);

x_fg = zeros(64, 61);
y_fg = zeros(64, 61);
x_bg = zeros(64, 61);
y_bg = zeros(64, 61);

for feature_idx = 1:64
    x_fg(feature_idx, :) = linspace(mean_fg(feature_idx) - 5 * std_fg(feature_idx), mean_fg(feature_idx) + 5 * std_fg(feature_idx), 61);
    y_fg(feature_idx, :) = normpdf(x_fg(feature_idx, :), mean_fg(feature_idx), std_fg(feature_idx));
    x_bg(feature_idx, :) = linspace(mean_bg(feature_idx) - 5 * std_bg(feature_idx), mean_bg(feature_idx) + 5 * std_bg(feature_idx), 61);
    y_bg(feature_idx, :) = normpdf(x_bg(feature_idx, :), mean_bg(feature_idx), std_bg(feature_idx));
end

for fig_idx = 1:8
    figure;
    for subplot_idx = 1:8
        feature_num = (fig_idx - 1) * 8 + subplot_idx;
        subplot(2, 4, subplot_idx);
        plot(x_fg(feature_num, :), y_fg(feature_num, :), '-b', x_bg(feature_num, :), y_bg(feature_num, :));
        title(['Marginal Density ', num2str(feature_num)]);
    end
    legend('P(x|cheetah)', 'P(x|grass)')
end

best_features = [1, 15, 22, 33, 38, 47, 49, 50];
worst_features = [2, 3, 5, 58, 59, 62, 63, 64];

figure;
for idx = 1:8
    subplot(2, 4, idx);
    plot(x_fg(best_features(idx), :), y_fg(best_features(idx), :), '-b', x_bg(best_features(idx), :), y_bg(best_features(idx), :));
    title(['Marginal Density ', num2str(best_features(idx))]);
end
legend('P(x|cheetah)', 'P(x|grass)');

figure;
for idx = 1:8
    subplot(2, 4, idx);
    plot(x_fg(worst_features(idx), :), y_fg(worst_features(idx), :), '-b', x_bg(worst_features(idx), :), y_bg(worst_features(idx), :));
    title(['Marginal Density ', num2str(worst_features(idx))]);
end
legend('P(x|cheetah)', 'P(x|grass)');

%c)
zigzag_indices = load('Zig-Zag Pattern.txt') + 1;
cheetah_image = im2double(imread('cheetah.bmp'));
[image_rows, image_cols] = size(cheetah_image);
small_constant = 1e-5;

bg_mean_matrix = repmat(mean_bg, bg_rows, 1);
fg_mean_matrix = repmat(mean_fg, fg_rows, 1);
covariance_bg = (bg - bg_mean_matrix)' * (bg - bg_mean_matrix) / bg_rows + small_constant * eye(64);
covariance_fg = (fg - fg_mean_matrix)' * (fg - fg_mean_matrix) / fg_rows + small_constant * eye(64);

result_64D = zeros(image_rows - 7, image_cols - 7);
result_8D = zeros(image_rows - 7, image_cols - 7);

for row = 1:(image_rows - 7)
    for col = 1:(image_cols - 7)
        dct_block = dct2(cheetah_image(row:row+7, col:col+7));

        feature_vector_64 = zeros(1, 64);
        for idx = 1:64
            [x_pos, y_pos] = find(zigzag_indices == idx);
            feature_vector_64(idx) = dct_block(x_pos, y_pos);
        end

        prob_bg = mvnpdf(feature_vector_64, mean_bg, covariance_bg) * prior_grass;
        prob_fg = mvnpdf(feature_vector_64, mean_fg, covariance_fg) * prior_cheetah;
        result_64D(row, col) = prob_fg > prob_bg;
    end
end

% Display image for 64-dimensional features
figure;
imagesc(result_64D);
title('64-Dimensional Gaussian Classification');
colormap gray(255);

for row = 1:(image_rows - 7)
    for col = 1:(image_cols - 7)
        dct_block = dct2(cheetah_image(row:row+7, col:col+7));

        feature_vector_8 = zeros(1, 8);
        for idx = 1:8
            [x_pos, y_pos] = find(zigzag_indices == best_features(idx));
            feature_vector_8(idx) = dct_block(x_pos, y_pos);
        end

        prob_bg = mvnpdf(feature_vector_8, mean_bg(best_features), covariance_bg(best_features, best_features)) * prior_grass;
        prob_fg = mvnpdf(feature_vector_8, mean_fg(best_features), covariance_fg(best_features, best_features)) * prior_cheetah;
        result_8D(row, col) = prob_fg > prob_bg;
    end
end

% Display image for 8-dimensional features
figure;
imagesc(result_8D);
title('8-Dimensional Gaussian Classification');
colormap gray(255);

true_mask = im2double(imread('cheetah_mask.bmp'));
error_64D = sum(sum(abs(true_mask(1:image_rows-7, 1:image_cols-7) - result_64D)));
error_8D = sum(sum(abs(true_mask(1:image_rows-7, 1:image_cols-7) - result_8D)));
error_rate_64D = error_64D / ((image_rows - 7) * (image_cols - 7));
error_rate_8D = error_8D / ((image_rows - 7) * (image_cols - 7));

fprintf('Error Rate (64D) = %.4f\n', error_rate_64D);
fprintf('Error Rate (8D) = %.4f\n', error_rate_8D);
