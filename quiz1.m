%a)
m = load('TrainingSamplesDCT_8.mat');
fg = m.TrainsampleDCT_FG;
bg = m.TrainsampleDCT_BG;
fg_rows = size(fg, 1);
bg_rows = size(bg, 1);
prior_cheetah = fg_rows / (fg_rows + bg_rows);
prior_grass = bg_rows / (fg_rows + bg_rows);
fprintf('P(Cheetah) = %.4f\n', prior_cheetah);
fprintf('P(Grass) = %.4f\n', prior_grass);

%b) 
zigzag_order = load('Zig-Zag Pattern.txt') + 1;
num_bins = 64;
cheetah_hg = zeros(1, num_bins);
grass_hg = zeros(1, num_bins);

for i = 1:fg_rows
    dct_vector = abs(TrainsampleDCT_FG(i, :));
    [~, sorted_indices] = sort(dct_vector, 'descend');
    second_largest_index = sorted_indices(2);
    zigzag_pos = find(zigzag_order == second_largest_index);
    cheetah_hg(zigzag_pos) = cheetah_hg(zigzag_pos) + 1;
end

for i = 1:bg_rows
    dct_vector = abs(TrainsampleDCT_BG(i, :));
    [~, sorted_indices] = sort(dct_vector, 'descend');
    second_largest_index = sorted_indices(2);
    zigzag_pos = find(zigzag_order == second_largest_index);
    grass_hg(zigzag_pos) = grass_hg(zigzag_pos) + 1;
end

cheetah_cond = cheetah_hg / sum(cheetah_hg);
grass_cond = grass_hg / sum(grass_hg);
figure;
bar(1:num_bins, cheetah_cond);
title('Index Histogram of P(x|Cheetah)')
xlabel('Index');
ylabel('P(x|Cheetah)');
figure;
bar(1:num_bins, grass_cond);
title('Index Histogram of P(x|Grass)');
xlabel('Index');
ylabel('P(x|Grass)');

%c)
[img_orig, ~] = imread('cheetah.bmp');
img_double = im2double(img_orig);
dct_coeffs = zeros(263*248, 64);
for i = 1:263  
    for j = 1:248 
        block = img_double(j:j+7, i:i+7);
        dct_block = dct2(block);
        dct2_block = dct2(dct_block);
        coeff_vector = zeros(1, 64);
        for row = 1:8
            for col = 1:8
                coeff_vector(zigzag_order(row, col)) = dct2_block(row, col);
            end
        end
        dct_coeffs((i-1)*248+j, :) = coeff_vector;  
    end
end

classmap = zeros(255, 270);
threshold = prior_grass / prior_cheetah;

for block_idx = 1:65224
    [~, sorted_idx] = sort(abs(dct_coeffs(block_idx, :)));

    if (cheetah_cond(sorted_idx(63)) / grass_cond(sorted_idx(63)) > threshold)
        classmap(rem(block_idx, 248)+1, floor(block_idx/248)+1) = 1;
    end
end

figure;  
imagesc(classmap); 
colormap gray(255); 
axis image; 
title('Cheetah Image Segmentation');

%d)
[mask, ~] = imread('cheetah_mask.bmp');
mask = mask / 255;
error = sum(sum(xor(mask, classmap))) / (255*277);
fprintf('P(Error) = %.4f\n', error);
