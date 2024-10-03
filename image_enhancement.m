clc;
clear;

% Read the image
input_image = imread('image4.jpg');

% Convert the image to double precision for calculations
DPI = im2double(input_image);

% Convert RGB to HSI
hsi_image = rgb2hsi(DPI);

% Extract HSI components
H = hsi_image(:,:,1); % Hue
S = hsi_image(:,:,2); % Saturation
I = hsi_image(:,:,3); % Intensity

% Compute mean values
mean_S = mean2(S);
mean_I = mean2(I);

% Gray-World Color Balancing in S and I channels
scaled_S = S * (0.4 / mean_S); % Adjust scaling factor as needed
scaled_I = I * (0.6 / mean_I); % Adjust scaling factor as needed

% Recombine HSI components
balanced_hsi_image(:,:,1) = H;
balanced_hsi_image(:,:,2) = scaled_S;
balanced_hsi_image(:,:,3) = scaled_I;

% Convert HSI back to RGB
balanced_image = hsi2rgb(balanced_hsi_image);

% Clip the values to the range [0, 1]
clipped_image = min(max(balanced_image, 0), 1);

% RGB Color Balancing to reduce yellowish color cast
mean_R = mean2(clipped_image(:,:,1));
mean_G = mean2(clipped_image(:,:,2));
mean_B = mean2(clipped_image(:,:,3));

% Scaling factors to balance the colors
scale_R = (mean_G + mean_B) / (2 * mean_R);
scale_B = (mean_R + mean_G) / (2 * mean_B);

balanced_rgb_image(:,:,1) = clipped_image(:,:,1) * scale_R;
balanced_rgb_image(:,:,2) = clipped_image(:,:,2);
balanced_rgb_image(:,:,3) = clipped_image(:,:,3) * scale_B;

% Clip the values to the range [0, 1]
balanced_rgb_image = min(max(balanced_rgb_image, 0), 1);
% Apply Gamma Correction
gamma_value = log(mean2(balanced_rgb_image)) / log(0.5); 

GCI(:,:,1) = balanced_rgb_image(:,:,1) .^ gamma_value;
GCI(:,:,2) = balanced_rgb_image(:,:,2) .^ gamma_value;
GCI(:,:,3) = balanced_rgb_image(:,:,3) .^ gamma_value;

% Apply Adaptive Histogram Equalization (CLAHE) for contrast enhancement
clipLimit = 0.01; % Adjust as needed
numTiles = [8 8]; % Adjust as needed
clahe_image(:,:,1) = adapthisteq(GCI(:,:,1), 'ClipLimit', clipLimit, 'NumTiles', numTiles);
clahe_image(:,:,2) = adapthisteq(GCI(:,:,2), 'ClipLimit', clipLimit, 'NumTiles', numTiles);
clahe_image(:,:,3) = adapthisteq(GCI(:,:,3), 'ClipLimit', clipLimit, 'NumTiles', numTiles);

% Convert the image back to uint8
final_image_uint8 = im2uint8(clahe_image);

% Display the resulted images
figure;
subplot(2, 2, 1);
imshow(input_image);
title('Original Image');

subplot(2, 2, 2);
imshow(balanced_image);
title('GWCB Image');
imwrite(balanced_image, 'balanced_image.jpg')

subplot(2, 2, 3);
imshow(balanced_rgb_image);
title('RGB Balanced Image');
imwrite(balanced_rgb_image,'RGB balanced.jpg')

subplot(2, 2, 4);
imshow(final_image_uint8);
title('CLAHE Enhanced Image');
imwrite(final_image_uint8, 'Final image.jpg')
% Function for converting RGB to HSI
function hsi = rgb2hsi(rgb)
    R = rgb(:,:,1);
    G = rgb(:,:,2);
    B = rgb(:,:,3);
    
    % Compute intensity
    I = (R + G + B) / 3;
    
    % Compute saturation
    min_val = min(min(R, G), B);
    S = 1 - (3 ./ (R + G + B + eps)) .* min_val;
    
    % Compute hue
    num = 0.5 * ((R - G) + (R - B)) ;
    den = sqrt((R - G).^2 + (R - B) .* (G - B) + eps);
    H = acos(num ./ den);
    H(B > G) = 2 * pi - H(B > G);
    H = H / (2 * pi);
    
    hsi = cat(3, H, S, I);
end

% Function for converting HSI to RGB
function rgb = hsi2rgb(hsi)
    H = hsi(:,:,1) * 2 * pi;
    S = hsi(:,:,2);
    I = hsi(:,:,3);
    
    % Convert hue, saturation, intensity to RGB
    R = zeros(size(H));
    G = zeros(size(H));
    B = zeros(size(H));
    
    % Compute RGB values
    for i = 1:numel(H)
        if H(i) >= 0 && H(i) < 2*pi/3
            B(i) = I(i) * (1 - S(i));
            R(i) = I(i) * (1 + (S(i) * cos(H(i)) / cos(pi/3 - H(i))));
            G(i) = 3 * I(i) - (R(i) + B(i));
        elseif H(i) >= 2*pi/3 && H(i) < 4*pi/3
            H(i) = H(i) - 2*pi/3;
            R(i) = I(i) * (1 - S(i));
            G(i) = I(i) * (1 + (S(i) * cos(H(i)) / cos(pi/3 - H(i))));
            B(i) = 3 * I(i) - (R(i) + G(i));
        else
            H(i) = H(i) - 4*pi/3;
            R(i) = I(i) * (1 + (S(i) * cos(H(i)) / cos(pi/3 - H(i))));
            G(i) = I(i) * (1 - S(i));
            B(i) = 3 * I(i) - (R(i) + G(i));
        end
    end
    
    rgb = cat(3, R, G, B);
end
