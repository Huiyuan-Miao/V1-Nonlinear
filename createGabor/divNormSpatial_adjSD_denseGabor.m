% this file is aimed at applying non-linear operations (divisive normalization) to the output of Gabor filter bank. 
% It can achieve 2 different types of divisive normalizaton across the sapce (surround suppression)
% 1) Gabor filter only normalized by weighted sum of nearby Gabors with the same orientation preference - acrossOri = 0
% 2) Gabor filters normalized by weighted sum of nearby Gabors from all possible orientation equally - acrossOri = 1
% please refer to Miao & Tong 2024 for detailed parameter setting.
% GWlab - is created when Gabor filters were generated - marked the Gabor features like orientation, spatial frequency etc.
% F0s is the Gabor's SF, you can choose which set of Gabor you would like to normalize. this parameter and adjSDParam collaboratively decide the impact from the neighboring parametersei
% thetas is the Gabor's orientation, you can choos which set of Gabor you would like to normalize
% iw is the input image size
% r controls the strength of normalization, larger r smaller normalization
% denominator is the normalization pool

function [outputs] = divNormSpatial_adjSD_denseGabor(inputs,GWlab,F0s,thetas,iw,r,acrossOri,adjSDParam,denominator)
% sd has a unit of pixel
outputs = inputs;
if acrossOri == 0
    for i = 1 : length(F0s)
        lambda = 1/F0s(i);freqBW = 1;
        sd = 1/pi * sqrt(log(2)/2)*(2^freqBW+1)/(2^freqBW-1) * lambda;
        sd = sd * adjSDParam;
        for O = thetas
            [x y] = meshgrid(linspace(-1/2,1/2,F0s(i)*2+1),linspace(-1/2,1/2,F0s(i)*2+1));
            gauss = exp(-(x.^2+y.^2)/(2*sd^2)); gauss = gauss./sum(gauss(:));
            id = find(GWlab(:,1) == F0s(i) & GWlab(:,2) == O);
            temp = denominator(id,:);
            for j = 1 : size(temp,2)
                temp2 = reshape(conv2(reshape(temp(:,j),F0s(i)*2,F0s(i)*2),gauss,'same'),(F0s(i)*2)^2,1);
                temp2(temp2==0) = 1;
                outputs(id,j) = outputs(id,j)./(r+temp2);
            end
        end
    end
elseif acrossOri == 1
    for i = 1 : length(F0s)
        lambda = 1/F0s(i);freqBW = 1;
        sd = 1/pi * sqrt(log(2)/2)*(2^freqBW+1)/(2^freqBW-1) * lambda;
        sd = sd * adjSDParam;
        [x y] = meshgrid(linspace(-1/2,1/2,F0s(i)*2+1),linspace(-1/2,1/2,F0s(i)*2+1));
        gauss = exp(-(x.^2+y.^2)/(2*sd^2)); gauss = gauss./sum(gauss(:));
        id = find(GWlab(:,1) == F0s(i));
        temp = denominator(id,:);
        temp2 = zeros(size(temp,1)/length(thetas),size(temp,2));
        for k = 1 : size(temp,1)/length(thetas)
            temp2(k,:) = mean(temp((k-1)*length(thetas)+1 : k*length(thetas),:));
        end
        for j = 1 : size(temp,2)
            for O = thetas
                id_O = find(GWlab(:,1) == F0s(i) & GWlab(:,2) == O);
                temp3 = reshape(conv2(reshape(temp2(:,j),F0s(i)*2,F0s(i)*2),gauss,'same'),(F0s(i)*2)^2,1);
                temp3(temp3==0) = 1;
                outputs(id_O,j) = outputs(id_O,j)./(r+temp3);
            end
        end
        
    end
end
end
