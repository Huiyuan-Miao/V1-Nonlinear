% Huiyuan Miao @ 2024
% this file is used for creating Gabor wavelet with 4 different phases for feature extraction. 
% iw - image width for your images to be processed. This is defined as your field of view (FOV) and the Gabors are defined relative to this. 
% F0s - the frequencys of your Gabors in a unit of x cycle/FOV
% thetas - the orientation of the Gabors
% freqBW - frequency bandwidth of the Gabors

% the output of this function is a group of Gabors with fixed location. If you don't want this restriction, you can use the output of GaborWaveletSimple at the bottom of the page. 

function GW = GaborWavelet_4phase_dense(iw,F0s,thetas,freqBW)
% originally the 2 c/fov have 2 * 2 gabors, now it would have 4 * 4, etc
GW = cell(length(F0s),4);
for i = 1 : length(F0s)
    GW1 = zeros(iw,iw,(F0s(i)*2)^2,length(thetas));
    GW2 = zeros(iw,iw,(F0s(i)*2)^2,length(thetas));
    GW3 = zeros(iw,iw,(F0s(i)*2)^2,length(thetas));
    GW4 = zeros(iw,iw,(F0s(i)*2)^2,length(thetas));
    gridWidth = iw/F0s(i)/2;
%     gridWidth = iw/F0s(i);
    ctr = [iw,iw+1];
    for j = 1 : length(thetas)
        [gabor1,gabor2,gabor3,gabor4] = GaborWaveletSimple(iw,F0s(i),thetas(j),freqBW);

        bg1 = gabor1;
        bg2 = gabor2;
        bg3 = gabor3;
        bg4 = gabor4;
        count = 0;
        # put Gabors at different locations of the FOV
        for k = 1 : F0s(i)*2
            l = (k - 1) * gridWidth;
            r = (F0s(i)*2 - k) * gridWidth;
            for m = 1 : F0s(i)*2
                u = (m - 1) * gridWidth;
                d = (F0s(i)*2 - m) * gridWidth;
                count = count + 1;
                GW1(:,:,count,j) = bg1((ctr(1)-(gridWidth/2-1)-l):(ctr(2)+(gridWidth/2-1)+r),...
                    (ctr(1)-(gridWidth/2-1)-u):(ctr(2)+(gridWidth/2-1)+d));
                GW2(:,:,count,j) = bg2((ctr(1)-(gridWidth/2-1)-l):(ctr(2)+(gridWidth/2-1)+r),...
                    (ctr(1)-(gridWidth/2-1)-u):(ctr(2)+(gridWidth/2-1)+d));
                GW3(:,:,count,j) = bg3((ctr(1)-(gridWidth/2-1)-l):(ctr(2)+(gridWidth/2-1)+r),...
                    (ctr(1)-(gridWidth/2-1)-u):(ctr(2)+(gridWidth/2-1)+d));
                GW4(:,:,count,j) = bg4((ctr(1)-(gridWidth/2-1)-l):(ctr(2)+(gridWidth/2-1)+r),...
                    (ctr(1)-(gridWidth/2-1)-u):(ctr(2)+(gridWidth/2-1)+d));
            end
        end
    end
    GW{i,1} = GW1;
    GW{i,2} = GW2;
    GW{i,3} = GW3;
    GW{i,4} = GW4;
end
end
%%
% The GaborWaveletSimple function will give a Gabors without spatial restrictions. 
function [gabor1,gabor2,gabor3,gabor4] = GaborWaveletSimple(iw,F0,theta,freqBW)
lambda = 1/F0;
sd = 1/pi * sqrt(log(2)/2)*(2^freqBW+1)/(2^freqBW-1) * lambda;
% [x y] = meshgrid(linspace(-0.5,0.5,iw),linspace(-0.5,0.5,iw));
[x y] = meshgrid(linspace(-1,1,iw*2),linspace(-1,1,iw*2));
y_ = x .* cos(theta) - y .* sin(theta);
x_ = x .* sin(theta) + y .* cos(theta);
gauss = exp(-(x_.^2+y_.^2)/(2*sd^2)); % assume circular 2D Gaussian
gauss(gauss<0.01*max(gauss(:))) = 0;
G1 = sin(2*pi*F0 * (x_));
G2 = sin(2*pi*F0 * (x_)+pi/2);
G3 = sin(2*pi*F0 * (x_)+pi);
G4 = sin(2*pi*F0 * (x_)+3*pi/2);
% G5 = cos(2*pi*F0 * (x_));
gabor1 = gauss.*G1;
gabor2 = gauss.*G2;
gabor3 = gauss.*G3;
gabor4 = gauss.*G4;


end
