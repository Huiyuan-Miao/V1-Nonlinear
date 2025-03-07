clear all
close all
figureID = 1;
% this is for saving out base Gabor model
% Two types of cells are simulated in this file. The simple cell and the complex cell. The Gabor output represents the simple cell, and combination of the quadrature pair Gabors represents the complex cell

%% load image and stimuli
% save two types of filters - one is filter output normalized by the max possible output of a filter, another is filter output not normalized. 
% the normalization is proposed because the filter output can be different because of the filter size. 
filterRsps = {'divMaxFiltRsp','noDivMaxFiltRsp'}; 
for fR = 1 : length(filterRsps)   
    type = {};
    count2 = 1;
    filterRsp = filterRsps{fR}
    F0ss = {[1,2,4,8]}
    for F = 1 : length(F0ss)
        F0s = F0ss{F};
        if F == 1
        load('preprocessedData_rescale40_bicubic_imgNormalized.mat');iw = 40; 
        % Preprocessed data. replicating the python preprocessing
        % steps.
        end        
        %%
        for ori = [8]%[4,8,12,16]
            saveDir = ['./GaborActivation4Phase_simpleComplex_',num2str(ori),'ori_',num2str(iw),'px_denseGabor/',filterRsps{fR}];
            if ~exist(saveDir)
                mkdir(saveDir)
            end
            %% create gabor model for step 1
            if ori == 4
                thetas = deg2rad(0:45:135);
            elseif ori == 8
                thetas = deg2rad(0:22.5:157.5);
            elseif ori == 12
                thetas = deg2rad(0:15:165);
            elseif ori == 16
                thetas = deg2rad(0:11.25:168.75);
            elseif ori == 20
                thetas = deg2rad(0:9:171);
            end
            freqBW = 1;
            GW = GaborWavelet_4phase_dense(iw,F0s,thetas,freqBW);
            numFilter = sum((F0s*2).^2) * length(thetas);
            GWReshape = zeros(numFilter,iw^2,4);% flatten the filters for matrix multiplication
            GWlab = zeros(numFilter,4); % save out the filter location, frequency, orientation 
            count = 0;
            ctrImg = iw/2 + 0.5;
            #whoIsZero = [];
            for i = 1 : length(F0s)
                for k = 1 : (F0s(i)*2)^2
                    ctrs = (1+(iw/F0s(i)/2)/2 :(iw/F0s(i)/2):iw)-0.5;
                    for m = 1 : length(thetas)
                        count = count + 1;
                        GWlab(count,1) = F0s(i);
                        GWlab(count,2) = thetas(m);
                        GWlab(count,3) = floor((k-1)/F0s(i)/2)+1;
                        GWlab(count,4) = mod((k-1),F0s(i)*2)+1;
                        f = reshape(GW{i,1}(:,:,k,m),[],iw^2);
                        GWReshape(count,:,1) = f/norm(f);
                        f = reshape(GW{i,2}(:,:,k,m),[],iw^2);
                        GWReshape(count,:,2) = f/norm(f);
                        f = reshape(GW{i,3}(:,:,k,m),[],iw^2);
                        GWReshape(count,:,3) = f/norm(f);
                        f = reshape(GW{i,4}(:,:,k,m),[],iw^2);
                        GWReshape(count,:,4) = f/norm(f);
                    end
                end
            end
            %% max filter responses calculation 
            BestImg = GWReshape;
            
            BestImg(BestImg>0) = 3;
            BestImg(BestImg<0) = -3;

            resp_Max_ = zeros(size(GWReshape,1),2);
            for i = 1 : 2
                resp1 = squeeze(sum(BestImg(:,:,i) .* GWReshape(:,:,1),2));
                resp2 = squeeze(sum(BestImg(:,:,i) .* GWReshape(:,:,2),2));
                resp_Max_(:,i) = (resp1.^2 + resp2.^2).^0.5;
            end
            resp1_Max = (squeeze(sum(BestImg(:,:,1) .* GWReshape(:,:,1),2)));
            resp2_Max = (squeeze(sum(BestImg(:,:,2) .* GWReshape(:,:,2),2)));
            resp3_Max = (squeeze(sum(BestImg(:,:,3) .* GWReshape(:,:,3),2)));
            resp4_Max = (squeeze(sum(BestImg(:,:,4) .* GWReshape(:,:,4),2)));
            resp_Complex_Max = (resp_Max_(:,1) + resp_Max_(:,2))/2;
            resp_Max = [resp_Complex_Max;resp1_Max;resp2_Max;resp3_Max;resp4_Max];
            resp_Max(resp_Max==0) = 1;
            
            %% Put img through filter
            % image preprocessing
            stimTrn_pro = zeros(size(stimTrn,1),iw^2);
            for i = 1 : size(stimTrn,1)
                stimTrn_pro(i,:) = reshape(squeeze(stimTrn(i,:,:)),[],iw^2);
            end
            % Put img through filter
            resp1 = GWReshape(:,:,1) * stimTrn_pro';
            resp2 = GWReshape(:,:,2) * stimTrn_pro';
            resp3 = GWReshape(:,:,3) * stimTrn_pro';
            resp4 = GWReshape(:,:,4) * stimTrn_pro';
            complex= (resp1.^2 + resp2.^2).^0.5;
            resp1_ = resp1;resp1_(resp1_<0) = 0;
            resp2_ = resp2;resp2_(resp2_<0) = 0;
            resp3_ = resp3;resp3_(resp3_<0) = 0;
            resp4_ = resp4;resp4_(resp4_<0) = 0;
            resps_con= [complex;resp1_;resp2_;resp3_;resp4_;];
            % image preprocessing
            stimVal_pro = zeros(size(stimVal,1),iw^2);
            for i = 1 : size(stimVal,1)
                stimVal_pro(i,:) = reshape(squeeze(stimVal(i,:,:)),[],iw^2);
            end
            % Put img through filter
            resp1 = GWReshape(:,:,1) * stimVal_pro';
            resp2 = GWReshape(:,:,2) * stimVal_pro';
            resp3 = GWReshape(:,:,3) * stimVal_pro';
            resp4 = GWReshape(:,:,4) * stimVal_pro';
            complex_Val = (resp1.^2 + resp2.^2).^0.5;
            resp1_ = resp1;resp1_(resp1_<0) = 0;
            resp2_ = resp2;resp2_(resp2_<0) = 0;
            resp3_ = resp3;resp3_(resp3_<0) = 0;
            resp4_ = resp4;resp4_(resp4_<0) = 0;
            resps_con_Val= [complex_Val;resp1_;resp2_;resp3_;resp4_];
            
            if fR == 1 % normalize filter output by the max possible filter responses
                resps_con = resps_con./resp_Max;
                resps_con_Val = resps_con_Val./resp_Max;
            end
            
            a = [resps_con,resps_con_Val];
            b = [stimTrnID;stimValID];
            [x,y] = sort(b);
            c = a(:,y);
            h5create([saveDir,'/layer0Processed.h5'],'/featuremap',[size(c,1),7250])
            h5write([saveDir,'/layer0Processed.h5'],'/featuremap',c)
        end
    end
end
