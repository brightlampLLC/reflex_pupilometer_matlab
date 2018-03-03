function reflexPupillometerParaProcessBeta(vidName)
% FUNCTION reflexPupillometerAlph is a wrapper-code designed to measure the
% rate of pupil dilation in the presence of visible stimulation. This code
% is broken into the following blocks:
% Set up for parallel processing

numCores = feature('numCores');
poolobj = gcp('nocreate'); % If no pool, do not create new one.
if isempty(poolobj)
    if (numCores - 1) < 1
        poolobj = parpool('local',numCores);
    else
        poolobj = parpool('local',numCores-1);
    end
end

[pathName,fileBase,fileExt]  = fileparts(vidName);
vidName     = fullfile(pathName,[fileBase,fileExt]);

tic

try
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%% VIDEO READER AND IMAGE LOADER BLOCK START %%%%%%%%%%%%%%%
    fStart      = 15;   % Set Start frame number
    tEnd        = 5.5;  % Set approximate end time
    vid         = VideoReader(vidName);	% Load video from reader
    vid.CurrentTime = (fStart)/vid.FrameRate;   % Set Current Time to start frame
    frameTimeSeries = linspace(1,vid.FrameRate*vid.Duration,...
        vid.FrameRate*vid.Duration)/vid.FrameRate;  % Generate frame time series
    capTime     = find(frameTimeSeries <= tEnd);    % Use end time to find cut-off
    
    if vid.Height > vid.Width
        if vid.Height > 1920 && vid.Width > 1080
            videoDims = [1920, 1080, 3, (capTime(end)-(fStart-1))];
        else
            videoDims = [vid.Height, vid.Width, 3, (capTime(end)-(fStart-1))];
        end
        video       = uint8(zeros(videoDims)); % Preallocate memory for video images
        counter = 1;
        while vid.CurrentTime <= tEnd
            video(:,:,:,counter) = imresize(readFrame(vid),...
                [videoDims(1) videoDims(2)],'nearest');    % Read in video using readFrame
            counter = counter + 1;
        end
    elseif vid.Height < vid.Width
        if vid.Width > 1080 && vid.Height > 1920
            videoDims = [1920, 1080, 3, (capTime(end)-(fStart-1))];
        else
            videoDims = [vid.Height, vid.Width, 3, (capTime(end)-(fStart-1))];
        end
        video       = uint8(zeros(videoDims)); % Preallocate memory for video images
        counter = 1;
        while vid.CurrentTime <= tEnd
            video(:,:,:,counter) = imresize(permute(readFrame(vid),[2 1 3]),...
                [videoDims(1) videoDims(2)],'nearest');    % Read in video using readFrame
            counter = counter + 1;
        end
    end
    
    video(:,:,:,counter:end) = [];  % Remove any extra empty frames
    frskip      = 1;    % Frame skip index values
    frsrs       = linspace(1,size(video,4),size(video,4));
    frsrs       = frsrs(frskip:frskip:end);
    video       = video(:,:,:,frskip:frskip:end);
    videoDims   = size(video);
    frameMedian         = double(median(reshape(video,...
        videoDims(1)*videoDims(2)*videoDims(3),videoDims(4)),1));
    tstamp              = logical((frameMedian/255 > 0.70));
    frsrs(tstamp)       = [];
    video(:,:,:,tstamp) = [];
    videoDims           = size(video);
    clear vid capTime
    %%%%%%%%%%%%%%%%% VIDEO READER AND IMAGE LOADER BLOCK END %%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%% IMAGE REGISTRATION BLOCK START  %%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%% Registration settings block %%%%%%%%%%%%%%%%%%%%%%%
    NoOfWedges      = 256;                                  % FMC operator number of wedges (OpenCV does 360)
    MaxRad          = min(videoDims(1), videoDims(2))/2-1;  % FMC operator maximum radius
    SpatialWindow   = hanning(videoDims(1))*hanning(videoDims(2))';     % Spatial Apod Window
    FMCWindow       = hanning(NoOfWedges)*hanning(videoDims(2))';       % FMC Apod Win
    [XCart, YCart]  = meshgrid(-videoDims(2)/2:1:(videoDims(2)/2-1),...
        -videoDims(1)/2:1:(videoDims(1)/2-1));                              % Cartesian grid
    [XLP, YLP]      = LogPolarCoordinates([videoDims(1),videoDims(2)],...
        NoOfWedges, videoDims(2),2 , MaxRad, 2*pi);                         % Log-Polar grid
    dispX = zeros([videoDims(4),1]); % Preallocate horizontal (x-axis) displacement
    dispY = zeros([videoDims(4),1]); % Preallocate vertical (y-axis) displacement
    dispS = zeros([videoDims(4),1]); % Preallocate scaling displacement
    %%%%%%%%%%%%%%%%%%%%%%%%%%% Registration Process %%%%%%%%%%%%%%%%%%%%%%%%%%
    ref     = double(rgb2gray(video(:,:,:,1)));         % Set reference frame
    FFT02   = fft2(SpatialWindow.*(ref-mean(ref(:))));  % Calculate ref FFT
    FMC02   = fft2(FMCWindow.*interp2(fftshift(abs(FFT02)),XLP,YLP,'linear',0));   % Take FFT of FMT on reference
    parfor k = 1:videoDims(end)
        cur = double(rgb2gray(video(:,:,:,k))); % Set current frame
        [dispX(k),dispY(k),dispS(k),~] = statisticalRegister(cur,FFT02,...
            FMC02,SpatialWindow,FMCWindow,videoDims(1),videoDims(2),2,MaxRad,...
            XCart,YCart,XLP,YLP,1E-1,20); % Run registration
%         fprintf('Evaluating frame %03i of %03i ...\r',k,videoDims(end));
    end
    clear FFT02 FMI02 FMC02 FMCWindow SpatialWindow ref cur
    %%%%%%%%%%%%%%%%%%%%%%%%%% Run Outlier Detection %%%%%%%%%%%%%%%%%%%%%%%%%%
    scaleThresh = -log(0.5)/log(MaxRad/2)*(videoDims(2) - 1);           % Simple threshold
    dispS       = velThresh(dispS,scaleThresh);                         % Velocity threshold scaling
    dispX       = velThresh(dispX,videoDims(2)/4);                      % Velocity threshold x-displacement
    dispY       = velThresh(dispY,videoDims(1)/4);                      % Velocity threshold y-displacment
    dispS       = velReplace(dispS');                                   % Scaling velocity replacement
    dispX       = velReplace(dispX');                                   % x-displacement velocity replacement
    dispY       = velReplace(dispY');                                   % y-displacement velocity replacement
    % dispS       = hampel(dispS,3,3);
    % dispX       = hampel(dispX,3,3);
    % dispY       = hampel(dispY,3,3);
    dispS(1) = 0;
    dispX(1) = 0;
    dispY(1) = 0;
    scale       = exp(-1 * log(MaxRad/2) * dispS/(videoDims(2) - 1));   % Convert scale
    %%%%%%%%%%%%%%%%%%%%% IMAGE REGISTRATION BLOCK END  %%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%% HAAR EYE DETECTOR BLOCK START %%%%%%%%%%%%%%%%%%%%%%%
    %%%%% STEP 1 - USING SUB-SAMPLED ROI, TRY TO FIND EYE
    rescaleFactor       = 4;
    storeEye            = zeros([videoDims(4),4]);                  % Preallocate feature matrix
    leDetector          = vision.CascadeObjectDetector('LeftEye');  % Left Eye features detector
    leDetector.MinSize  = ceil((1/5)*[min([videoDims(2) videoDims(1)]) ...
        min([videoDims(2) videoDims(1)])]/rescaleFactor); % Set threshold for minimum feature size
    leDetector.MaxSize  = ceil([max([videoDims(2) videoDims(1)]) ...
        max([videoDims(2) videoDims(1)])]/rescaleFactor); % Set threshold for maximum feature size
    leDetector.MergeThreshold = 4;  % Set merge threshold between levels
    reDetector          = vision.CascadeObjectDetector('RightEye'); % Left Eye features detector
    % Copy from Left Eye detector
    reDetector.MinSize  = leDetector.MinSize ;
    reDetector.MaxSize  = leDetector.MaxSize ;
    reDetector.MergeThreshold = leDetector.MergeThreshold;
    % Run Haar feature detector
    parfor k = 1:videoDims(end)
%         fprintf('Haar detector, frame %03i of %03i ...\r',k,videoDims(end));
        cur = video(:,:,:,k); Tf = [scale(k) 0 dispX(k); 0 scale(k) dispY(k); 0 0 1];
        for n = 1:3                                 % Register Images
            curF        = griddedInterpolant(XCart',YCart',double(cur(:,:,n))','linear','none');
            cur(:,:,n)  = uint8(tformImage(XCart,YCart,Tf,[videoDims(1) videoDims(2)],curF));
        end
        try
            img = cur(round(videoDims(1)/2+(-videoDims(2)/2:rescaleFactor:videoDims(2)/2)),...
                rescaleFactor:rescaleFactor:end,:);
        catch
            keyboard
        end
        % Run Haar feature detector using left and right eye detectors
        tempEyeL = step(leDetector,img); tempEyeR = step(reDetector,img);
        % Check if any of the feature detectors fail
        if isempty(tempEyeL), tempEyeL = tempEyeR; end
        if isempty(tempEyeR), tempEyeR = tempEyeL; end
        % For multiple outputs, parse through and attempt to find the best fit
        diffMat = zeros(size(tempEyeL,1),size(tempEyeR,1));
        try
            for p = 1:size(tempEyeR,1)
                diffMat(:,p) = abs(tempEyeL(:,1)-tempEyeR(p,1));
            end
            [indr,~] = find(diffMat == min(diffMat(:)));
            indL = indr(1);
            diffMat = zeros(size(tempEyeR,1),size(tempEyeL,1));
            for p = 1:size(tempEyeL,1)
                diffMat(:,p) = abs(tempEyeL(p,1)-tempEyeR(:,1));
            end
            [indr,~] = find(diffMat == min(diffMat(:)));
            indR = indr(1);
            boxEye = max(cat(1,tempEyeL(indL,:),tempEyeR(indR,:)),[],1);    % Use Mean between eyes
            if ~isempty(boxEye)
                storeEye(k,:) = boxEye;
            end
%             Ieye        = insertObjectAnnotation(img, 'rectangle',boxEye,'Eye');
%             figure(10), imshow(Ieye), title('Detected eye');
%             pause(1E-2);
        catch
            boxEye = [nan nan nan nan];
            storeEye(k,:) = boxEye;
        end
    end
    clear ref cur
    % Scale up detected features due to undersampling
    boxEye = rescaleFactor*round(nanmedian(storeEye,1));
    boxEye(:,2) = boxEye(:,2) + round(videoDims(1)/2-videoDims(2)/2);
    %%%%% STEP 2 - REFINEMENT OF WINDOW & PUPIL CENTER
    XCENT   = zeros(videoDims(4),1); YCENT = XCENT; posDiam = XCENT;
    for k = 1:videoDims(end)
%         fprintf('Refined Center, frame %03i of %03i ...\r',k,videoDims(end));
        cur = video(:,:,:,k); Tf = [scale(k) 0 dispX(k); 0 scale(k) dispY(k); 0 0 1];
        for n = 1:3                                 % Register Images
            curF        = griddedInterpolant(XCart',YCart',double(cur(:,:,n))');
            cur(:,:,n)  = uint8(tformImage(XCart,YCart,Tf,[videoDims(1) videoDims(2)],curF));
        end
        % Gaussian blur ROI for better center point detection
        roi = imfilter(cur,fspecial('gaussian',...
            round(0.1*min(videoDims(1:2))),0.01*min(videoDims(1:2))),255);
        % Tack image complement of ROI
        ROI = double(imcomplement(uint8(max(double(...
            roi(boxEye(2)+(1:(boxEye(4)-1)),...
            boxEye(1)+(1:(boxEye(3)-1)),:)),[],3))));
        [~,indROI] = max(ROI(:));   % Find index of maximum from image
        [ycent,xcent] = ind2sub(size(ROI),indROI);  % Go from index to subs
        XCENT(k) = xcent+boxEye(1); % Set new center X
        YCENT(k) = ycent+boxEye(2); % Set new center Y
        % Unwrap using Log Polar; use image to find new center
        imUnwrap = zeros(size(XLP,1),size(XLP,2),3);
        for n = 1:3
            imUnwrap(:,:,n) = interp2(double(squeeze(cur(:,:,n))),...
                XLP+(XCENT(1)-round(videoDims(2)/2)),...
                YLP+(YCENT(1)-round(videoDims(1)/2)),'linear',0);
        end
        posPoints = zeros(NoOfWedges,1);
        for i = 1:NoOfWedges
            [~,ind] = max(linspace(1,0,numel(XLP(i,:))).*...
                mean(imUnwrap(i,:,:),3));
            posPoints(i) = ind;
        end
        posPoints(posPoints <= 25) = nan;
        try
            posDiam(k) = sqrt((XLP(i,round(nanmedian(posPoints)))-XLP(i,1)).^2+...
                (YLP(i,round(nanmedian(posPoints)))-YLP(i,1)).^2);
        catch
            posDiam(k) = nan;
        end
%         figure(14); imagesc(cur); axis image; hold on; scatter(XCENT(k),YCENT(k),25,'filled');
%         hold off;
%         pause(1E-5);
    end
    clear ref cur
    xcent = nanmedian(XCENT); ycent = nanmedian(YCENT);
    %%%%%%%%%%%%%%%%%%%%%% HAAR EYE DETECTOR BLOCK END %%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % clear XLP YLP
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%% PUPIL DILATION START %%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%% Correlation settings block %%%%%%%%%%%%%%%%%%%%%%%%
    winsize  = ceil(0.45*(4*round(nanmedian(posDiam)))); % Set window size based on physical
    if isnan(mean(winsize))
        winsize = 128;
        storeEye = [round(videoDims(2)/2) round(videoDims(1)/2) 128 128];
    end
    dilateMaxRad        = winsize/2-1;  % Set maximum radius
    dilateSpatialWindow = hanning(winsize)*hanning(winsize)';
    dilateFMCWindow     = hanning(256)*hanning(winsize)';
    [xCart,yCart]       = meshgrid(ceil(-winsize/2):1:ceil(winsize/2-1),...
        ceil(-winsize/2):1:ceil(winsize/2-1));                % Cartesian Grid
    [xLP,yLP]           = LogPolarCoordinates([winsize, winsize],...
        256, winsize,2 , dilateMaxRad, 2*pi);   % Polar Grid
    dispS       = zeros([videoDims(4),1]);      % Preallocate dilation velocity
    dilateScale =  ones([videoDims(4),1]);      % Preallocate dilation scale
    xCenter     = round(nanmedian(storeEye(:,1))+nanmedian(storeEye(:,3))/2); % First Approx of x Center
    yCenter     = round(nanmedian(storeEye(:,2))+nanmedian(storeEye(:,4))/2); % First Approx of y Center
    %%%%%%%%%%%%%%%%%%%%%%%%%%% Correlation Process %%%%%%%%%%%%%%%%%%%%%%%%%%%
    frameStep   = 1;    % Set Frame Step to 1 (Using central Difference approach)
    tVect       = [0 frsrs(3:end)-(frsrs(:,3:end)-frsrs(:,1:end-2))/2-1 frsrs(end)-1]; % Time vector for scale
    tStep       = [2 (frsrs(:,3:end)-frsrs(:,1:end-2)) 2]/2;                           % Time vector for integration
    parfor k = 2:(videoDims(4)-1)
        current   = video(:,:,:,k+frameStep);  % Get current image
        reference = video(:,:,:,k-frameStep);  % Get reference image
        T1  = [ scale(k-frameStep) 0 dispX(k-frameStep); 0 scale(k-frameStep) dispY(k-frameStep); 0 0 1];
        T2  = [ scale(k+frameStep) 0 dispX(k+frameStep); 0 scale(k+frameStep) dispY(k+frameStep); 0 0 1];
        for ii = 1:3 % Register Images
            refF = griddedInterpolant(XCart',YCart',double(reference(:,:,ii))');
            curF = griddedInterpolant(XCart',YCart',double(current(:,:,ii))');
            reference(:,:,ii)   = uint8(tformImage(XCart,YCart,T1,...
                [videoDims(1) videoDims(2)],refF));     % Register reference
            current(:,:,ii)     = uint8(tformImage(XCart,YCart,T2,...
                [videoDims(1) videoDims(2)],curF));       % Register current
        end
        %%%%%%%%%%%%%%%%%%%%%% Local update of Image Centroid %%%%%%%%%%%%%%%%%%%%%
        %%%% Local update of Image Centroid %%%%%%%%%%%%%%%%%%%%%
        refIM = reference(yCart(:,1)+round(ycent),...
            xCart(1,:)+round(xcent),:); % Get final reference subregion
        curIM =   current(yCart(:,1)+round(ycent),...
            xCart(1,:)+round(xcent),:); % Get Final current subregion
        refIM = rgb2hsv(refIM); refIM(:,:,2) = histeq(refIM(:,:,2)); refIM(:,:,3) = histeq(refIM(:,:,3));
        refIM = hsv2rgb(refIM); refIM = uint8(round(refIM*255/max(refIM(:)))); % To RGB
        curIM = rgb2hsv(curIM); curIM(:,:,2) = histeq(curIM(:,:,2)); curIM(:,:,3) = histeq(curIM(:,:,3));
        curIM = hsv2rgb(curIM); curIM = uint8(round(curIM*255/max(curIM(:)))); % To RGB
        refIM = repmat(rgb2gray(refIM),[1 1 3]);
        curIM = repmat(rgb2gray(curIM),[1 1 3]);
        refIM = imcomplement(refIM);    % Take Image Complement for reference
        curIM = imcomplement(curIM);    % Take Image Complement for current
        for jj = 1:3
            refIM(:,:,jj) = medfilt2(refIM(:,:,jj),[3 3]);
            curIM(:,:,jj) = medfilt2(curIM(:,:,jj),[3 3]);
        end
        
%         figure(12); imagesc(imfuse(refIM,curIM)); pause(1E-5);
        
        frame01 = zeros(size(curIM)); % Preallocate Frames storage
        FFT02   = zeros([256 winsize size(curIM,3)]);    % Preallocate FFT storage
        refSize = size(refIM);
        for ii = 1:size(refIM,3)
            fr01 = double((curIM(:,:,ii)));           % Grab Current Frame, current channel
            frame01(:,:,ii) = fr01 - mean(fr01(:)); % Subtract Mean
            fr02 = double(refIM(:,:,ii));             % Grab Reference Frame, current channel
            fr02 = fr02 - mean(fr02(:));            % Subtract Mean
            FMI02 = interp2(fftshift(abs(fft2(dilateSpatialWindow.*...
                double(fr02)))),xLP, yLP, 'linear', 0);     % Compute Reference Frame FMI
            FFT02(:,:,ii) = fft2(dilateFMCWindow.*FMI02);   % Compute FFT of Reference FMI
        end
        [dispS(k),~,~] =...
            dilationEstimator(frame01,FFT02,xCart,yCart,xLP,yLP,...
            dilateSpatialWindow,dilateFMCWindow,2,...
            dilateMaxRad,1E-4,100);                 % Run Dilation Estimation
        dispS(k)         = dispS(k)/(2*tStep(k));   % Scale Displacement based on Frame Step
    end
    clear xLP yLP dilateFMCWindow dilateSpatialWindow ref cur
    %%%%%%%%%%%%%%%%%%%%%%%%%% PUPIL DILATION END %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%% TIME SERIES ANALYSIS START %%%%%%%%%%%%%%%%%%%%%%%%%
    scaleThresh = -log(0.8)/log(dilateMaxRad/2)*(256 - 1);  % Threshold for displacment
    dispSVal    = zeros(size(dispS));  % Preallocate Validation Vector
    for ii = 1:1
        dispSVal(:,ii) = hampel(dispS(:,ii),3,3);
        dispSVal(:,ii) = velThresh(dispSVal(:,ii),scaleThresh);    % Perform Velocity Thresholding
        dispSVal(:,ii) = velReplace(dispSVal(:,ii)');           % Perform Velocity Replacement
    end
    % Use Biometrics and Scaling to give physical units to dilation
    [xCart,yCart]   = meshgrid(ceil(-min(nanmedian(storeEye(:,end-1:end),1))/2):1:ceil(min(nanmedian(storeEye(:,end-1:end),1))/2-1),...
        ceil(-min(nanmedian(storeEye(:,end-1:end),1))/2):1:ceil(min(nanmedian(storeEye(:,end-1:end),1))/2-1));                % Cartesian Grid

    refImg = video(yCart(:,1)+round(ycent),xCart(1,:)+round(xcent),:,1);
    refImg = rgb2hsv(refImg); refImg(:,:,2) = histeq(refImg(:,:,2)); refImg(:,:,3) = histeq(refImg(:,:,3));
    refImg = hsv2rgb(refImg); refImg = uint8(round(refImg*255/max(refImg(:)))); % To RGB
    pupilDia        = find(rgb2gray(refImg(ceil(min(nanmedian(storeEye(:,end-1:end),1))/2),:,:)) < 0.1*max(rgb2gray(refImg(ceil(min(nanmedian(storeEye(:,end-1:end),1))/2),:,:))));
    pixeltomm       = min(median(storeEye(:,end-1:end),1))/24;
    approxPupilDia  = numel(pupilDia)/pixeltomm;
    dilationRatio   = exp(1*log(dilateMaxRad/2)*...
        (cumsum(dispSVal.*tStep')*2)/(winsize));   % Compute Dilation
    dilation    = dilationRatio * approxPupilDia;
    
%     figure(18);
%     plot(tVect*(1/30),dilationRatio,'LineWidth',2);
%     grid on;
%     set(gca,'LineWidth',2,'FontSize',18,'FontWeight','bold');
%     xlabel('Recording Time (s)','FontSize',18,'FontWeight','bold');
%     ylabel('Dilation Ratio','FontSize',18,'FontWeight','bold');
%     set(gcf,'Color',[1 1 1]);
%     pause(1E-5);
    fid = fopen(fullfile(pathName,[fileBase,'_results.txt']),'w');  % Open Text File
    fprintf(fid, [ 'Time(s)' ' ' 'Dilation Ratio' ' ' ' Pupil Diameter ' ' ' 'Reponse Rate' '\n']);  % Specify Headers
    fprintf(fid, '%f %f %f %f \n', [tVect(:)*(1/30) dilationRatio(:) dilation(:)  dispSVal(:)]');   % Print Results
    fclose(fid);    % Close Text File
    % Compute onset, max constriction, 75% recovery, average constriction
    % velocity and average dilation velocity
    try
        onsetInd            = find(tstamp == 1);
        onsetTime           = tVect(onsetInd(1) + 1);
        constrictInd        = find(dilationRatio == min(dilationRatio(:)));
        maxConstrictTime    = tVect(constrictInd);
        recoveryInd         = find(dilationRatio(constrictInd:end) >=...
            0.75*abs(1-dilationRatio(constrictInd))+dilationRatio(constrictInd));
        if isempty(recoveryInd)
            recoveryInd  = numel(tVect)-constrictInd;
            recoveryTime = tVect(end);
        else
            recoveryTime = tVect(constrictInd+recoveryInd(1));
        end
        
        tempInd     = find(tstamp == 1);
        recovInd    = constrictInd+recoveryInd(1);
        try
            averageConstriction = polyfit(tVect(tempInd(1)-2:constrictInd)',dilationRatio(tempInd(1)-2:constrictInd),1);
            averageDilation     = polyfit(tVect(constrictInd:recovInd(1))',dilationRatio(constrictInd:recovInd(1)),1);
        catch
            averageConstriction = polyfit(tVect(tempInd(1):constrictInd)',dilationRatio(tempInd(1):constrictInd),1);
            averageDilation     = polyfit(tVect(constrictInd:recovInd(1))',dilationRatio(constrictInd:recovInd(1)),1);
        end
        
        fid = fopen(fullfile(pathName,[fileBase,'_timeSeriesAnalysis.txt']),'w');  % Open Text File
        fprintf(fid, [ 'Onset Time' ' ' 'Max Constriction Time' ' ' 'Max Constriction Ratio' ' ' ' Recovery Time ' ' ' 'Average Constriction' ' ' 'Average Dilation' '\n']);  % Specify Headers
        fprintf(fid, '%f %f %f %f %f %f \n', [onsetTime maxConstrictTime dilationRatio(constrictInd) recoveryTime  averageConstriction(1) averageDilation(1)]');   % Print Results
        fclose(fid);    % Close Text File
    catch
        keyboard
    end
    save(fullfile(pathName,[fileBase,'.mat']));
    
    toc
    delete(poolobj);
catch
    fid = fopen(fullfile(pathName,[fileBase,'_results.txt']),'w');  % Open Text File
    fclose(fid);    % Close Text File
    fid = fopen(fullfile(pathName,[fileBase,'_timeSeriesAnalysis.txt']),'w');  % Open Text File
    fclose(fid);    % Close Text File
    delete(poolobj);
end
%%%%%%%%%%%%%%%%%%%%%%% TIME SERIES ANALYSIS END %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

function [dispX,dispY,dispS,iteration] =  statisticalRegister(frame01,...
    FFT02,FMC02,SpatialWindow,FMCWindow,height,width,MinRad,MaxRad,...
    xCart,yCart,xLP,yLP,err_thresh,iteration_thresh)
% IMAGE REGISTRATION BY ENSEMBLE CROSS-CORRELATION OF ALL CHANNELS
err     = numel(frame01(:)); iteration= 1; % Initialize Error & Iterations
scale   = 1; dispX = 0; dispY = 0; dispS = 0; % Initialize disp & scale
imgF    = griddedInterpolant(xCart',yCart',frame01');
while err >= err_thresh && iteration <= iteration_thresh
    T1      = [scale 0 dispX; 0 scale dispY; 0 0 1]; % Build Scaling Matrix
    fr01    = tformImage(xCart,yCart,T1,[height width],imgF);
    FFT01   = fft2(SpatialWindow.*(fr01-mean(fr01(:)))); % FFT of frame 01
    FMI01   = interp2(fftshift(abs(FFT01)),xLP, yLP, 'linear', 0);
    FMC01   = fft2(FMCWindow.*FMI01);
    FMC01   = FMC01./numel(FMC01(:));
    ccSpectral  = FFT01.*conj(FFT02); % Do Standard Cross-Correlation
    ccSpatial   = fftshift(abs(ifft2(ccSpectral)));
    [peakLocX, peakLocY] = subPixel2D(ccSpatial);
    tdx     = peakLocX;     tdy     = peakLocY;
    dispX   = dispX + tdx;  dispY   = dispY + tdy;
    ccSpectral  = FMC01.*conj(FMC02); % Do Standard Cross-Correlation
    ccSpatial   = fftshift(abs(ifft2(sum(ccSpectral,3))));
    [peakLocX, ~]   = subPixel2D(ccSpatial);
    sdx     = peakLocX;
    dispS   = dispS + sdx;
    scale   = exp(-1 * log(MaxRad/MinRad) * dispS/(width - 1));
    err     = max([sqrt(tdx.^2+tdy.^2),sqrt(sdx^2)]);
    iteration   = iteration + 1;
end
end

function [dispS,minRad,maxRad] =  dilationEstimator(frame01,FFT02,...
    xCart,yCart,xLP,yLP,SpatialWindow,FMCWindow,...
    minRad,maxRad,err_thresh,iter_thresh)
%%%%%% PUPIL DILATION BLOCK START %%%%%
sizeFrame = size(frame01);
if numel(sizeFrame) == 2
    sizeFrame(3) = 1;
end
err = 1; iter= 1; dispS = 0; dilation = 1;
while err >= err_thresh && iter <= iter_thresh
    % Build Scaling Matrix
    T1 = [dilation 0 0;...
        0 dilation 0; 0 0 1];
    FFT01       = zeros(size(FFT02));
    for ii = 1:sizeFrame(end)
        imgF = griddedInterpolant(xCart',yCart',frame01(:,:,ii)');
        fr01 = SpatialWindow.*tformImage(xCart,yCart,T1,...
            [size(xCart,1) size(xCart,2)],imgF);
        fr01 = fftshift(abs(fft2(fr01)));
        % Log-polar unwrap first image subregion
        FMI01 = interp2( fr01, xLP, yLP, 'linear', 0);
        % Take FFT of First Image
        FFT01(:,:,ii) = fft2(FMCWindow.*FMI01);
    end
    % Do Standard Cross-Correlation
    ccSpectral  = mean(FFT01.*conj(FFT02),3);
    [ccPhase, spectFilt] = split_complex(ccSpectral);
    %     spectFilt = fftshift(energyfilt(size(ccPhase,2),size(ccPhase,1),[6 6],0));
    % Convert from Spectral to Spatial Domain
    ccSpatial = fftshift(abs(ifft2(ccPhase.*...
        spectFilt)));
    % Find Scale Shift
    [peakLocX, ~] = subPixel2D(ccSpatial);
    dispS = dispS + peakLocX;
    % Estimate dilation by conversion
    dilation = (maxRad/minRad)^(-dispS/size(ccSpatial,2));
    %     dilation = exp(-1 * ...
    %         log(maxRad/minRad) * ...
    %         dispS/(size(ccSpatial,2) - 1));
    % Cut-off criteria for convergence
    err = sqrt((peakLocX)^2);
    iter= iter+1;
    %     keyboard
end
% keyboard
end

function [XLP, YLP] = LogPolarCoordinates(IMAGESIZE, NUMWEDGES,...
    NUMRINGS, RMIN, RMAX, MAXANGLE)
% LOGPOLARCOORDINATES Constructs polar coordinate matrix values which map a
%   variable of interest from Cartesian space to polar space.
h       = IMAGESIZE(1); % Image Height
w       = IMAGESIZE(2); % Image Width
nw      = NUMWEDGES;    % Number of Wedges
nr      = NUMRINGS;     % Number of Rings
rMax    = RMAX;         % Maximum Radius
rMin    = RMIN;         % Minimum Radius
xZero   = (w + 1)/2;    % X offset
yZero   = (h + 1)/2;    % Y offset
logR    = linspace(log(rMin), log(rMax), nr);
rv      = exp(logR);
thMax   =  MAXANGLE * (1 - 1 / nw);
thv     = linspace(0, thMax, nw);
[r, th] = meshgrid(rv, thv);    % Build RHO-Theta Grids
[x, y]  = pol2cart(th, r);      % Convert to Cartesian
XLP     = x + xZero;            % Apply X offset
YLP     = y + yZero;            % Aplly Y offset
end

function [SPECTRAL_PHASE, SPECTRAL_MAGNITUDE] = split_complex(COMPLEX_CROSS_CORRELATION_PLANE)
% PHASE_ONLY_CORRELATION_PLANE = phaseOnlyFilter(COMPLEX_CROSS_CORRELATION_PLANE)
% performs phase-only filtering of a spectral-domain cross correlation signal

% Calculate the spectral magnitude of the complex cross correlation
% in the frequency domain
SPECTRAL_MAGNITUDE = abs(COMPLEX_CROSS_CORRELATION_PLANE);
% Set zeros to ones
SPECTRAL_MAGNITUDE(SPECTRAL_MAGNITUDE == 0) = 1;
% Divide cross correlation by its nonzero magnitude to extract the phase information
SPECTRAL_PHASE = COMPLEX_CROSS_CORRELATION_PLANE ./ SPECTRAL_MAGNITUDE;
% Replace infinites with the original complex value.
SPECTRAL_PHASE(isinf(SPECTRAL_PHASE)) = ...
    COMPLEX_CROSS_CORRELATION_PLANE(isinf(SPECTRAL_PHASE));
end

function imgtform = tformImage(x,y,M,S,imgint)
% TFORMIMAGE Maps images from reference coordinates to current coordinates
%   after image deformation(scaling), shifting, shearing, or rotation
% Inputs:
%   x is the reference N x M coordinates in the x-direction
%   y is the reference N x M coordinates in the y-direction
%   M is the 3 x 3 matrix of global mapping values
%   S is the size of the current image
%   imgint is the intial image
% OUTPUTS:
%   imgtform is the mapped image to the current coordinates

% 2 x nPoints vector of coordinates.
interpPoints = M \ [x(:)'; y(:)'; ones(size(y(:)))'];
% Generate mapped image from raw image
imgtform = imgint(reshape(interpPoints(1,:),S)',...
    reshape(interpPoints(2,:),S)')';
end

function [dispX, dispY, dX, dY] = subPixel2D(plane)
planeShape  = size(plane);
[lY,lX] = find(plane == max(plane(:)));
subPixelOffset = zeros([2 1]);
if mod(planeShape(1),2) == 1
    subPixelOffset(1) = -0.5;
end
if mod(planeShape(2),2) == 1
    subPixelOffset(2) = -0.5;
end
dispX = planeShape(2)/2 - lX + subPixelOffset(2) + 1;
dispY = planeShape(1)/2 - lY + subPixelOffset(1) + 1;
if lY <= planeShape(1)-1 && lY >= 2 && lX <= planeShape(2)-1 && lX >= 2
    lPm1X       = log( plane( lY     , lX - 1 ));
    lP00X       = log( plane( lY     , lX     ));
    lPp1X       = log( plane( lY     , lX + 1 ));
    shiftErrX   = ( lPm1X - lPp1X )/( 2*( lPm1X + lPp1X - 2 * lP00X ));
    betax       = abs(lPm1X-lP00X)/((-1-shiftErrX)^2-(shiftErrX)^2);
    dX          = 4/sqrt((2*betax));
    
    lPm1Y       = log( plane( lY - 1 , lX     ));
    lP00Y       = log( plane( lY     , lX     ));
    lPp1Y       = log( plane( lY + 1 , lX+1   ));
    shiftErrY   = ( lPm1Y - lPp1Y )/( 2*( lPm1Y + lPp1Y - 2 * lP00Y ));
    betay       = abs(lPm1Y-lP00Y)/((-1-shiftErrY)^2-(shiftErrY)^2);
    dY          = 4/sqrt((2*betay));
    
    dispY = dispY - shiftErrY;
    dispX = dispX - shiftErrX;
end
end

function [x] = velThresh(x,threshold)
% [x] = velThresh(x,threshold) Performs thresholding of velocity values
% within the vector entry, x.
x(abs(x) > threshold) = nan;
end

function [x] = movingMADUOD(x,iter,wsize,threshold)
% [x] = movingMADUOD(x,threshold) Performs regional Median Absolute
% Difference (MAD) Outlier detection
% Example: dispX = movingMADUOD(dispX,1,9,2);
for k = 1:iter
    for n = 1:numel(x)
        if ~isnan(x(n))
            % Make sure left index stays in vector space
            ind1 = max([1 n-(wsize+1)/2]);
            % Make sure right index stays in vector space
            ind2 = min([numel(x) n+(wsize+1)/2]);
            % Pull window (or block in this case) of vector entry values
            block = x(ind1:ind2);
            % Remove the "value of interest" from the block
            irmv = find(block == x(n));
            X = block(irmv);
            block(irmv) = [];
            % Calculate the block median & MAD
            block = sort(block);
            M = nanmedian(block);
            MAD = nanmedian(abs(x-M));
            % Compute the z-score (statistical score)
            R = abs(X-M)/MAD;
            % If score is above threshold, set "value of interest" to NaN
            if R > threshold
                x(n) = nan;
            end
        end
    end
end
end

function [x] = velReplace(x)
% [x] = velReplace(x) Performs weighted Median outlier replacement for NaN
% values in a vector entry
% Get the full index of vector entry
indvect = 1:numel(x);
% Discard index locations that are NaN
indvect(isnan(x)) = [];
% In the for-loop, step through, find NaNs, and replace
for n = 1:numel(x)
    if isnan(x(n))
        % Find & Sort the entry distances from the "value of interest"
        [distvect,I] = sort(abs(n-indvect));
        % Preserve only the non-NaN entries
        ind = indvect(I);
        % Perform weighted-median replacement
        x(n) = sum(distvect*x(ind)')/sum(distvect);
    end
end
end

function [W] = energyfilt(Nx,Ny,d,q)
% --- RPC Spectral Filter Subfunction ---
if numel(d) == 1
    d(2) = d;
end
%assume no aliasing
if nargin<4
    q = 0;
end
%initialize indices
[k1,k2]=meshgrid(-pi:2*pi/Ny:pi-2*pi/Ny,-pi:2*pi/Nx:pi-2*pi/Nx);
%particle-image spectrum
Ep = (pi*255*(d(1)*d(2))/8)^2*exp(-d(2)^2*k1.^2/16).*exp(-d(1)^2*k2.^2/16);
%aliased particle-image spectrum
Ea = (pi*255*(d(1)*d(2))/8)^2*exp(-d(2)^2*(k1+2*pi).^2/16).*exp(-d(1)^2*(k2+2*pi).^2/16)+...
    (pi*255*(d(1)*d(2))/8)^2*exp(-d(2)^2*(k1-2*pi).^2/16).*exp(-d(1)^2*(k2+2*pi).^2/16)+...
    (pi*255*(d(1)*d(2))/8)^2*exp(-d(2)^2*(k1+2*pi).^2/16).*exp(-d(1)^2*(k2-2*pi).^2/16)+...
    (pi*255*(d(1)*d(2))/8)^2*exp(-d(2)^2*(k1-2*pi).^2/16).*exp(-d(1)^2*(k2-2*pi).^2/16)+...
    (pi*255*(d(1)*d(2))/8)^2*exp(-d(2)^2*(k1+0*pi).^2/16).*exp(-d(1)^2*(k2+2*pi).^2/16)+...
    (pi*255*(d(1)*d(2))/8)^2*exp(-d(2)^2*(k1+0*pi).^2/16).*exp(-d(1)^2*(k2-2*pi).^2/16)+...
    (pi*255*(d(1)*d(2))/8)^2*exp(-d(2)^2*(k1+2*pi).^2/16).*exp(-d(1)^2*(k2+0*pi).^2/16)+...
    (pi*255*(d(1)*d(2))/8)^2*exp(-d(2)^2*(k1-2*pi).^2/16).*exp(-d(1)^2*(k2+0*pi).^2/16);
%noise spectrum
En = pi/4*Nx*Ny;
%DPIV SNR spectral filter
W  = Ep./((1-q)*En+(q)*Ea);
W  = W'/max(max(W));
end

function [Q] = simpsonH(F,h)
%   [Q] = function SimpsonH(F,h);
%
%   Perform numerical integration of function f(x) on x=[a,b] sampled at
%   k evenly spaced grid points on interval h=(b-a)/(k-1).
%
%   If k is odd, normal composite Simpson's rule will be used.
%   If k is even, Simpson's 3/8's rule will be used to complete the final
%   interval.
%
%   Integration will be performed on the columns of F.
[M,~] = size(F);
%could add check to transpose if data is in single row instead of column
if M <3
    error('F must have at least 3 points to integrate')
end
W = zeros(1,M);
if M==4             %use only 3/8's rule
    W = [1 3 3 1];
    Q = 3/8*h*W*F;
elseif mod(M,2)     %M is odd - use only Simpson's rule
    W(1) = 1;
    for m=2:2:M-1
        W(m) = 4;
    end
    for m=3:2:M-2
        W(m) = 2;
    end
    W(M) = 1;
    Q = h/3*W*F;
else                %M is even - need 3/8's rule to finish interval
    W(1) = 8;
    for m=2:2:M-4
        W(m) = 32;
    end
    for m=3:2:M-5
        W(m) = 16;
    end
    W(M-3)=17;
    W(M-2)=27;
    W(M-1)=27;
    W(M) = 9;
    Q = h/24*W*F;
end
end