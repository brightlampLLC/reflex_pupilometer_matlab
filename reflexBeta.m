function reflexBeta(vidName)
% FUNCTION reflexPupillometerBeta is a wrapper-code designed to measure the
% rate of pupil dilation in the presence of visible stimulation. This code
% is broken into the following blocks:
tic
[pathName,fileBase,fileExt]  = fileparts(vidName);                          % Parse through file name parts
vidName  =  fullfile(pathName,[fileBase,fileExt]);                          % Rebuild filename
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% VIDEO READER AND IMAGE LOADER BLOCK START %%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
vid     = VideoReader(vidName);                                             % Load video using reader
fStart  = round(0.8*vid.FrameRate)+1;                                       % Set Start frame number (0.8 seconds in)
tEnd    = 5.8;                                                              % Set approximate end time (at 5.8 seconds)
vid.CurrentTime = fStart/vid.FrameRate;                                     % Set Current Time to start frame
frameTimeSeries = linspace(1,vid.FrameRate*vid.Duration,...
    vid.FrameRate*vid.Duration)/vid.FrameRate;                              % Generate frame time series
capTime     = find(frameTimeSeries <= tEnd);                                % Use end time to find cut-off
deltaTime   = vid.FrameRate;
% CHECK VIDEO SIZES TO MAKE SURE THAT VIDEO IS ORIENTED & SIZED CORRECTLY
if vid.Height > vid.Width
    if vid.Height > 1920 && vid.Width > 1080
        videoDims = [1920, 1080, 3, (capTime(end)-(fStart-1))];             % Get loaded video dimensions
    else
        videoDims = [vid.Height, vid.Width, 3, (capTime(end)-(fStart-1))];  % Get loaded video dimensions
    end
    video   = uint8(zeros(videoDims));                                      % Preallocate memory for video images
    counter = 1;                                                            % Initialize video read counter
    while vid.CurrentTime <= tEnd && vid.CurrentTime < vid.Duration
        video(:,:,:,counter) = imresize(readFrame(vid),...
            [videoDims(1) videoDims(2)],'nearest');                         % Read in video using readFrame
        counter = counter + 1;                                              % Update video read counter
    end
elseif vid.Height < vid.Width
    if vid.Width > 1080 && vid.Height > 1920
        videoDims = [1920, 1080, 3, (capTime(end)-(fStart-1))];             % Get loaded video dimensions
    else
        videoDims = [vid.Width, vid.Height, 3, (capTime(end)-(fStart-1))];  % Get loaded video dimensions
    end
    video   = uint8(zeros(videoDims));                                      % Preallocate memory for video images
    counter = 1;                                                            % Initialize video read counter
    while vid.CurrentTime <= tEnd && vid.CurrentTime < vid.Duration
        video(:,:,:,counter) = imresize(permute(readFrame(vid),[2 1 3]),...
            [videoDims(1) videoDims(2)],'nearest');                         % Read in video using readFrame
        counter = counter + 1;                                              % Update video read counter
    end
end
video(:,:,:,counter:end) = [];                                              % Remove any extra empty frames
frskip      = 1;                                                            % Frame skip index values
frsrs       = linspace(0,counter-1,counter)+fStart;                         % Build frame series vector
video       = video(:,:,:,frskip:frskip:end);                               % Store final video series that will be processed
videoDims   = size(video);                                                  % Get new video dimensions
frameMedian = double(mean(reshape(video,...
    videoDims(1)*videoDims(2)*videoDims(3),videoDims(4)),1));               % Compute mean of each frame
tstamp      = logical((frameMedian/255 > 0.70));                            % Threshold to remove frames that are too bright
tInd        = find(tstamp ==1);                                             % Find indices equal to 1 (too bright)
tstamp(tInd(1)-1)   = 1;                                                    % Make sure to remove a frame before its too bright
tstamp(tInd(end)+1) = 1;                                                    % Make sure to remove a frame after its too bright
frsrs(tstamp)       = [];                                                   % Remove frames that are too bright (equal to 1)
video(:,:,:,tstamp) = [];                                                   % Remove frames that are too bright (equal to 1)
videoDims           = size(video);                                          % Get new video dimensions
clear vid capTime tEnd fStart counter frameTimeSeries frskip frameMedian tInd
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% VIDEO READER AND IMAGE LOADER BLOCK END %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% try
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% IMAGE REGISTRATION BLOCK START  %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%% Registration settings block %%%%%%%%%%%%%%%%%%%%%%%
rescaleFactor   = 4;                                                        % Rescaling factor to reduce resolution (speed-up processing)
refImg          = double(imresize(video(:,:,1,1),1/rescaleFactor));         % Grab reference image, rescale
NoOfWedges      = 360;                                                      % FMC operator number of wedges
MinRad          = 2;                                                        % FMC operator minimum radius
MaxRad          = min(size(refImg,1), size(refImg,2))/2;                    % FMC operator maximum radius
SpatialWindow   = hanning(size(refImg,1))*hanning(size(refImg,2))';         % Spatial Apod Window
FMCWindow       = hanning(NoOfWedges)*hanning(size(refImg,2))';             % FMC Apod Win
[XCart, YCart]  = meshgrid(-videoDims(2)/2:1:(videoDims(2)/2-1),...
    -videoDims(1)/2:1:(videoDims(1)/2-1));                                  % Cartesian grid
[xSub, ySub]  = meshgrid(-size(refImg,2)/2:1:(size(refImg,2)/2-1),...
    -size(refImg,1)/2:1:(size(refImg,1)/2-1));                              % Cartesian grid for reduced resolution
[XLP, YLP]      = LogPolarCoordinates([size(refImg,1),size(refImg,2)],...
    NoOfWedges, size(refImg,2), MinRad , MaxRad, 2*pi);                     % Log-Polar grid
dispX = zeros([videoDims(4),1]);                                            % Preallocate horizontal (x-axis) displacement
dispY = zeros([videoDims(4),1]);                                            % Preallocate vertical (y-axis) displacement
dispS = zeros([videoDims(4),1]);                                            % Preallocate scaling displacement
refF  = griddedInterpolant(xSub',ySub',refImg','spline','linear');            % Build reference frame interp Function
%%%%%%%%%%%%%%%%%%%%%%%%%%% Registration Process %%%%%%%%%%%%%%%%%%%%%%%%%%
for k = 1:videoDims(end)
    curF    = griddedInterpolant(xSub',ySub',...
        double(imresize(video(:,:,1,k),1/rescaleFactor))','spline','linear'); % Build reference frame interp Function
    [dispX(k),dispY(k),dispS(k),~] = statisticalRegister(refF,curF,...
        SpatialWindow,FMCWindow,size(refImg,1),size(refImg,2),...
        MinRad,MaxRad,xSub,ySub,XLP,YLP,1E-2/(rescaleFactor^2),50);         % Run registration
end
%%%%%%%%%%%%%%%%%%%%%%%%%% Run Outlier Detection %%%%%%%%%%%%%%%%%%%%%%%%%%
scaleThresh = -log(0.5)/log(MaxRad/MinRad)*(size(refImg,2));                % Scale threshold
dispS       = velThresh(dispS,scaleThresh);                                 % Velocity threshold scaling
dispX       = velThresh(dispX,size(refImg,2)/4);                            % Velocity threshold x-displacement
dispY       = velThresh(dispY,size(refImg,1)/4);                            % Velocity threshold y-displacment
dispS       = velReplace(dispS');                                           % Scaling velocity replacement
dispX       = velReplace(dispX');                                           % x-displacement velocity replacement
dispY       = velReplace(dispY');                                           % y-displacement velocity replacement
dispS(1)    = 0; dispX(1) = 0; dispY(1) = 0;
scale       = (MaxRad/MinRad).^(-dispS/size(refImg,2));                     % Convert scale
clear XLP YLP NoOfWedges MinRad MaxRad dispS SpatialWindow FMCWindow
clear refF curF scaleThresh refImg k
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% IMAGE REGISTRATION BLOCK END  %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% HAAR EYE DETECTOR BLOCK START %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% STEP 1 - USING SUB-SAMPLED ROI, TRY TO FIND EYE
storeEye     = zeros([videoDims(4),4]);                                     % Preallocate feature matrix
eyeDetector  = vision.CascadeObjectDetector(...
    fullfile(pwd,'haarcascade_eye.xml'));
eyeDetector.MinSize         = ceil((1/4)*[min([videoDims(2) videoDims(1)]) ...
    min([videoDims(2) videoDims(1)])]/rescaleFactor);                       % Minimum threshold feature size
eyeDetector.MaxSize         = ceil([max([videoDims(2) videoDims(1)]) ...
    max([videoDims(2) videoDims(1)])]/rescaleFactor);                       % Maximum threshold feature size
eyeDetector.MergeThreshold  = 5;                                            % Set merge threshold between levels
for k = 1:videoDims(end)                                                    % Run Haar feature detector
    cur = imresize(video(:,:,:,k),1/rescaleFactor);                         % Grab current frame, resize
    Tf = [scale(k) 0 dispX(k); 0 scale(k) dispY(k); 0 0 1];                 % Construct current frame transform matrix
    for n = 1:3                                                             % Register Images
        curF = griddedInterpolant(xSub',ySub',...
            double(cur(:,:,n))','linear','none');
        cur(:,:,n) = uint8(tformImage(xSub,ySub,Tf,...
            [size(cur,1) size(cur,2)],curF));
    end
    cur(isnan(cur)) = 0;
    tempEye = step(eyeDetector,cur);                                        % Run Haar detector
    roi     = imfilter(cur,fspecial('gaussian',[25 25],3),255);             % Filter current image with gaussian filter
    try
        [tempEye,~]     = sortrows(tempEye,3,'descend');                    % Sort Haar detector results to list largest region first
        storeEye(k,:)   = rescaleFactor*tempEye(1,:);                       % Rescale Haar detector results
        ROI             = double(imcomplement(uint8(max(double(...
            roi(tempEye(2)+(1:(tempEye(4)-1)),...
            tempEye(1)+(1:(tempEye(3)-1)),:)),[],3))));                     % Crop to eye, take image complement
        [~,indROI]      = max(ROI(:));                                      % Find index of maximum from image
        [ycent,xcent]   = ind2sub(size(ROI),indROI);                        % Go from index to subs
        storeEye(k,1)   = rescaleFactor*(xcent+tempEye(1));                 % Set new center X
        storeEye(k,2)   = rescaleFactor*(ycent+tempEye(2));                 % Set new center Y
        storeEye(k,3)   = storeEye(k,3)/2;
        storeEye(k,4)   = storeEye(k,4)/2;
    catch
        storeEye(k,:)   = NaN(1,4);
    end
end
xcent = round(nanmedian(storeEye(:,1)));                                    % Update eye center x-position
ycent = round(nanmedian(storeEye(:,2)));                                    % Update eye center y-position
dispX = dispX*rescaleFactor;                                                % Rescale registration x-position
dispY = dispY*rescaleFactor;                                                % Rescale registration y-position
clear ref cur roi eyeDetector curF tempEye k n ROI rescaleFactor indROI Tf
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%% HAAR EYE DETECTOR BLOCK END %%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%% PUPIL DILATION START %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%% Correlation settings block %%%%%%%%%%%%%%%%%%%%%%%%
winsize             = ceil(0.8*nanmedian(nanmedian(storeEye(:,3:4),2),1));  % Subregion window size from Haar detector
dilateMinRad        = 1;                                                    % Dilation FMC minimum radius
dilateMaxRad        = winsize/2;                                            % Dilation FMC maximum radius
dilateNoOfWedges    = 360;                                                  % Dilation FMC number of wedges
dilateSpatialWindow = hanning(winsize)*hanning(winsize)';                   % Dilation Spatial window
dilateFMCWindow     = hanning(dilateNoOfWedges)*hanning(winsize)';          % Dilation FMC window
[xCart,yCart]       = meshgrid(ceil(-winsize/2):1:ceil(winsize/2-1),...
    ceil(-winsize/2):1:ceil(winsize/2-1));                                  % Cartesian Grid
[xLP,yLP]           = LogPolarCoordinates([winsize, winsize],...
    dilateNoOfWedges, winsize, dilateMinRad , dilateMaxRad, 2*pi);          % Polar Grid
dispS       = zeros([videoDims(4),1]);                                      % Preallocate dilation velocity
dispx       = zeros([videoDims(4),1]);                                      % Preallocate dilation velocity
dispy       = zeros([videoDims(4),1]);                                      % Preallocate dilation velocity
tVect = [frsrs(3:end)-(frsrs(:,3:end)-frsrs(:,1:end-2))/2-1 ...
    frsrs(end)-1]/deltaTime;                                                % Time vector for scale
tStep = [2 (frsrs(:,3:end)-frsrs(:,1:end-2)) 2]/2;                          % Time vector for integration
%%%%%%%%%%%%%%%%%%%%%%%%%%% Correlation Process %%%%%%%%%%%%%%%%%%%%%%%%%%%
for k = 2:(videoDims(4)-1)                                                  % Register image pair, perform image pre-processing
    reference   = rgb2hsv(video(:,:,:,k-1));                                % Get reference image
    current     = rgb2hsv(video(:,:,:,k+1));                                % Get current image
    T1      = [ scale(k-1) 0 dispX(k-1); 0 scale(k-1) dispY(k-1); 0 0 1];   % Set reference transform matrix
    T2      = [ scale(k+1) 0 dispX(k+1); 0 scale(k+1) dispY(k+1); 0 0 1];   % Set reference transform matrix
    refF    = griddedInterpolant(XCart',YCart',...
        double(reference(:,:,3))','spline','linear');
    curF    = griddedInterpolant(XCart',YCart',...
        double(current(:,:,3))','spline','linear');
    refIM   = tformImage(XCart,YCart,T1,...
        [videoDims(1) videoDims(2)],refF)*255;                              % Register reference
    curIM   = tformImage(XCart,YCart,T2,...
        [videoDims(1) videoDims(2)],curF)*255;                              % Register current
    refIM(isnan(refIM)) = 0; curIM(isnan(curIM)) = 0;
    refIM   = double(imcomplement(medfilt2(histeq(uint8(...
        refIM(yCart(:,1)+ycent,xCart(1,:)+xcent,:))))));                    % Reference subregion
    curIM   = double(imcomplement(medfilt2(histeq(uint8(...
        curIM(yCart(:,1)+ycent,xCart(1,:)+xcent,:))))));                    % Current subregion
%     figure(12); imagesc(imfuse(curIM,refIM)); pause(1E-2);
    %%%%% Peform correlation process
    fr01F = griddedInterpolant(xCart',yCart',refIM','spline','linear');
    fr02F = griddedInterpolant(xCart',yCart',curIM','spline','linear');
    try
        [dispS(k),dispx(k),dispy(k)] =  dilationEstimator(fr01F,fr02F,...
            xCart,yCart,xLP,yLP,dilateSpatialWindow,dilateFMCWindow,...
            dilateMinRad,dilateMaxRad,winsize,1E-3,100);
        dispS(k)    = dispS(k)/(2*tStep(k));                                % Scale Displacement based on Frame Step
    catch
        dispS(k) = nan;
    end
end
clear xLP yLP dilateFMCWindow dilateSpatialWindow dilateNoOfWedges
clear current reference T1 T2 refF curF refIM curIM fr01F fr02F
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
scaleThresh     = -log(0.8)/log(dilateMaxRad/dilateMinRad)*(winsize);       % Threshold for displacment
dispSVal        = velThresh(dispS,scaleThresh);                             % Perform Velocity Thresholding
dispSVal        = velReplace(dispSVal');                                    % Perform Velocity Replacement
dispSVal        = filloutliers(dispSVal,'pchip','movmedian',3);             % Perform outlier detection
dilationRatio   = (dilateMaxRad/dilateMinRad).^...
    ((cumsum(-dispSVal(:).*tStep(1:end-1)',1)*2)/(winsize));                    % Compute Dilation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%% PUPIL DILATION END %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%% TIME SERIES ANALYSIS START %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
refImg      = rgb2hsv(video(yCart(:,1)+round(ycent),...
    xCart(1,:)+round(xcent),:,1));                                          % Crop to only the eye
refImg      = histeq(refImg(:,:,3));                                        % Equalize in RGB space
pupilProps  = regionprops(imbinarize(imcomplement(uint8(255*refImg)),0.1),...
    'EquivDiameter','MajorAxisLength','MinorAxisLength');                   % Use regionprops to estimate pupil diameter
for k = 1:numel(pupilProps)
    if k == 1
        pupilDiam = pupilProps(k).EquivDiameter;                            % Store pupil diameter
    else
        pupilDiam = max(pupilDiam,pupilProps(k).EquivDiameter);             % Store maximum value to pupil diameter
    end
end
pixeltomm       = (2*nanmedian(storeEye(:,end)))/24;                        % Use Biometrics to get physical units of dilation   
approxPupilDia  = (pupilDiam/2)/pixeltomm;                                  % Physical approximate pupil diameter
dilation        = dilationRatio * approxPupilDia;                           % Dimensionalize for physical dilation
fid = fopen(fullfile(pathName,...
    [fileBase,'_timeseries_measurements.txt']),'w');                        % Open Text File
fprintf(fid, [ 'Time(s)' ' ' 'Dilation Ratio' ' ' ' Pupil Diameter ' ' '...
    'Unvalidated Velocity' ' ' 'Validated Velocity' '\n']);                 % Specify Headers
fprintf(fid, '%f %f %f %f %f \n', [tVect' dilationRatio(:)...
    dilation(:) dispS(:)  dispSVal(:)]');                                   % Print Results
fclose(fid);                                                                % Close Text File

constrictInd        = find(dilationRatio == min(dilationRatio(:)));
maxConstrictTime    = tVect(constrictInd);
[pks,locs]          = findpeaks(abs(socdiff(dispSVal(1:constrictInd),1,2)));
quant50ind          = find(pks >= median(pks));
onsetInd            = locs(quant50ind(1));                                    
onsetTime           = tVect(onsetInd);
recoveryInd         = find(dilationRatio(constrictInd:end) >=...
    0.75*abs(1-dilationRatio(constrictInd))+dilationRatio(constrictInd));
if isempty(recoveryInd)
    recoveryInd     = numel(tVect)-constrictInd;
    recoveryTime    = tVect(end);
else
    recoveryTime    = tVect(constrictInd+recoveryInd(1));
end
averageConstriction = trapz(dispSVal(onsetInd:constrictInd))/(constrictInd-onsetInd);
averageDilation     = trapz(dispSVal(constrictInd:recoveryInd(1)))/(constrictInd-recoveryInd(1));
fid = fopen(fullfile(pathName,...
    [fileBase,'_timeseries_parameters.txt']),'w');                          % Open Text File
fprintf(fid, [ 'Onset Time,s' ' ' 'Peak Time,s' ' '...
    'Max Ratio' ' ' ' Recovery Time,s' ' '...
    'Constriction Velocity' ' ' 'Dilation Velocity' '\n']);                   % Specify Headers
fprintf(fid,['%03.3f        %03.3f       %03.3f      %03.3f ',...
    '          %03.3f                %03.3f  \n'], [onsetTime maxConstrictTime ...
    (1-dilationRatio(constrictInd))*100 recoveryTime-maxConstrictTime ...
    averageConstriction(1) averageDilation(1)]');                           % Print Results
fclose(fid);                                                                % Close Text File
toc

figure(1); 
subplot(2,1,1)
plot(tVect,dispSVal,'LineWidth',2);
grid on;
axis([0 5 -3 3]);
set(gca,'LineWidth',2,'FontSize',18,'FontWeight','bold');
xlabel('Time, seconds','FontSize',18,'FontWeight','bold');
ylabel('Dilatation Velocity, ppf','FontSize',18,'FontWeight','bold');
subplot(2,1,2)
plot(tVect,dilationRatio*100,'LineWidth',2);
axis([0 5 20 150]);
grid on;
set(gca,'LineWidth',2,'FontSize',18,'FontWeight','bold');
xlabel('Time, seconds','FontSize',18,'FontWeight','bold');
ylabel('Dilatation, percent','FontSize',18,'FontWeight','bold');
set(gcf,'Position',[100 100 400 550],'Color',[1 1 1]);
export_fig(gcf,fullfile(pathName,[fileBase,'_curves.png']),'-a1','-r100');
%%
keyboard
%%
% catch
%     fid = fopen(fullfile(pathName,[fileBase,'_results.txt']),'w');  % Open Text File
%     fclose(fid);    % Close Text File
%     fid = fopen(fullfile(pathName,[fileBase,'_timeSeriesAnalysis.txt']),'w');  % Open Text File
%     fclose(fid);    % Close Text File
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%% TIME SERIES ANALYSIS END %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

function [dispX,dispY,dispS,iteration] =  statisticalRegister(refF,curF,...
    SpatialWindow,FMCWindow,height,width,MinRad,MaxRad,...
    xCart,yCart,xLP,yLP,err_thresh,iteration_thresh)
% FUNCTION FOR REGISTRATION OF SINGLE COLOR CHANNEL FROM IMAGE PAIR
err = inf; iteration= 1;                                                    % Initialize Error & Iterations
dispX = 0; dispY = 0; dispS = 0;                                            % Initialize displacements & scale
TRev = eye(3); TFor = eye(3);                                               % Initialize Image Transform Matrices
% Run iterative registration
while err >= err_thresh && iteration <= iteration_thresh
    % Reconstruct images based on transform matrices
    fr01    = tformImage(xCart,yCart,TFor,[height width],curF);
    fr02    = tformImage(xCart,yCart,TRev,[height width],refF);
    fr01(isnan(fr01)) = 0; fr02(isnan(fr02)) = 0;
    % Calculate FFTs for image pair
    FFT01   = fft2(SpatialWindow.*(fr01-mean(fr01(:))));
    FFT02   = fft2(SpatialWindow.*(fr02-mean(fr02(:))));
    % Run Fourier-Mellin Transform on FFTs
    FMT01   = interp2(fftshift(abs(FFT01)),xLP,yLP,'spline',0);
    FMT02   = interp2(fftshift(abs(FFT02)),xLP,yLP,'spline',0);
    % Calculate FFTs of FMTs
    FMC01   = fft2(FMCWindow.*(FMT01-mean(FMT01(:))));
    FMC02   = fft2(FMCWindow.*(FMT02-mean(FMT02(:))));
    % Perform displacement-based cross-correlation
    dispSpectral    = FFT01.*conj(FFT02);                                   % Do Standard Cross-Correlation
    [plX, plY]      = subPixel2D(fftshift(abs(ifft2(dispSpectral))));
    dispX           = dispX + plX;  
    dispY           = dispY + plY;
    % Perform scaling-based cross-correlation
    fmcSpectral     = FMC01.*conj(FMC02);                                   % Do Standard Cross-Correlation
    [fmcLocX, ~]    = subPixel2D(fftshift(abs(ifft2(fmcSpectral))));
    dispS           = dispS + fmcLocX;
    scale           = (MaxRad/MinRad)^(-dispS/width);
    % Update Error and Iterations
    err             = max(sqrt(plX.^2+plY.^2),abs(fmcLocX));
    iteration       = iteration + 1;
    % Update Transform Matrices
    TRev(1,1)       = sqrt(1/scale); TRev(2,2)       = sqrt(1/scale);
    TRev(1,3)       = -dispX/2;      TRev(2,3)       = -dispY/2;
    TFor(1,1)       = sqrt(1*scale); TFor(2,2)       = sqrt(1*scale);
    TFor(1,3)       =  dispX/2;      TFor(2,3)       =  dispY/2;
end
end

function [dispS,dispX,dispY] =  dilationEstimator(fr01F,fr02F,...
    xCart,yCart,xLP,yLP,SpatialWindow,FMCWindow,...
    MinRad,MaxRad,winsize,err_thresh,iteration_thresh)
%%%%%% PUPIL DILATION BLOCK START %%%%%
err   = Inf; iteration= 1;                                                  % Initialize Error & Iterations
TRev  = eye(3); TFor = eye(3);
dispX = 0; dispY = 0; dispS = 0;                                            % Initialize disp & scale
while err >= err_thresh && iteration <= iteration_thresh
    % Reconstruct images based on transform matrices
    fr01  = tformImage(xCart,yCart,TFor,[winsize winsize],fr01F);           % Register reference
    fr02  = tformImage(xCart,yCart,TRev,[winsize winsize],fr02F);           % Register reference
    fr01(isnan(fr01)) = 0; fr02(isnan(fr02)) = 0;
    % Calculate FFTs and perform image pair displacement cross-correlation
    FFT01           = fftshift(fft2(SpatialWindow.*(fr01-mean(fr01(:)))));
    FFT02           = fftshift(fft2(SpatialWindow.*(fr02-mean(fr02(:)))));
    dispSpectral    = FFT01.*conj(FFT02);
    [plX, plY]      = subPixel2D(fftshift(abs(ifft2(dispSpectral))));
    dispX = dispX + plX; dispY = dispY + plY;
    % Perform FMT on FFTs and scaling-based cross-correlation
    FMI01           = interp2(abs(FFT01),xLP,yLP,'spline',0);
    FMI02           = interp2(abs(FFT02),xLP,yLP,'spline',0);
    fmcSpectral     = fft2(FMCWindow.*(FMI01-mean(FMI01(:)))).*...
        conj(fft2(FMCWindow.*(FMI02-mean(FMI02(:)))));
    [fmcLocX, ~]    = subPixel2D(fftshift(abs(ifft2(fmcSpectral))));
    dispS           = dispS + fmcLocX;
    scale           = (MaxRad/MinRad)^(-dispS/winsize);
    % Update Error and Iterations
    iteration       = iteration + 1;
    err             = abs(fmcLocX);
    % Update Transform Matrices
    TRev(1,1)       = sqrt(1/scale); TRev(2,2)       = sqrt(1/scale);
    TRev(1,3)       = -dispX/2;      TRev(2,3)       = -dispY/2;
    TFor(1,1)       = sqrt(1*scale); TFor(2,2)       = sqrt(1*scale);
    TFor(1,3)       =  dispX/2;      TFor(2,3)       =  dispY/2;
end
end

function [XLP, YLP] = LogPolarCoordinates(IMAGESIZE, NUMWEDGES,...
    NUMRINGS, RMIN, RMAX, MAXANGLE)
% LOGPOLARCOORDINATES Constructs polar coordinate matrix values which map a
%   variable of interest from Cartesian space to polar space.
h       = IMAGESIZE(1);                                                     % Image Height
w       = IMAGESIZE(2);                                                     % Image Width
nw      = NUMWEDGES;                                                        % Number of Wedges
nr      = NUMRINGS;                                                         % Number of Rings
rMax    = RMAX;                                                             % Maximum Radius
rMin    = RMIN;                                                             % Minimum Radius
xZero   = (w + 1)/2;                                                        % X offset
yZero   = (h + 1)/2;                                                        % Y offset
logR    = linspace(log(rMin), log(rMax), nr);
rv      = exp(logR);
thMax   =  MAXANGLE * (1 - 1 / nw);
thv     = linspace(0, thMax, nw);
[r, th] = meshgrid(rv, thv);                                                % Build RHO-Theta Grids
[x, y]  = pol2cart(th, r);                                                  % Convert to Cartesian
XLP     = x + xZero;                                                        % Apply X offset
YLP     = y + yZero;                                                        % Aplly Y offset
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
    lPp1Y       = log( plane( lY + 1 , lX     ));
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