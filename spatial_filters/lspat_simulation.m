%% comparing linear spatial filters
% manuscript currently under review at J Neuroscience Methods
% mikexcohen.com

% This script requires the following files to be in the MATLAB path:
%   - jader.m
%   - filterFGx.m
%   - ssd.m
%   - emptyEEG.mat

% the eeglab toolbox is used for topographical plotting

% frequencies to simulate/analyze. You really only need one frequency. The
% paper uses 60 steps between 2 and 80 Hz. Depending on your computer, this
% script takes a few tens of seconds to a few minutes per frequency.
frex = logspace(log10(2),log10(80),10);

% main filter parameters.
fwhm4filt = 2; % for defininig the spatial filter. Should be narrow.
fwhm4anal = 5; % after applying the spatial filter to broadband data. Should be less narrow.

%% preliminaries

addpath("C:\Users\yilma\OneDrive - TUM\Programming\MATLAB\TUM\Ingenieurspraxis");

% mat file containing EEG, leadfield and channel locations
load emptyEEG
origEEG = EEG;

% indices of dipole locations (best to leave these as specified)
dipoleLoc1 =  94;
dipoleLoc2 = 205;
whichOrientation = 1; % 1 for "EEG" and 2 for "MEG"

%% initializations

spatmaps = zeros(EEG.nbchan,length(frex),6,2);
[cordata,snrs] = deal( nan(length(frex),6,2) );
filtnames = cell(6,1);

tidx = dsearchn(EEG.times',[EEG.xmin+.5 EEG.xmax-.5]');

%% loop over frequencies

for fi=1:length(frex)
% for fi=length(frex):length(frex)
% for fi=1:1
    % fi
    
    % simulate EEG data
    
    EEG = origEEG;
    
    dip1freq = frex(fi);
    dip2freq = dip1freq+rand*5+1;
    
    % create data time series
    ampl1    = 10+10*filterFGx(randn(1,EEG.pnts),EEG.srate,3,10);
    freqmod1 = detrend(10*filterFGx(randn(1,EEG.pnts),EEG.srate,3,10));
    k1       = (dip1freq/EEG.srate)*2*pi/dip1freq;
    signal1  = ampl1 .* sin(2*pi.*dip1freq.*EEG.times + k1*cumsum(freqmod1));
    
    ampl2    = 10+10*filterFGx(randn(1,EEG.pnts),EEG.srate,3,10);
    freqmod2 = detrend(10*filterFGx(randn(1,EEG.pnts),EEG.srate,3,10));
    k2       = (dip2freq/EEG.srate)*2*pi/dip2freq;
    signal2  = ampl2 .* sin(2*pi.*dip2freq.*EEG.times + k2*cumsum(freqmod2));
    
    
    % create dipole data
    pspect = filterFGx( bsxfun(@times,rand(EEG.pnts,size(lf.Gain,3)),linspace(-1,1,EEG.pnts)'.^20)',EEG.srate,10,50)';
    data   = 100*real(ifft(complex(pspect,2*pi*rand(size(pspect))-pi)));
    data(:,dipoleLoc1) = signal1 + randn(1,EEG.pnts);
    data(:,dipoleLoc2) = signal2 + randn(1,EEG.pnts);
    
    % finally, the total simulated EEG data
    tmpdata   = ( data*squeeze(lf.Gain(:,whichOrientation,:))' )';
    EEG.data  = (tmpdata(:,tidx(1):tidx(2)));
    EEG.pnts  = size(EEG.data,2);
    EEG.times = EEG.times(tidx(1):tidx(2));
    
    signal1   = signal1(tidx(1):tidx(2));
   
    % possibly replace O1 with pure noise
    
    for noisei=1:1
        
        if noisei==2
            m = mean(EEG.data(strcmpi('o1',{EEG.chanlocs.labels}),:));
            v = var( EEG.data(strcmpi('o1',{EEG.chanlocs.labels}),:));
            EEG.data(strcmpi('o1',{EEG.chanlocs.labels}),:) = m + v*randn(1,EEG.pnts);
        end
        
        % extract data for covariance matrix
        filtdat = filterFGx( EEG.data,EEG.srate,frex(fi),fwhm4filt );
        filtcov = (filtdat*filtdat')/length(filtdat);
        bbcov   = (EEG.data*EEG.data')/EEG.pnts;

        % filtcov = load('filtcov').filtcov;
        % bbcov = load('bbcov').bbcov;
        % filtdat = load('filtdat').filtdat;
        % EEG.data = load('data').data;
        % signal1 = load('signal1').signal1;
        % 
        % find frequency indices
        hz      = linspace(0,EEG.srate,diff(tidx)+1);
        freqidx = dsearchn(hz',frex(fi));
        freqlow = dsearchn(hz',frex(fi)-5):dsearchn(hz',frex(fi)-1);
        freqhih = dsearchn(hz',frex(fi)+1):dsearchn(hz',frex(fi)+5);
        
        % best-electrode
        filtnum=1; filtnames{filtnum}='bestElec';

        % filter around frequency and compute average power
        epower = abs(hilbert(filtdat)).^2;

        % find maximum electrode
        [~,maxe] = max(mean(epower,2));

        % get topomap
        spatmaps(:,fi,filtnum,noisei) = mean(epower,2);

        % get time course and correlate with simulated signal
        tcdat = filterFGx(EEG.data(maxe,:),EEG.srate,frex(fi),fwhm4anal);
        cordata(fi,filtnum,noisei) = corr(tcdat(:),signal1(:))^2;

        % compute spectral snr
        f = abs(fft(EEG.data(maxe,:))/EEG.pnts).^2;
        snrs(fi,filtnum,noisei) = f(freqidx) / mean(f([freqlow freqhih]));

        % PCA
        filtnum=2; filtnames{filtnum}='PCA';

        % find the best component
        [pcavecs,evals] = eig( filtcov );
        pcdata = (filterFGx(EEG.data,EEG.srate,frex(fi),fwhm4anal)'*pcavecs)';
        fftpow = abs(fft(pcdata,[],2)).^2;
        [~,bestcomp] = max(fftpow(:,dsearchn(hz',frex(fi))));
        maps = pcavecs(:,bestcomp);

        % force positive sign
        [~,idx] = max(abs(maps));
        spatmaps(:,fi,filtnum,noisei) = maps * sign(maps(idx));

        cordata(fi,filtnum,noisei) = corr(pcdata(bestcomp,:)',signal1(:))^2;

        pcdata = EEG.data'*pcavecs(:,idx);
        f      = abs(fft(pcdata)/EEG.pnts).^2;
        snrs(fi,filtnum,noisei) = f(freqidx) / mean(f([freqlow freqhih]));

        % ICA
        filtnum=5; filtnames{filtnum}='ICA';
        % 
        % ics = jader(filterFGx(EEG.data,EEG.srate,frex(fi),fwhm4anal),30);
        % 
        % % find the best component
        % icdata = ics*filterFGx(EEG.data,EEG.srate,frex(fi),fwhm4anal);
        % fftpow = abs(fft(icdata,[],2)).^2;
        % hz = linspace(0,EEG.srate,length(icdata));
        % [~,bestcomp] = max(fftpow(:,dsearchn(hz',frex(fi))));
        % 
        % maps   = pinv(ics);
        % maps   = maps(:,bestcomp);
        % icdata = icdata(bestcomp,:);
        % 
        % % force positive sign
        % [~,idx] = max(abs(maps));
        % spatmaps(:,fi,filtnum,noisei) = maps * sign(maps(idx));
        % 
        % cordata(fi,filtnum,noisei) = corr(icdata(:),signal1(:))^2;
        % 
        % icdata = ics*EEG.data;
        % f      = abs(fft(icdata(bestcomp,:))/EEG.pnts).^2;
        % snrs(fi,filtnum,noisei) = f(freqidx) / mean(f([freqlow freqhih]));
        % 
        % % JDfilt
        filtnum=4; filtnames{filtnum}='JDfilt';

        % sphere the data
        [evecsO,evalsO] = eig( bbcov );
        spheredata = (EEG.data'*evecsO*sqrt(pinv(evalsO)))';

        % bias filter and covariance with data
        biasfilt = toeplitz(sin(2*pi*frex(fi)*EEG.times));
        zbar = biasfilt*spheredata';
        [evecsF,evalsF] = eig( (zbar'*zbar)/length(zbar) );

        % compute weights and map
        jdw    = evecsO * sqrt(pinv(evalsO)) * evecsF;
        jdmaps = pinv(jdw)';

        % check for sign-flip
        [~,maxcomp] = max(diag(evalsF));
        [~,idx]     = max(abs(jdmaps(:,maxcomp)));
        spatmaps(:,fi,filtnum,noisei) = jdmaps(:,maxcomp) * sign(jdmaps(idx,maxcomp));


        % apply spatial filter to raw data
        jddata = filterFGx(EEG.data,EEG.srate,frex(fi),fwhm4anal)'*jdw(:,maxcomp);

        cordata(fi,filtnum,noisei) = corr(jddata(:),signal1(:))^2;

        jddata = EEG.data'*jdw(:,maxcomp);
        f      = abs(fft(jddata)/EEG.pnts).^2;
        snrs(fi,filtnum,noisei) = f(freqidx) / mean(f([freqlow freqhih]));

        % GEDb
        filtnum=3; filtnames{filtnum}='GEDb';

        [gedvecs,evals] = eig( filtcov,bbcov );

        % find biggest component and get map
        [~,maxidx] = max(diag(evals));
        maps = filtcov * gedvecs / (gedvecs' * filtcov * gedvecs);
        % maps = filtcov * gedvecs ;
        maps = maps(:,maxidx);

        % apply spatial filter to gently temporally filtered data
        geddata = filterFGx(EEG.data,EEG.srate,frex(fi),fwhm4anal)'*gedvecs(:,maxidx);

        % force positive sign
        [~,idx] = max(abs(maps));
        spatmaps(:,fi,filtnum,noisei) = maps * sign(maps(idx));

        cordata(fi,filtnum,noisei) = corr(geddata(:),signal1(:))^2;

        geddata = EEG.data'*gedvecs(:,maxidx);
        f      = abs(fft(geddata)/EEG.pnts).^2;
        snrs(fi,filtnum,noisei) = f(freqidx) / mean(f([freqlow freqhih]));
        % 
        % % SSD
        filtnum=6; filtnames{filtnum}='SSD';

        % try
        %     [~,maps,~,~,ssddata] = ssd(EEG.data',[frex(fi)-2 frex(fi)+2; frex(fi)-4 frex(fi)+4; frex(fi)-3 frex(fi)+3],EEG.srate,2,[]);
        % 
        %     [~,idx] = max(abs(maps(:,1)));
        %     spatmaps(:,fi,filtnum,noisei) = maps(:,1) * sign(maps(idx,1));
        % 
        %     cordata(fi,filtnum,noisei) = corr(ssddata(:,1),signal1(:))^2;
        % 
        %     f = abs(fft(ssddata(:,1))/EEG.pnts).^2;
        %     snrs(fi,filtnum,noisei) = f(freqidx) / mean(f([freqlow freqhih]));
        % 
        % catch me
        % end
        
    end % end add noise
end % end frequency loop

%% inter-filter map correlations

allcors = zeros(numel(frex),2);
for fi=1:length(frex)
    for noisei=1:2
        allcs = nonzeros(triu(corr(squeeze(real(spatmaps(:,fi,:,noisei)))))).^2;
        allcors(fi,noisei) = mean( allcs(allcs<1 | isfinite(allcs)) );
    end
end

%%


%% now for some plotting

cord = get(groot,'DefaultAxesColorOrder');

freq2plot = frex([3 round(length(frex)/2) end]);
clim = [-8 8];

figure(1), clf
for i=1:2
    subplot(2,1,i)
    h = plot(repmat(frex',1,6),squeeze(cordata(:,:,i)),'s-','linew',2,'markersize',10,'markerface','w');
    legend(filtnames)
    set(gca,'ylim',[0 1],'xlim',[frex(1)-.1 frex(end)+1],'xscale','lo','xtick',round(10*logspace(log10(1),log10(length(frex)),20))/10)
    xlabel('Frequency (Hz)'), ylabel('Fit to signal (R^2)')
    
    for ii=1:6
        set(h(ii),'color',cord(ii,:),'MarkerFaceColor',cord(ii,:),'markersize',5)
    end
end

%%

figure(2), clf
subplot(212)
plot(frex,allcors,'-s','linew',1)
set(gca,'xlim',[frex(1)-.1 frex(end)+1],'xscale','lo','xtick',round(10*logspace(log10(1),log10(length(frex)),20))/10)
xlabel('Frequency (Hz)'), ylabel('Inter-map correlations')
legend({'no noise';'with noise'})

%%

figure(12), clf

frex2plot = dsearchn(frex',[3 9 20 40 70]');
% frex2plot = dsearchn(frex', [80]);

nfilts = 4;
for fi=1:length(frex2plot)
    for filti=1:nfilts
        subplot(length(frex2plot),nfilts,(fi-1)*nfilts + filti)
        topoplot(squeeze(real(spatmaps(:,frex2plot(fi),filti,1))),EEG.chanlocs,'plotrad',.65,'electrodes','off','numcontour',0,'shading','interp');
        title([ filtnames{filti} ': ' num2str(round(frex(frex2plot(fi)))) ' Hz' ])
    end
end

%%

figure(3), clf
for i=1:2
    subplot(2,1,i)
    h = plot(repmat(frex',1,6),squeeze(snrs(:,:,i)),'s-','linew',2,'markersize',10,'markerface','w');
    legend(filtnames(1:6))
    set(gca,'ylim',[0 1.3*max(reshape(snrs,1,[]))],'xlim',[frex(1)-.1 frex(end)+1],'xscale','lo','xtick',round(10*logspace(log10(1),log10(length(frex)),20))/10)
    
    for ii=1:6
        set(h(ii),'color',cord(ii,:),'MarkerFaceColor',cord(ii,:),'markersize',5)
    end
end
%%

figure(40), clf
subplot(211)
bar(squeeze(nanmean(cordata)))
set(gca,'xticklabel',filtnames)
ylabel('Fit to simulated signal (R^2)')
legend({'no noise';'O1 noise'})

subplot(212)
bar(squeeze(nanmean(snrs,1)))
set(gca,'xticklabel',filtnames)
ylabel('SNR')
legend({'no noise';'O1 noise'})

%% end.
