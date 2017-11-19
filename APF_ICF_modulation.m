clear all; clc; close all;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% In this script, we can examine how chains of simple, modulated filters
% can affect the spectrum of simple sinusoidal input wave. We will built
% two chains allpass filters and allpass fractional delay inverse comb
% filters that we'll modulate at the same rates. The length of the chain is
% going to be the same for both chains, starting with N = 10 filters at
% each case. This, alongside with the modulation depth M (originally at
% 0.4) should be considered the main controls of the script to vary the
% resulting sounds. Finally, plays back the last stages of both chains and
% plots the magnitude of the first and last filter in each chain.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Global parameters
SR = 44100;     % sampling rate
k = 1/SR;       % sampling period
TF = 1;         % simulation length (in seconds)
t = (0:k:TF-k); % time vector
NF = length(t); % number of total samples

%% Input and modulation settings
f_c = 440;              % Input frequency
x  = sin(2*pi*f_c*t);   % Input wave       

f_m = 100;              % Modulation rate (Hertz)
M = 0.4;                % Modulation depth
if ( abs(M) >=1 )       % Error check to ensure filter stability
    error('Error: Modulation depth is greater than 1. The filters will become unstable!');
end
m = M*sin(2*pi*f_m*t);  % Modulation wave

%% Choose chain length and set up main state matrices
N = 10;                         % Chain length
if ( (N < 0) || (abs(N)~=N) )   % Error check to ensure positive,integer length
    error('Error: Invalid choice of chain length. Please select a positive integer value');
end
out_APF = zeros(N,NF);          % We will keep track of the output of each stage 
out_ICF = zeros(N,NF);          % of the two chains 

I = eye(N);
bottom_diag = zeros(N,N);   % NxN matrix with ones on the lower diagonal
for i=2:N
    bottom_diag(i,i-1) = 1;     
end

% Inverse comb chain matrices time-independent matrix   
C = I - 0.5 * bottom_diag;

% Vector to be multiplied with different input samples
in_Vec = zeros(N,1); in_Vec(1) = 1;

%% Main process loop
for n = 3:NF  
    % APF
    A = I - m(n) * bottom_diag;                             % to be multiplied with with output stages at the current sample
    B = bottom_diag -  diag(  m(n) * ones(1,N) );           % to be multiplied with with output stages at (n-1) sample 
    out_APF(:,n) = (A \ B) * out_APF(:,n-1) +   m(n) * in_Vec * x(n) +  in_Vec * x(n-1);    
    
    % ICF
    D = diag( m(n) * ones(1,N) ) + m(n) * bottom_diag;      % to be multiplied with output stages at (n-1) sample
    E = -0.5 * m(n) * bottom_diag;                          % to be multiplied with output stages at (n-2) sample
    out_ICF(:,n) = (C \ D) * out_ICF(:,n-1) + (C \ E) * out_ICF(:,n-2) + 0.5 * in_Vec * x(n)  + m(n) * in_Vec * x(n-1) - 0.5*m(n) * in_Vec * x(n-2);
end

% Result playback
disp('Playing last output of allpass chain...')
soundsc(out_APF(N,:),SR);
pause(TF+0.2);
disp('Playing last output of inverse comb chain...')
soundsc(out_ICF(N,:),SR);


%% Plotting part

 NF = 2^nextpow2(NF);
 APF = zeros(N,NF);
 ICF = zeros(N,NF);
 
 for i=1:N
     APF(i,:) = fft(out_APF(i,:),NF);
     ICF(i,:) = fft(out_ICF(i,:),NF);
 end
  
fig1=figure(1);
set(fig1, 'Position', get(0,'Screensize'),'name',strcat('Magnitude spectrum of output signals. f_m = ',num2str(f_m),' Hz , M = ', num2str(M),' , chain length = ', num2str(N)));

ax(1)=subplot(2,2,1);
set(ax,'FontSize',10);
plot(abs(APF(1,:))); xlim([0 SR/2]);
title('Spectrum of output of the first filter in the chain of 10 APF')
grid on;

ax(2)=subplot(2,2,2);
set(ax,'FontSize',10);
plot(abs(APF(N,:)));  xlim([0 SR/2]);
title('Spectrum of output of the last filter in the chain of 10 APF');
grid on;

ax(3)=subplot(2,2,3);
set(ax,'FontSize',10);
plot(abs(ICF(1,:)));  xlim([0 SR/2]);
title('Spectrum of output of the first filter in the chain of 10 APFDICF')
grid on;

ax(4)=subplot(2,2,4);
set(ax,'FontSize',10);
plot(abs(ICF(N,:)));  xlim([0 SR/2]);
title('Spectrum of output of the last filter in the chain of 10 APFDICF');
grid on;


 





