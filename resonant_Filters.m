clear all; clc; close all;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% In this script, we will examine the two analysed techniques of working
% with resonant filters for decaying harmonic or inharmonic tones.
% First, we will build a Mathews-Smith resonant filter with a selected
% frequency and decay time and listen to the output. Then, we will
% implement the experiments with a Scott van Duyne pattent that has been
% analysed in the dissertation. This second system will produce two
% results: An oscillator and a decaying, inharmonic tone.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Global parameters
SR = 44100;     % sampling rate
k = 1/SR;       % sampling period
TF = 1;         % simulation length (in seconds)
t = (0:k:TF-k); % time vector
NF = length(t); % number of total samples

u = zeros(1,NF); u(1) = 1;  % Input impulse


%% Max Mathews - Julius Smith inspired resonant filter

f = 100;                    % Resonant frequency
tau = 0.2;                  % Decay time
x = exp(-1/(tau*SR)) * cos(2*pi*f*k);   % Coefficient 1
y = exp(-1/(tau*SR)) * sin(2*pi*f*k);   % Coefficient 2
out_MS = zeros(1,NF);

for n = 3:NF
  out_MS(n) = u(n-1) - x * u(n-2) + 2 * x * out_MS(n-1) - (x^2 + y^2) * out_MS(n-2);
end

out_MS = out_MS / max(abs(out_MS));
soundsc(out_MS,SR);
disp('Playing Mathews-Smith resonant filter output...');


%% Based on Scott Van Duyne's pattern, we will implement two systems.
% The first one is simply the superposition of the closed loops
% in the original topology if we use them uncoupled. Since each uncoupled 
% branch can operate as a "lossless" resonant filter, this system will 
% basically produce inharmonic waveforms, much like additive synthesis,
% using only one impulse input. For the second system, we will introduce
% some signal attenuation by replacing the coupling filter with an
% averaging operation which will produce inharmonic decaying tones. In both
% approaches we'll use N total branches, initialised at N = 5.Instead of 
% specifying all the desired resonant frequencies, which would impractical
% for a larger number of branches, we'll define the frequency range of the
% desired outcome and generate the appropriate coefficients randomly in
% this range.

MIN_FREQ = 100;
MAX_FREQ = 500;

N = 5;      % Number of branches

frequencies = ((MAX_FREQ-MIN_FREQ)*rand(N,1) + MIN_FREQ);  % N desired resonant frequencies
A = - cos(2*pi*frequencies*k);              % Allpass filter coefficient
x = u;      % This will be used as the updated input to the decaying system

% Overall results
out_VD_oscillating  = zeros(1,NF); 
out_VD_decaying     = zeros(1,NF); 

% Let's also define the output of each branch at each instance
Y_osc = zeros(N,NF);    Y_dec = zeros(N,NF); 
Y_osc(:,1) = 1;         Y_dec(:,1) = 1; 

for n = 3:NF
    % Oscillating result. The output is the supeposition of all the tones
    Y_osc(:,n) = (1 + A) * u(n-2) - 2 * (A.*Y_osc(:,n-1)) - Y_osc(:,n-2);
    out_VD_oscillating(n) = sum(Y_osc(:,n)) / N;
    
    % Decaying result. The output is the supeposition of all the tones and
    % we need to update the input to be used in next instances
    Y_dec(:,n) = (1 + A) * x(n-2) - 2 * (A.*Y_dec(:,n-1)) - Y_dec(:,n-2); 
    out_VD_decaying(n) = sum(Y_dec(:,n)) / N;
    x(n)  = sum(Y_dec(:,n)) / N;
    
end

%% Result playback
pause(0.5); 
soundsc(out_VD_decaying,SR);
disp('Playing Van Duyne decaying result...')
pause(1);
soundsc(out_VD_oscillating,SR);
disp('Playing Van Duyne lossless result...')

%% Plotting part

fig1=figure(1);
set(fig1, 'Position', get(0,'Screensize'),'name','Results of experiments with resonant filters');
ax(1)=subplot(3,1,1);
plot(out_MS)
title(strcat('Output of Mathews-Smith resonant filter, f = ',num2str(f),' Hz , \tau = ',num2str(tau),'s'));
grid on;
%%%
ax(2)=subplot(3,1,2);
plot(out_VD_decaying)
title( strcat('Output of Van-Duyne inspired decaying inharmonic tone. Minimum f = ',num2str(MIN_FREQ),'Hz , Maximum f = ',num2str(MAX_FREQ),'Hz'));
grid on;
%%%
ax(3)=subplot(3,1,3);
plot(out_VD_oscillating)
title( strcat('Output of Van-Duyne inspired losless wave. Minimum f = ',num2str(MIN_FREQ),'Hz , Maximum f = ',num2str(MAX_FREQ),'Hz'));
grid on;






