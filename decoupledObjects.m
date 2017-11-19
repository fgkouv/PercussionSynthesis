clear all; clc; close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% In this script, we will simulate two simple decoupled objects, a plate
% and a bar. When running the script, the user is asked to choose an
% excitation option, either a physical strike or a dry audio signal.
% Choosing the first one will render two organic percussive sounds while
% choosing the second one will produce two different audio effects. Plates
% usually add artificial reverberation while bars will produce a more
% 'metallic' effect to the input. (In case you might want to try this with
% some other input file please change the audioread line (line 47) to
% include your file of preference) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Excitation options and proper initialisation of global parameters 
disp('Please choose an exciation option.')
prompt = 'Choose 1 for a physical strike or 2 for a dry audio signal excitation! \n ';
option = input(prompt);

% If the user accidentally presses ENTER withour choosing, we will 
% process and play back the "organic" vibration
if isempty(option)
    option = 1;
end

while (option ~= 1) && (option ~= 2)
   prompt = 'Invalid input. Please choose 1 for a physical strike or 2 for a dry audio signal excitation.';
   option = input(prompt);
   if isempty(option)   % check for empty input again
    option = 1;
   end
end

if (option == 1)
     % Raised cosine
    SR = 44100;                     % sample rate 
    k = 1/SR;                       % time resolution
    TF = 5;                         % duration of simulation (s)
    NF = floor(SR*TF);              % duration of simulation (samples)
    RC = zeros(NF,1); 
    t0 = 50;    % starting sample of strike
    dur = 200;  % duration of strike
    f0 = 10;    % strike amplitude 
    RC(t0:t0+dur) =    f0/2 * (1-cos((pi/dur)*((t0:t0+dur)'-t0)));
else 
    % Dry audio excitation
    [RC,SR] = audioread('singing_anechoic_OpenAirLib.wav');
    if size(RC,2) == 2
        RC = 0.5* ( RC(:,1) + RC(:,2) ); % If stereo, convert to mono
    end
    k = 1/SR;
    NF = length(RC);
    TF = NF / SR;
end

%% PLATE
%%%%%%%%%

% Plate physical parameters
L_x = 0.6; L_y = 0.7;             % dimensions
x_0 = 0.2; y_0 = 0.43;            % striking points
N_t = 60;                         % tension
E = 180e10;                       % Young's modulus
volDens = 8050;                   % volumetric density
H = 0.001;                        % thickness
mu = volDens * H;                 % surface density
nu = 0.3;                         % Poisson's ratio 
D = (E*H^3) / (12*(1-nu^2));      % flexural rigidity
sigma0 = 0.01;                    % damping 1 (frequency-independent)

% Calculate maximum allowed number of modes
ALPHA =  (D / mu) * ((pi/L_x)^4 + (2*pi^4)./(L_x^2 * L_y^2) + (pi/L_y)^4);
BETA =  -(N_t / mu) * ((pi/L_x)^2 + (pi/L_y)^2); 
GAMMA = - 4 / (k^2);
DELTA = BETA^2 - 4*ALPHA*GAMMA;
MODES = floor(sqrt( (-BETA + sqrt(DELTA)) / (2*ALPHA)  )) - 1;
[m,n] = meshgrid((1:MODES),(1:MODES));
m = m(:); n = n(:);
NmP = MODES^2;

% Find expressions for the modal frequencies that we'll need for the
% frequency-dependent damping factors.
OMEGA_PLATES = sqrt(   (D /mu) * (    (m*pi/L_x).^2  +   (n*pi/L_y).^2).^2 + (N_t/mu) * (  (m*pi/L_x).^2 + (n*pi/L_y).^2) ) ;
FREQ_PLATES = OMEGA_PLATES / (2*pi);  % We need the frequencies in Hertz

% The alpha term will be used as a damping control for our set of plates
alpha = 0.0001;
% Find frequency-dependent coefficients
sigma1_p = alpha * FREQ_PLATES.^0.6;

% Define output options
x_s = 0.3; y_s = 0.1;  % output points

% Projection of modal shapes on output points
spatial = sin(m'*pi*x_s/L_x) .* sin(n'*pi*y_s/L_y);

% Set up discrete-time scheme
S = (   (pi^2  * m.^2 ) / (L_x^2)    +   (pi^2  * n.^2 ) / (L_y^2) ); 
% To be multiplied with current displacement vector
A = ( 2  -  k^2  *  ((N_t / mu) * S + (D/mu) * S.^2)  )  ./  ( 1 + k * (sigma0 + sigma1_p .* S  ) ) ;
% To be multiplied with previous displacement vector
B = ( - 1 + k * (sigma0 + sigma1_p .* S)   ) ./  ( 1 + k * (sigma0 + sigma1_p .* S  ) ) ;
% To be multiplied with current excitation value
C = (k^2 * ( 4 / (mu*L_x*L_y) ) * (  sin(m * pi * x_0 / L_x)  .* sin(n * pi * y_0 / L_y)))   ./  ( 1 + k * (sigma0 + sigma1_p .* S  ) ) ;

% Initialise and build discrete scheme
p1 = zeros(NmP,1); p2 = zeros(NmP,1); p3 = zeros(NmP,1);
out_p = zeros(NF,1);
for i = 3:NF
    
   p1 = A .* p2 + B .* p3 + C*RC(i);        % Temporal dispacement component
   out_p(i) =  spatial * p1;                % Inner product with modal shapes  
   % Update
   p3 = p2;         
   p2 = p1;
    
end

out_p = out_p / max(abs(out_p));


disp('Playing plate vibration...')
soundsc(diff(out_p),SR)

%% BAR
%%%%%%%%%
L_b = 1;            % bar length
T = 60;             % tension
E_b = 210e10;       % Young's modulus
radius =  0.9364e-2;                % cross-section radius
volDens = 6050;                     % volumetric density
lamda = volDens * (pi*radius^2);    % linear density = volumetricDensity * area
I = (pi*radius^4) / 4;              % moment of inertia
sigma0 = 0.06;                      % frequency-independent damping
x_0_b = 0.3;                        % striking point


% Find max number of modes
ALPHA_b = (E*I*pi^4) / (lamda*L_b^4);
BETA_b  =  (T*pi^2)/(lamda*L_b^2);
DELTA_b = BETA_b^2 - 4*ALPHA_b*GAMMA;
NmB     = floor(sqrt( (-BETA_b + sqrt(DELTA_b)) / (2*ALPHA_b)  )) -1;       % number of modes
m_b = (1:NmB)';

OMEGA_BARS =   sqrt(  (E*I/lamda) * (m_b*pi/L_b).^4 + (T/lamda) * (m_b*pi/L_b).^2 );
xi = ( (0.5*lamda)/(E*I) )*( -T/lamda + sqrt(  (T/lamda)^2 + (4/lamda)*E*I * OMEGA_BARS.^2 ));
% Similarly with the previous alpha, T60 will now be the damping control
T60 = 3; % in seconds
% and the frequency dependent damping factors will be:
sigma1_b = ((6 * log(10)) ./ (T60 * xi));

x_s_b = 0.57;   % output point
% Projection of modal shapes on output
spatial_b = sin(m_b'*pi*x_s_b/L_b); 

% Set up discrete-time scheme
B =    ( (pi  * m_b ) / L_b).^2 ; 
% To be multiplied with current displacement vector
D = ( 2  -  k^2  *  ( (T / lamda) * B + (E*I/lamda) * B.^2)  )  ./  ( 1 + k * (sigma0 + sigma1_b .* B  ) ) ;
% To be multiplied with previous displacement vector
E = ( - 1 + k * (sigma0 + sigma1_b .* B)   ) ./  ( 1 + k * (sigma0 + sigma1_b .* B) ) ;
% To be multiplied with current excitation value
F = (k^2 * ( 4 / (lamda*L_b) ) * (  sin(m_b * pi * x_0_b / L_b) ))   ./  ( 1 + k * (sigma0 + sigma1_b .* B  ) ) ;

% Initialise and build discrete scheme
b1 = zeros(NmB,1); b2 = zeros(NmB,1); b3 = zeros(NmB,1);
out_b = zeros(NF,1);

for i = 3:NF
    
   b1 = D .* b2 + E .* b3 + F*RC(i);        % Temporal dispacement component
   out_b(i) =  spatial_b * b1;              % Inner product with modal shape   
   
   % Update
   b3 = b2;         
   b2 = b1;
    
end

out_b = out_b / max(abs(out_b));

pause(TF)
disp('Playing bar vibration...')
soundsc(diff(out_b),SR)




%% Plotting part

subplot(2,1,1);
plot(diff(out_p)); title( strcat('Plate vibration. L_x = ', num2str(L_x), 'm , L_y =  ',num2str(L_y), 'm' ));
subplot(2,1,2);
plot(diff(out_b)); title( strcat('Bar vibration. L_b = ', num2str(L_b) , 'm'));





