clear all; clc; close all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script implements our main generalised percussion environment.
% The user is asked to specify a desired number of overall plates and then
% the system proceeds to assing random numbers of bars between 0 and 10 to
% to attach to each one. All the physical parameters are generated
% randomnly inside some predefined "plausible" ranges. In order to try
% different effects, you could alter these ranges. WARNING! Extreme values
% (especially for the plate thicknesses and Young's modulus) can
% dramatically increase the number of modes and thus make the system quite
% slow. Nevertheless, it is worth waiting for the rendered result because
% some results can be quite surprising! Finally, the plate excitation 
% patterns are also created arbitrarily, within a specified block of
% options found below. The bars are all struck once at the beginning of the
% simulation to produce a more high-pitched result than the (usual) plate
% vibrations. After the main simulation process, the system produces two
% semi-equalised results (a sum for all the plates and a sum for all the
% bars) and plays them back. To visualise some of the resulting output
% signals, there are also some exemplary plots for the frequency content of
% the largest and smallest 1D and 2D objects of the setup.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Global parameters ================================================
SR = 44100;             % sample rate (Hz)
TF = 5;                 % duration of simulation (sec)
k = 1/SR;               % time resolution
NF = floor(SR*TF);      % duration of simulation (samples)

%% Input error check ================================================
prompt = 'Please specify the desired number of plates ';
NP = input(prompt); % number of plates
while (NP <= 1)
    prompt = 'Please select more than one plate...';
    NP = input(prompt); % number of plates
end

if (round(NP) ~= NP)
    warning('Non-integer number of modes. Input will be rounded to closest integer');
    NP = round(NP);
end

%% Initial settings: Generate number of bars and connection "maps" ======

% Generate all the possible pairs of plates
comb = nchoosek(NP,2);
% Keep a vector of the consecutive plate indexes
plateVec = (1:NP);
% Create a matrix containing the indexes of the plates at each pair
combVec= nchoosek(plateVec,2);

mask = zeros(NP,comb);  % Create the logic "mask" for plates interaction
for i=1:NP
    for j=1:comb
        if combVec(j,1) == i  
            mask(i,j) = 1;
        elseif combVec(j,2) == i
            mask(i,j) = -1; % taking the direction of the forces into consideration!
        end
     end
end

bars = round(10*rand(NP,1)); % how many bars is attached to each plate (0-10)?
NB = sum(bars);              % total number of bars
 
% Extend plates displacement matrix:
indxP = 1; interactionBP = zeros(NB,1);
for i=1:length(bars)
    interactionBP(indxP:indxP + bars(i) -1 ) = i;   % This will create a vector of length NB
    indxP = indxP + bars(i);                        % containing the indices of plates that interact with each string
end

mask2 = zeros(NP,NB);  % Create binary "mask" for plates-bars interactions
for i=1:NP
    for j=1:NB
        if interactionBP(j) == i  
            mask2(i,j) = 1;
        end
    end
end


%% Excitation signals ============================================
f0 = 1000;                           % forcing amplitude                         
RC = zeros(NF,1);        t0 = 1;   % starting sample of pluck
dur = 200;                         % duration of strike (in samples)
% Raised cosine signal
RC(t0:t0+dur) = f0/2 * (1-cos((pi/dur)*((t0:t0+dur)'- t0))); 

% Generate a pattern played on plates:
% Randomly create a vector of the possible striking options for each plate
RCP = zeros(NF,NP);
options = round(3*rand(NP,1)); 
for i = 1:NP
   switch options(i)
       case 0   % single strike at random instance
           shift = round((NF-dur-1)*rand(1,1) + 1);
           RCP(shift:shift+dur,i) = 0.5 * f0 * (1-cos((pi/dur)*((shift:shift+dur)'- shift)));
       case 1   % single strike at the start of simulation
           RCP(:,i) = RC;
       case 2   % equally spaced "roll" of 5 hits
           hits = 5;
           RCP(:,i) = (sawtooth(2*pi*hits*(0:1/NF:1-1/NF))+1);
       case 3   % equally spaced, uneven roll of 5 hits
           hits = 5;
           RCP(:,i) =(sawtooth(2*pi*hits*(0:1/NF:1-1/NF))+1);
           starts = round((0.5*NF-1)*rand(hits,1) + 1);  
           ends = starts + round((0.4*NF-1)*rand(hits,1) + 1);           
            for jj = 1 : hits
                RCP(starts(jj): ends(jj),i) = 0;
            end
   end   
end

RCB = zeros(NF,NB);
options_bars = round(3*rand(NB,1));
for i = 1:NB
   switch options_bars(i)
       case 0   % single strike at random instance
           shift = round((NF-1)*rand(1,1) + 1);
           RCB(shift:shift+dur,i) = 0.5 * f0 * (1-cos((pi/dur)*((shift:shift+dur)'- shift)));
       case 1   % single strike at the start of simulation
           RCB(:,i) = RC;
       case 2   % equally spaced "roll" of 10 hits
           hits = 8;
           RCB(:,i) = (sawtooth(2*pi*hits*(0:1/NF:1-1/NF))+1);
       case 3   % equally spaced "roll" of 20 hits with variable amplitudes
           hits = 5;
           RCP(:,i) =(sawtooth(2*pi*hits*(0:1/NF:1-1/NF))+1);
           starts = round((0.5*NF-1)*rand(hits,1) + 1);  
           ends = starts + round((0.4*NF-1)*rand(hits,1) + 1);           
            for jj = 1 : hits
                RCP(starts(jj): ends(jj),i) = 0;
            end
   end   
end


%%  Bars parameters ================================================

% The total number of bars can often be higher than 50. Thus, it is very
% impractical to specify the parameters of every single one. Instead, we
% define a plausible range for each property and generate random values
% inside these ranges.

kappaRange = [1,100];           % spring constants acceptable value range
lengthRange = [0.5,1.5];          % lengths acceptable value range              
tensionRange = [50,200];        % tension acceptable value range  
youngRange = [100e09,200e09];   % Young's modulus acceptable value range  
radiusRange = [0.1e-03,1e-03];  % bar radius acceptable value range 
densityRange = [3000,10000];    % density acceptable value range 

kappa_b = (kappaRange(2) - kappaRange(1)) * rand(1,NB) + kappaRange(1);         % non-linear springs between plates and strings
L_b = (lengthRange(2) - lengthRange(1)) * rand(1,NB) + lengthRange(1);          % bar lengths
T = (tensionRange(2) - tensionRange(1)) * rand(1,NB) + tensionRange(1);         % bar tensions
E =     (youngRange(2) - youngRange(1)) * rand(1,NB) + youngRange(1);           % Young's moduli
radius  =   (radiusRange(2) - radiusRange(1)) * rand(1,NB) + radiusRange(1);    % bar radiuses
volDens = (densityRange(2) - densityRange(1)) * rand(1,NB) + densityRange(1);   % bar volumetric densities 

lamda = volDens .* (pi*radius.^2);                      % linear densities = volumetricDensities * area
I = (pi*radius.^4) / 4;                                 % moments of inertia

% We need to find a way to ensure that all the striking points are going to be inside each bar:
pointsRange = [0.05, min(L_b)]; % output and striking points acceptable value range
output_bars = (pointsRange(2) - pointsRange(1)) * rand(1,NB) + pointsRange(1);   % output points
x_0 = (pointsRange(2) - pointsRange(1)) * rand(1,NB) + pointsRange(1);  % striking points 
x_b = (pointsRange(2) - pointsRange(1)) * rand(1,NB) + pointsRange(1);  % coupling with membrane points 


 
%%  Plates parameters ==============================================
nu = 0.3;                            % Poisson's ratio 
sigma0 = 0.01;                       % common loss factor (for every object)
kappa_p = (kappaRange(2) - kappaRange(1)) * rand(1,comb) + kappaRange(1);   % non-linear springs between plates 

dimensionsRange = [0.6,2];  % plate dimensions range
thickRange = [1e-3, 1e-1];  % thickness range
youngRange = [10e8,100e9];   
densityRange = [6000,9000]; 

L_x = (dimensionsRange(2) - dimensionsRange(1)) * rand(1,NP) + dimensionsRange(1);          % plates "widths"
L_y = (dimensionsRange(2) - dimensionsRange(1)) * rand(1,NP) + dimensionsRange(1);          % plates "heights"
N_t =   (tensionRange(2) - tensionRange(1)) * rand(1,NP) + tensionRange(1);                 % plates tensions
H =   (thickRange(2) - thickRange(1)) * rand(1,NP) + thickRange(1);                         % plates thicknesses
E_m = (youngRange(2) - youngRange(1)) * rand(1,NP) + youngRange(1);                         % plates Young's moduli
volDensM = (densityRange(2) - densityRange(1)) * rand(1,NP) + densityRange(1);              % plates volumetric densities

mu = volDensM .* H;                     % surface density
D = (E_m.*(H.^3)) / (12*(1-nu^2));      % flexural rigidity

% We are also going to extend some vectors containing some parameters, so 
% that every element is replicated as many times as each plate is connected 
% with other objects
L_x_ext = L_x(interactionBP);   % replicate dimensions for connections with bars
L_y_ext = L_y(interactionBP);
N_t_ext = N_t(interactionBP);   % replicate tensions for connections with bars
mu_ext = mu(interactionBP);     % replicate densities for connections with bars
D_ext = D(interactionBP);       % replicate rigidities for connections with bars

% Define coupling, striking and output points ranges
pointsX_Range = [0.05, min(L_x)];
pointsY_Range = [0.05, min(L_y)];

% Output points
output_membranesX = (pointsX_Range(2) - pointsX_Range(1)) * rand(1,NP) + pointsX_Range(1);
output_membranesY = (pointsY_Range(2) - pointsY_Range(1)) * rand(1,NP) + pointsY_Range(1);

x_0p = (pointsX_Range(2) - pointsX_Range(1)) * rand(1,NP) + pointsX_Range(1);          % striking points // x-axis
y_0p = (pointsY_Range(2) - pointsY_Range(1)) * rand(1,NP) + pointsY_Range(1);          % striking points // y-axis

x_p = (pointsX_Range(2) - pointsX_Range(1)) * rand(1,NB) + pointsX_Range(1);          % connection points with bars // x-axis
y_p = (pointsY_Range(2) - pointsY_Range(1)) * rand(1,NB) + pointsY_Range(1);          % connection points with bars // y-axis

% For the interactions between plates, we need to define coupling points
% for each plate contained in each pair
x_pp1 = (pointsX_Range(2) - pointsX_Range(1)) * rand(1,comb) + pointsX_Range(1);          % First plate of each pair connection points  // x-axis 
y_pp1 = (pointsY_Range(2) - pointsY_Range(1)) * rand(1,comb) + pointsY_Range(1);          % First plate of each pair connection points  // y-axis

x_pp2 = (pointsX_Range(2) - pointsX_Range(1)) * rand(1,comb) + pointsX_Range(1);          % Second plate of each pair connection points  // x-axis
y_pp2 = (pointsY_Range(2) - pointsY_Range(1)) * rand(1,comb) + pointsY_Range(1);          % Second plate of each pair connection points  // y-axis


%% Define number of modes ==============================================
% For each plate and each bar we will calculate the maximum allowed
% number of modes. Then, we will find the common numbers for the set of
% plates and the set of bars that allows the overall system to run without
% blowing up numerically

% PLATES
% We are assuming that the maximum happens at a "square" setting where the
% two mode indices (for x'x and y'y) are equal. To be sure that this does
% note jeopardise the stability, we will actually use one less than that
ALPHA = (D  ./ mu) .* ((pi./L_x).^4 + (2*pi^4)./(L_x.^2 .* L_y.^2) + (pi./L_y).^4);
BETA =  (N_t./ mu) .* ((pi./L_x).^2 + (pi./L_y).^2);
GAMMA = - 4 / (k^2);
DELTA = BETA.^2 - 4*ALPHA*GAMMA;
% From the set of modes for every plate, choose the minimum
MODES = min(floor(sqrt( (-BETA + sqrt(DELTA)) ./ (2*ALPHA)  ))) - 1;

% Build two vectors containing the evolution of the two mode indices for
% each dimension. Then, the total number of modes is actually the square of
% the maximum value of these indices.
[m,n] = meshgrid((1:MODES),(1:MODES));
m = m(:); n = n(:);
NmP = MODES^2;

% BARS
% In this 1D case, the maximum number of modes coincides with the maximum
% index, so we do not need to make any assumptions
ALPHA1 = (E.*I*pi^4) ./ (lamda .* L_b.^4);
BETA1  = (T  * pi^2) ./ (lamda .* L_b.^2);
GAMMA  = - 4 / (k^2);
DELTA1 = BETA1.^2 - 4*ALPHA1*GAMMA;
MODES1 = min(floor(sqrt( (-BETA1 + sqrt(DELTA1)) ./ (2*ALPHA1)  )));
NmB = MODES1;
m_b = (1:NmB)';


if (MODES < 1) ||(MODES1 < 1)      %~~ |Error check for extreme cases| ~~%
    error('Parameters chosen do not allow a modal physical model');
end                                %~~ |_____________________________| ~~%

%% Find the frequency-dependent loss terms. 
% For each object we need to calculate the modal frequencies, from which we
% will find the damping terms.

% PLATES
OMEGA_PLATES = zeros(NP,NmP);
for i = 1:NP
    % modal angular frequencies
    OMEGA_PLATES(i,:) = sqrt(   (D(i)/mu(i)) * (    (m*pi/L_x(i)).^2  +   (n*pi/L_y(i)).^2).^2 + (N_t(i)/mu(i)) * (  (m*pi/L_x(i)).^2 + (n*pi/L_y(i)).^2) ) ;
end
FREQ_PLATES = OMEGA_PLATES / (2*pi);  % We need the frequencies in Hertz
% The alpha term will be used as a damping control for our set of plates
alpha = 0.0015;
% Find freq. dependent coefficients
sigma1_p = alpha * FREQ_PLATES.^0.6;
% and extend them for the needed interactions
sigma1_p_ext = (sigma1_p(interactionBP,:))';


% BARS
OMEGA_BARS = zeros(NB,NmB);   xi = zeros(NB,NmB);
for i = 1:NB
    % modal angular frequencies
    OMEGA_BARS(i,:) =   sqrt(  (E(i)*I(i)/lamda(i)) * (m_b*pi/L_b(i)).^4 + (T(i)/lamda(i)) * (m_b*pi/L_b(i)).^2 );
    xi(i,:) = ( (0.5*lamda(i))/(E(i)*I(i)) )*( -T(i)/lamda(i) + sqrt(  (T(i)/lamda(i))^2 + (4/lamda(i))*E(i)*I(i) * OMEGA_BARS(i,:).^2 ));
end
% Similarly with the previous alpha, T60 will now be the damping control
T60 = 2; % in seconds
sigma1_b = ((6 * log(10)) ./ (T60 * xi))';



%%  Basic "coefficients". We will define some terms that appear multiple 
% times in certain operations.

% PLATES
Ap = ( (pi*m).^2 ) * ( ones(1,NP) ./ ((L_x).^2) )  +  ( (pi*n).^2 ) * ( ones(1,NP) ./ ((L_y).^2) );
% This is also useful in coupling with bars so let's "extend" it as previously
Ap_ext  = Ap(:,interactionBP); 
% Projection of the modal shapes on the coupling with bars points
Bp = sin( (pi*m) * (x_p ./ L_x_ext ) ) .* sin( (pi*n) * (y_p ./ L_y_ext ));

% BARS
Ab = ( (pi*m_b).^2 ) * ( ones(1,NB) ./ ((L_b).^2) );
Bb =  sin(pi*(m_b* (x_b./L_b))); % projections of modal shapes for interactions with plates
Bbf = sin(pi*(m_b* (x_0./L_b))); % projections of modal shapes at striking points

% Spatial terms: Projections of the modal shapes at output points
spatial_bars   =  sin(m_b*pi*(output_bars ./ L_b) );
spatial_plates =  sin(m*pi*(output_membranesX ./ L_x) ) .* sin(n*pi*(output_membranesY ./ L_y) );



%% Coupling schemes "coefficients"
% Here, we will define the coefficients than need to be multiplied with
% each term in the update schemes for the coupling terms. These emerged
% from the analysis AFTER we introduced the discrete time operators.

%~~~~~~~ Coupling bars with plates ~~~~~~~% 
% To be multiplied with current plates' displacements values
H_BP1 = ((-(k^2)/(1+sigma0*k)) * ((repmat(N_t_ext,[NmP,1]).*Ap_ext.*Bp) ./ repmat(mu_ext,[NmP,1]) + (repmat(D_ext,[NmP,1]).*(Ap_ext.^2).*Bp) ./ repmat(mu_ext,[NmP,1]) ))' + ((-(2*k)/(1+sigma0*k)) * (sigma1_p_ext  .* Ap_ext .* Bp))';    
% To be multiplied with previous plates' displacements values
H_BP2 = ((-(2*k)/(1+sigma0*k)) * ( sigma1_p_ext  .* Ap_ext .* Bp))';   
% To be multiplied with current bars' displacements values
H_BP3 = (((k^2)/(1+sigma0*k)) *  ((repmat(T,[NmB,1]) .* Ab .* Bb) ./ repmat(lamda,[NmB,1]) + (repmat(E.*I,[NmB,1]) .* (Ab.^2) .* Bb)./repmat(lamda,[NmB,1]) ))' + (((2*k)/(1+sigma0*k)) *  (sigma1_b .* Ab .* Bb))';               
% To be multiplied with previous bars' displacements values
H_BP4 = (((2*k)/(1+sigma0*k)) *  (sigma1_b .* Ab .* Bb))';      
% To be multiplied with current coupling values
H_BP5 = (   (2 - k^2 * kappa_b .* (  (ones(1,NB) ./ lamda ) + (ones(1,NB) ./ mu_ext ) ) ) / (1+sigma0*k))';                                                                    
% To be multiplied with previous coupling values
H_BP6 = (-1+sigma0*k) / (1+sigma0*k);                                                                                                                                          
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ %


%~~~~~~~ Coupling plates with plates ~~~~~~~% 
% First we need to extend every parameter vector because we are using 
% pairs of plates interacting with each other
N_t_intext1 = N_t(combVec(:,2)); N_t_intext2 = N_t(combVec(:,1));
D_intext1 = D(combVec(:,2)); D_intext2 = D(combVec(:,1));
mu_intext1 = mu(combVec(:,2)); mu_intext2 = mu(combVec(:,1));
L_x_intext1 = L_x(combVec(:,2)); L_x_intext2 = L_x(combVec(:,1)); 
L_y_intext1 = L_y(combVec(:,2)); L_y_intext2 = L_y(combVec(:,1)); 



sigma1_p_freq_inter_ext1 = sigma1_p(combVec(:,2),:); 
sigma1_p_freq_inter_ext2 = sigma1_p(combVec(:,1),:);
sigma1_p_freq_inter_ext1 = sigma1_p_freq_inter_ext1'; 
sigma1_p_freq_inter_ext2 = sigma1_p_freq_inter_ext2';

Ap1 = ( (pi*m).^2 ) * ( ones(1,comb) ./ ((L_x_intext1).^2) )  +  ( (pi*n).^2 ) * ( ones(1,comb) ./ ((L_y_intext1).^2) );
Bp1 = sin( (pi*m) * (x_pp1 ./ L_x_intext1 ) ) .* sin( (pi*n) * (y_pp1 ./ L_y_intext1 ) );

Ap2 = ( (pi*m).^2 ) * ( ones(1,comb) ./ ((L_x_intext2).^2) )  +  ( (pi*n).^2 ) * ( ones(1,comb) ./ ((L_y_intext2).^2) );
Bp2 = sin( (pi*m) * (x_pp2 ./ L_x_intext2 ) ) .* sin( (pi*n) * (y_pp2 ./ L_y_intext2 ) );

% Now we can define the coefficients of the interaction scheme
% To be multiplied with the current displacements values of the bottom plates
H_PP1 = ((-(k^2)/(1+sigma0*k)) * ( (repmat(N_t_intext1,[NmP,1]) .* Ap1 .* Bp1) ./ repmat(mu_intext1,[NmP,1]) + (repmat(D_intext1,[NmP,1]) .* (Ap1.^2) .* Bp1) ./ repmat(mu_intext1,[NmP,1]) ))'  + ((-(2*k)/(1+sigma0*k)) * (  sigma1_p_freq_inter_ext1  .* Ap1 .* Bp1 ))';    
% To be multiplied with the previous displacements values of the bottom plates
H_PP2 = ((-(2*k)/(1+sigma0*k)) * (  sigma1_p_freq_inter_ext1 .* Ap1 .* Bp1 ))'; 
% To be multiplied with the current displacement values of the top plates 
H_PP3 = (((k^2)/(1+sigma0*k))  * ( (repmat(N_t_intext2,[NmP,1]) .* Ap2 .* Bp2) ./ repmat(mu_intext1,[NmP,1]) + (repmat(D_intext2,[NmP,1]) .* (Ap2.^2) .* Bp2) ./ repmat(mu_intext2,[NmP,1]) ))'  + (((2*k)/(1+sigma0*k)) *  (  sigma1_p_freq_inter_ext2 .* Ap2 .* Bp2 ))' ;    
% To be multiplied with the previous displacements values of the top plates
H_PP4 = (((2*k)/(1+sigma0*k)) *  (  sigma1_p_freq_inter_ext2 .* Ap2 .* Bp2 ))'; 
% To be multiplied with the current coupling value
H_PP5 = ( (2 - k^2 * kappa_p .*  (  (ones(1,comb) ./ mu_intext1 ) + (ones(1,comb) ./ mu_intext2 ) ) )/ (1+sigma0*k))';     
% To be multiplied with the previous coupling value 
H_PP6 = (-1+sigma0*k) / (1+sigma0*k);   

%% Bar displacements schemes "coefficients"

% To be multiplied with the current displacements of bars
B1 = ( 2 - k^2 * ( ((pi*m_b).^2) * ( T ./ (lamda.*(L_b.^2) ) ) + ((pi*m_b).^4) * ( (E.*I)./(lamda.*(L_b.^4))) ) ) ./  (1 + k*(sigma0 +  sigma1_b .* (((pi*m_b).^2 ) * ( 1 ./ (L_b.^2) )) ) );  
% To be multiplied with the previous displacements of bars
B2 = (- 1 + k*(sigma0 +  sigma1_b .* (((pi*m_b).^2 ) * ( 1 ./ (L_b.^2) )) ) )   ./ (1 + k*(sigma0 +  sigma1_b .* (((pi*m_b).^2 ) * ( 1 ./ (L_b.^2) )) ) ); 
% To be multiplied with the current coupling term (with plates) 
B3 = ( repmat( (2*k^2*kappa_b)    ./ (lamda.*L_b) , [NmB,1]) .* Bb )     ./ (1 + k*(sigma0 +  sigma1_b .* (((pi*m_b).^2 ) * ( 1 ./ (L_b.^2) )) ) );                                            
% To be multiplied with the current strike value
B4 = ( repmat( (2*k^2*ones(1,NB))   ./ (lamda.*L_b) , [NmB,1]) .* Bbf )  ./ (1 + k*(sigma0 +  sigma1_b .* (((pi*m_b).^2 ) * ( 1 ./ (L_b.^2) )) ) );                                            



%% Plate displacements schemes "coefficients"

% First we will build an extended x_pp,y_pp matrix of dimensions(comb x NP) 
% as follows: At each column (representing every plate), we will
% insert the points where every connection happens. The rest of the rows
% are going to be left at zero.
extend_xpp = zeros(NP,comb); extend_ypp = zeros(NP,comb);
for i=1:NP
        for j=1:comb
            if combVec(j,1) == i  
                extend_xpp(i,j) = x_pp1(j);
                extend_ypp(i,j) = y_pp1(j);
            elseif combVec(j,2) == i
                extend_xpp(i,j) = x_pp2(j);
                extend_ypp(i,j) = y_pp2(j);
            end
        end
end
extend_xpp = extend_xpp'; extend_ypp = extend_ypp';


% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ %
% Inside the terms to be multiplied with the coupling terms, the sine
% functions "contain" a factor that depends on the modes (m), a factor that
% depends on the current plate (L_x) and another one that depends on the
% coupling relation (points extend_xpp / x_p). This will lead to every
% plate having two 2D matrices as coupling "coefficients". Instead of
% building a 3D matrix to continue our matricised approach, we'll work
% iteratively inside the main process loop. Now, we will build the repeated
% matrices as a struct of two "lists", one for interactions with other 
% plates and one for interactions with bars.
coef_list = [];
for i=1:NP
    a =  (4*k^2/mu(i)) * ((repmat(kappa_p,  [NmP,1]) .* sin((pi*m / L_x(i)) * extend_xpp(:,i)').* sin((pi*n / L_y(i)) * extend_ypp(:,i)')) );
    b =  (4*k^2/mu(i)) *   repmat(kappa_b,  [NmP,1]) .* sin((pi*m / L_x(i)) * x_p).* sin((pi*n / L_y(i)) * y_p) ;
    s = struct('coef_P',a,'coef_B',b);
    coef_list = [coef_list,s];
end 
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ %

% Now we can define the rest of the "coefficients" for the schemes
% calculating the plates' displacements.
% To be multiplied with the current displacement value of the plates
P1 = ( 2 - k^2 * (repmat( N_t ./ mu,[NmP,1]) .* Ap + repmat( D ./ mu,[NmP,1]) .* (Ap.^2)) ) ./ (1 + k*(sigma0 + sigma1_p' .* Ap ));                                           
% To be multiplied with the previous displacement value of the plates
P2 = ( -1 + k*(sigma0 + sigma1_p' .* Ap )) ./ (1 + k*(sigma0 + sigma1_p' .* Ap ));  
% To be multiplied with  the current striking term 
P3 = ( ( repmat( (4 * k^2)./ (mu .* L_x .* L_y),[NmP,1]) .* sin( (pi*m) * (x_0p ./ L_x ) ) .* sin( (pi*n) * (y_0p ./ L_y))) )./ (1 + k*(sigma0 + sigma1_p' .* Ap ));        
% All the couplings term coefficients need to divided by:
P4 = (1 + k*(sigma0 + sigma1_p' .* Ap)) ;                                                                                                                                  

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ~~~~~~~~~~~~~ MAIN PROCESSING LOOP ~~~~~~~~~~~~~         
% Initialise schemes : 
% 1 -> next time step // 2 -> current time step // 3 -> previous time step

% Plate displacements   ~~~~~~~~~~~~~
plt1 = zeros(NmP,NP);
plt2 = zeros(NmP,NP);
plt3 = zeros(NmP,NP);
% Bar displacements     ~~~~~~~~~~~~~
bars1 = zeros(NmB,NB);
bars2 = zeros(NmB,NB);
bars3 = zeros(NmB,NB);
% Bar/plate couplings   ~~~~~~~~~~~~~
interBP1 = zeros(NB,1);
interBP2 = zeros(NB,1);
interBP3 = zeros(NB,1);
% Plate/plate couplings ~~~~~~~~~~~~~
interPP1 = zeros(comb,1);
interPP2 = zeros(comb,1);
interPP3 = zeros(comb,1);
% Output signals        ~~~~~~~~~~~~~
out_plt = zeros(NP,NF);
out_bar = zeros(NB,NF);

for i=3:NF    
     % Bar-plate interactions:          ~~~~~~~~~~~~~~~~~~~~~~~~~~    
     % We need to find the current/previous displacement of the 
     % right platinteracting with each bar
     plates2 = plt2(:,interactionBP);   plates3 = plt3(:,interactionBP); 
     interBP1 = diag(H_BP1 * plates2 + H_BP2 * (-plates3) + H_BP3 * bars2 + H_BP4 * (-bars3)) + 1.2 * H_BP5 .* ((interBP2 .^ 5)) + H_BP6 .* interBP3;
          
     % Plate-plate interactions:        ~~~~~~~~~~~~~~~~~~~~~~~~~~
     % We need to find which ones are "working" in pairs at every instance
     bottom_cur  = plt2(:,combVec(:,2)); 
     bottom_prev = plt3(:,combVec(:,2)); 
     top_cur     = plt2(:,combVec(:,1)); 
     top_prev    = plt3(:,combVec(:,1));
     interPP1 =  diag(H_PP1 * bottom_cur + H_PP2 * (-bottom_prev) + H_PP3 * top_cur + H_PP4 * (-top_prev))  + 1.2 * H_PP5 .* ((interPP2 .^ 5)) + H_PP6 .* interPP3;
     
     % Next bar displacements            ~~~~~~~~~~~~~~~~~~~~~~~~~~
     bars1 = B1 .* bars2 + B2 .* bars3 + 0.33 * B3 .* repmat(((interBP1 .^ 3))',[NmB,1]) + B4 .* repmat(RCB(i,:),[NmB,1]) ;
    
     % Next plate displacements          ~~~~~~~~~~~~~~~~~~~~~~~~~~
     for j = 1:NP 
       plt1(:,j) = P1(:,j) .*  plt2(:,j) +  P2(:,j) .*  plt3(:,j) + P3(:,j) * RCP(i,j)  -  0.2 * (coef_list(j).coef_B * (mask2(j,:) * diag((interBP1 .^ 5)))')./ P4(:,j)  + 0.2 * (coef_list(j).coef_P * (mask(j,:) * diag((interPP1 .^ 5)))') ./ P4(:,j);
     end
     
     % Output samples. To produce output audio signals we take the inner
     % product of the temporal components with the projectios of our modal
     % shapes on the chose output points along each object
     out_plt(:,i) = diag(spatial_plates' * plt1 );
     out_bar(:,i) = diag(spatial_bars'   * bars1);
     
     % Updates                           ~~~~~~~~~~~~~~~~~~~~~~~~~~
     interBP3 = interBP2;   interBP2 = interBP1;
     interPP3 = interPP2;   interPP2 = interPP1;
     bars3 = bars2;         bars2 = bars1;
     plt3 = plt2;           plt2 = plt1;
        
end


%% Create a "parametric" mixer to produce two automated, semi-equalized results
% Plates        ~~~~~~~~~~~~~~~~~~~~~~~~~~
rmsP = zeros(1,NP);
for i=1:NP
    rmsP(i) = rms(out_plt(i,:));
end
% Find the output signal with the highest RMS value and create a scaling
% vector called vector based on the deviation of the rest of the RMS values
% from the maximum. The signals that have a significantly lower such value
% will be "boosted" more.
[val,ind] = max(abs(rmsP));
mixer = 1 - rmsP ./ val;
mixer(ind) = 0.3; % Set the scaling factor of the "loudest" signal to a low value
mix = zeros(1,NF-1);

for i=1:NP-1
    test = diff(out_plt(i,:));
    mix = mix + mixer(i) * (test/max(abs(test)));
end
mixP = mix / max(abs(mix));
disp('Playing a sum of the resulting plate vibrations...')
soundsc(mixP,SR);

% Bars          ~~~~~~~~~~~~~~~~~~~~~~~~~~
rmsB = zeros(1,NB);
for i=1:NB
    rmsB(i) = rms(out_bar(i,:));
end

[val,ind] = max(abs(rmsB));
mixer = 1 - rmsB ./ val;
mixer(ind) = 0.3;

mix = zeros(1,NF-1);
for i=1:NB-1
    test = diff(out_bar(i,:));
    mix = mix + mixer(i) * (test/max(abs(test)));
end
mixB = mix / max(abs(mix));
pause(TF+0.5);
disp('Playing a sum of the resulting bar vibrations...')
soundsc(mixB,SR);



%% Plotting part

% In this section, we will plot the frequency content of some exemplary
% results. Depending on the chosen setup, it might be impractical to plot
% all our results, so will plot characteristics of the biggest and 
% largest plates as well as the largest and smallest bars.
areas = L_x .* L_y;
[max1,maxPlate] = max(areas);
[min1,minPlate] = min(areas);
maxPlateVel = diff(out_plt(maxPlate,:));
minPlateVel = diff(out_plt(minPlate,:));

[max2,maxBar] = max(L_b);
[min2,minBar] = min(L_b);
maxBarVel = diff(out_bar(maxBar,:));
minBarVel = diff(out_bar(minBar,:));

NF = 2^nextpow2(NF);


fig1=figure(1);
set(fig1, 'Position', get(0,'Screensize'),'name','Frequency content of largest and smallest objects vibrations');
%~~~~~~~~~~~~~
subplot(2,2,1);
plot(20*log10(abs(fft(maxPlateVel,NF))),'Color',[0 0.1 0.5]);   xlim([0 SR/2]); title(strcat('Frequency content of the vibration of the largest plate.Area = ',num2str(areas(maxPlate)),'m^2'));
%~~~~~~~~~~~~~
subplot(2,2,2);
plot(20*log10(abs(fft(minPlateVel,NF))),'Color',[1 0.6 0.2]);   xlim([0 SR/2]); title(strcat('Frequency content of the vibration of the smallest plate.Area = ',num2str(areas(minPlate)),'m^2'));
%~~~~~~~~~~~~~
subplot(2,2,3);
plot(20*log10(abs(fft(maxBarVel,NF))),  'Color',[0.7 0.3 0.2]); xlim([0 SR/2]); title(strcat('Frequency content of the vibration of the longest bar. L_b = ',num2str(L_b(maxBar)),'m'));
%~~~~~~~~~~~~~
subplot(2,2,4);
plot(20*log10(abs(fft(minBarVel,NF))),  'Color',[0.6 0.7 0.5]); xlim([0 SR/2]); title(strcat('Frequency content of the vibration of the shortest bar.L_b = ',num2str(L_b(minBar)),'m'));








