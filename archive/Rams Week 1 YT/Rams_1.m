%% =========================================================================
%  tau_model_W1.m
%  Physics-Informed Reliability of AV Camera Perception Under Thermal Degradation
%  RAMS 2027 
%
%  AUTHOR : Yashitha 
%  PURPOSE: analytical functional form for the
%           time-resolved optical transmittance τ(t; ΔT, RH, surface).
%           This script defines the model SHAPE only (rise, plateau, decay).
%           Calibration against experimental data comes in Week 2.
%
%  PHYSICAL BASIS
%  --------------
%  Layer 1 – Heterogeneous nucleation onset (classical nucleation theory):
%    Condensation initiates when the enclosure window surface temperature
%    T_s drops below the dewpoint T_dp(RH, T_amb).  The induction
%    time t_ind before visible droplet nucleation is drawn from
%    classical CNT (Volmer-Weber, Fletcher 1958; Carey 1992):
%
%       J ~ exp( -ΔG* / kT )   where   ΔG* ∝ σ³ / (ΔT_sub)²
%
%    For engineering use we parameterise this as an effective onset lag
%    that depends on subcooling ΔT_sub = T_dp - T_s and contact angle θ_c.
%
%  Layer 2 – Droplet growth and optical attenuation (Beer-Lambert + Mie):
%    Once nuclei exist, condensate coverage fraction C(t) grows.
%    Optical transmittance τ is related to C via a generalised
%    Beer-Lambert / Mie scattering model:
%
%       τ(t) = exp( -α_eff · h(t) )
%
%    where h(t) is an effective optical depth proportional to C(t),
%    and α_eff encodes the scattering cross-section of the droplet
%    population (size-dependent; hydrophilic vs. hydrophobic surface).
%
%  Layer 3 – Recovery (evaporation / drainage):
%    When T_s rises back above T_dp (heater activation, solar gain,
%    convective warm-up), the condensate clears.  The recovery timescale
%    τ_rec is controlled by the latent heat of evaporation and the
%    boundary-layer vapour flux.
%
%  FUNCTIONAL FORM (shape-only, not yet calibrated)
%  -------------------------------------------------
%
%    C(t) = C_max * f_rise(t) * f_decay(t)
%
%  where:
%    f_rise(t)  = sigmoid rise after induction lag t_ind
%    f_decay(t) = exponential clearing after heater activation at t_heat
%
%  Full transmittance kernel:
%
%    τ(t; ΔT, RH, surface) = τ_min + (1 - τ_min) * [1 - C(t)]
%
%  so τ = 1 at t=0 (clean), τ = τ_min at full coverage (plateau),
%  and τ → 1 as surface clears.
%
%  =========================================================================

clear; clc; close all;

%% -------------------------------------------------------------------------
%  SECTION 1 – ENVIRONMENTAL PARAMETERS
%  (nominal values for shape sketch; calibration replaces these)
% --------------------------------------------------------------------------

T_amb   = 20;        % [°C]  ambient air temperature
RH      = 0.80;      % [–]   relative humidity (0–1)
T_s0    = 5;         % [°C]  initial surface temperature (cold transition)
T_s_rec = 25;        % [°C]  surface temperature after heater activation [°C]
t_heat  = 180;       % [s]   time at which heater activates (or sun warms surface)
t_end   = 600;       % [s]   total simulation window

%% -------------------------------------------------------------------------
%  SECTION 2 – DEWPOINT (Magnus approximation, ±0.4°C for 0–60°C)
%  Reference: O. A. Alduchov & R. E. Eskridge, J. Appl. Met., 1996
% --------------------------------------------------------------------------
% Magnus formula constants (August-Roche-Magnus):
a = 17.625;
b = 243.04;   % [°C]

T_dp = b * ( log(RH) + a*T_amb/(b + T_amb) ) / ...
           ( a - log(RH) - a*T_amb/(b + T_amb) );

fprintf('--- Environmental state ---\n');
fprintf('  T_amb  = %.1f °C\n', T_amb);
fprintf('  RH     = %.0f %%\n', RH*100);
fprintf('  T_dp   = %.2f °C  (Magnus approx.)\n', T_dp);

% Subcooling: positive means surface IS below dewpoint → condensation occurs
DeltaT_sub = T_dp - T_s0;
fprintf('  ΔT_sub = %.2f °C  (T_dp - T_s)\n', DeltaT_sub);

if DeltaT_sub <= 0
    warning('Surface is above dewpoint — no condensation expected.');
end

%% -------------------------------------------------------------------------
%  SECTION 3 – INDUCTION LAG  t_ind(ΔT_sub, θ_c)
%
%  Physical basis: classical heterogeneous nucleation (Fletcher 1958).
%  The nucleation rate J ∝ exp(-ΔG*/kT), with ΔG* ∝ f(θ_c)/ΔT_sub².
%  We parameterise t_ind as:
%
%       t_ind = k_ind * f(θ_c) / ΔT_sub^n_ind
%
%  where:
%    k_ind   = empirical coefficient [s·°C^n]  (calibrated in W2)
%    n_ind   = subcooling exponent              (literature: ~1.5–2.5)
%    f(θ_c)  = contact-angle weighting factor  (hydrophilic < hydrophobic)
%
%  Surface contact-angle presets (literature values):
%    untreated glass : θ_c ≈ 20°   → f = 0.6   (rapid nucleation)
%    hydrophilic coat: θ_c ≈ 10°   → f = 0.3   (very fast)
%    hydrophobic coat: θ_c ≈ 100°  → f = 2.5   (delayed)
%    superhydrophobic: θ_c ≈ 150°  → f = 5.0   (strongly delayed)
% --------------------------------------------------------------------------

k_ind  = 60;     % [s · °C^n_ind]  — placeholder; calibrated W2
n_ind  = 2.0;    % subcooling exponent
theta_c_deg = 20;               % contact angle [°] — untreated glass
f_theta = contact_angle_factor(theta_c_deg);   % dimensionless

t_ind = k_ind * f_theta / max(DeltaT_sub, 0.1)^n_ind;

fprintf('\n--- Induction ---\n');
fprintf('  θ_c    = %.0f°  →  f(θ_c) = %.2f\n', theta_c_deg, f_theta);
fprintf('  t_ind  = %.1f s  (%.1f min)\n', t_ind, t_ind/60);

%% -------------------------------------------------------------------------
%  SECTION 4 – COVERAGE FRACTION C(t) FUNCTIONAL FORM
%
%  RISE PHASE (t > t_ind, t < t_heat):
%    C_rise(t) = C_max * sigmoid( (t - t_ind - t_rise_half) / w_rise )
%
%    Sigmoid (logistic) form chosen because:
%      - Zero derivative at onset (nucleation lag)
%      - Monotone growth to plateau (physical — coverage saturates at C_max)
%      - Single parameter (w_rise) controls steepness; maps to ΔT_sub
%
%    w_rise = w0 / sqrt(ΔT_sub)     [faster rise at larger subcooling]
%
%  PLATEAU:
%    C(t) = C_max while t_heat has not been reached.
%    C_max = C_max_base * (1 - exp(-ΔT_sub / ΔT_char)) * g(RH)
%    g(RH) = RH^n_RH   (coverage scales with vapour availability)
%
%  DECAY PHASE (t > t_heat):
%    C_decay(t) = C(t_heat) * exp( -(t - t_heat) / tau_rec )
%
%    Exponential decay chosen because:
%      - Evaporation rate ∝ remaining condensate mass (linear ODE)
%      - tau_rec controlled by latent heat & boundary-layer vapour flux
%
%    tau_rec = tau_rec0 / (T_s_rec - T_dp)   [faster clearance above dewpoint]
% --------------------------------------------------------------------------

% --- Coverage plateau parameters (shape; not calibrated) ------------------
DeltaT_char   = 10;      % [°C]  characteristic subcooling scale
n_RH      = 1.5;     % RH exponent (literature: ~1–2)
C_max_base= 0.85;    % max possible coverage fraction [–]

C_max = C_max_base * (1 - exp(-DeltaT_sub / DeltaT_char)) * RH^n_RH;
C_max = min(C_max, 0.98);   % physical upper bound

fprintf('\n--- Coverage plateau ---\n');
fprintf('  C_max  = %.3f\n', C_max);

% --- Rise shape parameters ------------------------------------------------
w0        = 40;      % [s · √°C]  — placeholder; calibrated W2
w_rise    = w0 / sqrt(max(DeltaT_sub, 0.1));   % [s]
t_rise_half = 1.5 * w_rise;   % centre of sigmoid after t_ind [s]

% --- Recovery parameters --------------------------------------------------
tau_rec0  = 300;    % [s · °C]  — placeholder; calibrated W2
DeltaT_rec = T_s_rec - T_dp;
if DeltaT_rec <= 0
    error('Heater surface temp must exceed dewpoint for recovery to occur.');
end
tau_rec = tau_rec0 / DeltaT_rec;

fprintf('  w_rise = %.1f s\n', w_rise);
fprintf('  tau_rec= %.1f s (%.1f min)\n', tau_rec, tau_rec/60);

%% -------------------------------------------------------------------------
%  SECTION 5 – TIME VECTOR AND C(t) COMPUTATION
% --------------------------------------------------------------------------

t = linspace(0, t_end, 2000);   % [s]
C = zeros(size(t));

for i = 1:length(t)
    ti = t(i);
    if ti < t_ind
        % Before nucleation onset
        C(i) = 0;
    elseif ti < t_heat
        % Rise phase: sigmoid in coverage
        xi = (ti - t_ind - t_rise_half) / w_rise;
        C(i) = C_max / (1 + exp(-xi));
    else
        % Decay phase: exponential from coverage at t_heat
        xi_heat = (t_heat - t_ind - t_rise_half) / w_rise;
        C_at_heat = C_max / (1 + exp(-xi_heat));
        C(i) = C_at_heat * exp(-(ti - t_heat) / tau_rec);
    end
end

%% -------------------------------------------------------------------------
%  SECTION 6 – TRANSMITTANCE τ(t) FROM COVERAGE
%
%  Generalised Beer-Lambert / Mie scattering kernel:
%
%    τ(t) = exp( -α_eff · L_eff · C(t) )
%
%  where α_eff · L_eff = -log(τ_min) is the effective attenuation at
%  full coverage.  τ_min is surface- and droplet-size dependent.
%
%  For the shape sketch, τ_min is a parameter:
%    untreated glass, high ΔT:  τ_min ≈ 0.05 – 0.15  (severe)
%    hydrophilic coating:       τ_min ≈ 0.30 – 0.50  (moderate; thin film)
%    hydrophobic coating:       τ_min ≈ 0.10 – 0.30  (delayed but severe)
%
%  Reference: Mie theory for droplet radii r ≈ 1–50 μm; droplet radius
%  grows as r(t) ∝ t^(1/3) (diffusion-limited growth, Lifshitz-Slyozov).
%  Full Mie integration deferred to W2 calibration.
% --------------------------------------------------------------------------

tau_min = 0.10;   % minimum transmittance at full coverage [–]
alpha_eff = -log(tau_min);   % effective attenuation coefficient [–]

tau = exp(-alpha_eff .* C);

fprintf('\n--- Optical model ---\n');
fprintf('  τ_min        = %.2f  (full coverage)\n', tau_min);
fprintf('  α_eff · L    = %.3f\n', alpha_eff);

%% -------------------------------------------------------------------------
%  SECTION 7 – AVAILABILITY METRIC A(t)
%
%  Define a perception availability threshold τ_th.
%  A perception stack is "available" when τ(t) ≥ τ_th.
%  This maps τ(t) to a binary availability signal, and the
%  time-resolved availability function A(t) ∈ {0, 1}.
%  Blackout interval T_BO = total time with A(t) = 0.
%  This is the key reliability metric for the paper.
% --------------------------------------------------------------------------

tau_th = 0.50;    % perception availability threshold [–]
A = double(tau >= tau_th);   % availability: 1 = available, 0 = unavailable

% Compute blackout interval
dt = t(2) - t(1);
T_blackout = sum(A == 0) * dt;

fprintf('\n--- Availability ---\n');
fprintf('  τ_th         = %.2f\n', tau_th);
fprintf('  T_blackout   = %.1f s (%.1f min)\n', T_blackout, T_blackout/60);

%% -------------------------------------------------------------------------
%  SECTION 8 – SURFACE COMPARISON (shape sweep over θ_c)
% --------------------------------------------------------------------------

surfaces = struct( ...
    'name',  {'Untreated glass', 'Hydrophilic coat', 'Hydrophobic coat', 'Superhydrophobic'}, ...
    'theta', {20,                10,                  100,                150}, ...
    'tau_min',{0.08,             0.40,                0.12,               0.25});

colors = {'b','g','r','m'};
tau_matrix = zeros(length(surfaces), length(t));

for s = 1:length(surfaces)
    f_th = contact_angle_factor(surfaces(s).theta);
    t_ind_s = k_ind * f_th / max(DeltaT_sub, 0.1)^n_ind;
    w_rise_s = w0 / sqrt(max(DeltaT_sub, 0.1));
    t_rise_half_s = 1.5 * w_rise_s;
    
    C_s = zeros(size(t));
    for i = 1:length(t)
        ti = t(i);
        if ti < t_ind_s
            C_s(i) = 0;
        elseif ti < t_heat
            xi = (ti - t_ind_s - t_rise_half_s) / w_rise_s;
            C_s(i) = C_max / (1 + exp(-xi));
        else
            xi_heat = (t_heat - t_ind_s - t_rise_half_s) / w_rise_s;
            C_at_heat_s = C_max / (1 + exp(-xi_heat));
            C_s(i) = C_at_heat_s * exp(-(ti - t_heat) / tau_rec);
        end
    end
    
    alpha_s = -log(surfaces(s).tau_min);
    tau_matrix(s,:) = exp(-alpha_s .* C_s);
end

%% -------------------------------------------------------------------------
%  SECTION 9 – PLOTS
% --------------------------------------------------------------------------

fig1 = figure('Name','tau(t) Model — Shape Sketch','Position',[100 100 1200 800]);

% --- Panel 1: Coverage fraction C(t) ---
subplot(2,2,1);
plot(t/60, C, 'k', 'LineWidth', 2);
xline(t_ind/60,  '--r', 't_{ind}', 'LabelVerticalAlignment','bottom');
xline(t_heat/60, '--b', 't_{heat}', 'LabelVerticalAlignment','bottom');
xlabel('Time [min]'); ylabel('Coverage fraction C(t)');
title('Condensate Coverage C(t)');
ylim([0 1]); grid on; box on;
text(0.05, 0.9, sprintf('C_{max} = %.2f', C_max), 'Units','normalized');

% --- Panel 2: Transmittance τ(t) — nominal ---
subplot(2,2,2);
plot(t/60, tau, 'k', 'LineWidth', 2); hold on;
yline(tau_th, '--g', '\tau_{th}', 'LabelHorizontalAlignment','left');
yline(tau_min,'--r', '\tau_{min}', 'LabelHorizontalAlignment','left');
xline(t_ind/60,  '--r', 't_{ind}');
xline(t_heat/60, '--b', 't_{heat}');
xlabel('Time [min]'); ylabel('\tau(t)  [–]');
title('\tau(t) — Transmittance (nominal, untreated glass)');
ylim([0 1.05]); grid on; box on;

% --- Panel 3: Surface comparison ---
subplot(2,2,3);
hold on;
for s = 1:length(surfaces)
    plot(t/60, tau_matrix(s,:), 'Color', colors{s}, 'LineWidth', 1.8);
end
yline(tau_th, '--k', '\tau_{th}');
xline(t_heat/60, '--b', 't_{heat}');
legend({surfaces.name}, 'Location','southeast');
xlabel('Time [min]'); ylabel('\tau(t)  [–]');
title('\tau(t) — Surface Treatment Comparison');
ylim([0 1.05]); grid on; box on;

% --- Panel 4: Availability A(t) ---
subplot(2,2,4);
area(t/60, A, 'FaceColor',[0.2 0.7 0.2], 'FaceAlpha', 0.4, 'EdgeColor','none'); hold on;
plot(t/60, tau / max(tau), 'k--', 'LineWidth', 1.2);
xlabel('Time [min]'); ylabel('A(t)  [available = 1]');
title(sprintf('Availability A(t)   [T_{blackout} = %.0f s]', T_blackout));
ylim([-0.05 1.15]); grid on; box on;
legend({'Available', '\tau(t) / max(τ)'}, 'Location','southeast');

sgtitle(sprintf(['\\tau(t; \\DeltaT=%.0f°C, RH=%.0f%%, \\theta_c=%.0f°)\n' ...
    'Shape Sketch — Uncalibrated'], ...
    DeltaT_sub, RH*100, theta_c_deg), 'FontSize', 13, 'FontWeight','bold');

%% -------------------------------------------------------------------------
%  SECTION 10 – PARAMETER SENSITIVITY SKETCH (ΔT_sub sweep)
% --------------------------------------------------------------------------

fig2 = figure('Name','Sensitivity Sweep — DeltaT','Position',[150 150 900 500]);
DeltaT_vec = [3, 6, 10, 15, 20];   % [°C]
cols_sweep  = parula(length(DeltaT_vec));

subplot(1,2,1);  title('τ(t) vs ΔT_{sub}');  hold on; grid on; box on;
subplot(1,2,2);  title('τ(t) vs RH');         hold on; grid on; box on;

for k = 1:length(DeltaT_vec)
    dT = DeltaT_vec(k);
    [tau_sw, ~] = compute_tau(t, dT, RH, theta_c_deg, k_ind, n_ind, w0, ...
                              C_max_base, DeltaT_char, n_RH, tau_rec, t_heat, tau_min, t_rise_half);
    subplot(1,2,1);
    plot(t/60, tau_sw, 'Color', cols_sweep(k,:), 'LineWidth', 1.6, ...
        'DisplayName', sprintf('\\DeltaT_{sub} = %.0f°C', dT));
end
subplot(1,2,1);
yline(tau_th,'--k','τ_{th}'); legend('show','Location','southeast');
xlabel('Time [min]'); ylabel('\tau(t)'); ylim([0 1.05]);

RH_vec = [0.50, 0.65, 0.80, 0.90, 0.98];
cols_RH = cool(length(RH_vec));
for k = 1:length(RH_vec)
    rh = RH_vec(k);
    T_dp_k = b * (log(rh) + a*T_amb/(b+T_amb)) / (a - log(rh) - a*T_amb/(b+T_amb));
    dT_k   = T_dp_k - T_s0;
    if dT_k <= 0
        subplot(1,2,2);
        plot(t/60, ones(size(t)), 'Color', cols_RH(k,:), 'LineWidth', 1.6, ...
            'DisplayName', sprintf('RH = %.0f%% (no fog)', rh*100));
        continue
    end
    [tau_sw, ~] = compute_tau(t, dT_k, rh, theta_c_deg, k_ind, n_ind, w0, ...
                              C_max_base, DeltaT_char, n_RH, tau_rec, t_heat, tau_min, t_rise_half);
    subplot(1,2,2);
    plot(t/60, tau_sw, 'Color', cols_RH(k,:), 'LineWidth', 1.6, ...
        'DisplayName', sprintf('RH = %.0f%%', rh*100));
end
subplot(1,2,2);
yline(tau_th,'--k','τ_{th}'); legend('show','Location','southeast');
xlabel('Time [min]'); ylabel('\tau(t)'); ylim([0 1.05]);

sgtitle('Sensitivity Sweep (Shape Only, Uncalibrated)', ...
    'FontWeight','bold');

%% -------------------------------------------------------------------------
%  SECTION 11 – PRINT PARAMETER TABLE
% --------------------------------------------------------------------------

fprintf('\n=== PARAMETER TABLE (Shape sketch — calibration pending) ===\n');
fprintf('%-30s %-15s %-20s\n', 'Parameter', 'Symbol', 'Value / Form');
fprintf('%s\n', repmat('-',1,68));
fprintf('%-30s %-15s %-20s\n', 'Induction coefficient',   'k_ind',    sprintf('%.0f  s·°C^n [placeholder]', k_ind));
fprintf('%-30s %-15s %-20s\n', 'Subcooling exponent',     'n_ind',    sprintf('%.1f  [lit: 1.5–2.5]', n_ind));
fprintf('%-30s %-15s %-20s\n', 'Rise width coefficient',  'w0',       sprintf('%.0f  s·√°C [placeholder]', w0));
fprintf('%-30s %-15s %-20s\n', 'Char. subcooling scale',  'ΔT_char',  sprintf('%.0f  °C', DeltaT_char));
fprintf('%-30s %-15s %-20s\n', 'RH exponent',             'n_RH',     sprintf('%.1f  [lit: 1–2]', n_RH));
fprintf('%-30s %-15s %-20s\n', 'Max base coverage',       'C_max0',   sprintf('%.2f', C_max_base));
fprintf('%-30s %-15s %-20s\n', 'Recovery time coeff',     'τ_rec0',   sprintf('%.0f  s·°C [placeholder]', tau_rec0));
fprintf('%-30s %-15s %-20s\n', 'Min transmittance',       'τ_min',    sprintf('%.2f  [surface-dep.]', tau_min));
fprintf('%-30s %-15s %-20s\n', 'Avail. threshold',        'τ_th',     sprintf('%.2f  [TBD from L3]', tau_th));
fprintf('=================================================================\n');


%% =========================================================================
%  LOCAL FUNCTIONS
%% =========================================================================

function f = contact_angle_factor(theta_deg)
% contact_angle_factor  Nucleation rate weighting factor from CNT.
%
%  Classical heterogeneous nucleation (Fletcher 1958):
%    f(θ) = (2 + cosθ)(1 - cosθ)² / 4
%  Ranges from 0 (θ=0°, complete wetting, instant nucleation)
%  to 1 (θ=180°, non-wetting, nucleation as hard as homogeneous).
%  We use 1/f(θ) as the induction lag multiplier (larger contact
%  angle → slower nucleation → longer t_ind).

    theta_rad = theta_deg * pi / 180;
    f_theta   = (2 + cos(theta_rad)) .* (1 - cos(theta_rad)).^2 / 4;
    % Invert so larger θ → larger t_ind
    f = 1 ./ max(f_theta, 1e-4);
    % Normalise to f(20°) = 1.0 (untreated glass baseline)
    f_ref = 1 / ( (2 + cos(20*pi/180)) * (1 - cos(20*pi/180))^2 / 4 );
    f = f / f_ref;
end

function [tau_out, C_out] = compute_tau(t, dT_sub, RH_in, theta_deg, ...
    k_ind, n_ind, w0, C_max_base, DT_char, n_RH, tau_rec, t_heat, tau_min, ~)
% compute_tau  Evaluate τ(t) for given environmental parameters.
%   Used for sensitivity sweeps.

    f_th   = contact_angle_factor(theta_deg);
    t_ind_s = k_ind * f_th / max(dT_sub, 0.1)^n_ind;
    w_rise_s = w0 / sqrt(max(dT_sub, 0.1));
    t_rise_half_s = 1.5 * w_rise_s;

    C_max_s = C_max_base * (1 - exp(-dT_sub/DT_char)) * RH_in^n_RH;
    C_max_s = min(C_max_s, 0.98);

    C_out = zeros(size(t));
    for i = 1:length(t)
        ti = t(i);
        if ti < t_ind_s
            C_out(i) = 0;
        elseif ti < t_heat
            xi = (ti - t_ind_s - t_rise_half_s) / w_rise_s;
            C_out(i) = C_max_s / (1 + exp(-xi));
        else
            xi_heat = (t_heat - t_ind_s - t_rise_half_s) / w_rise_s;
            C_at_heat_s = C_max_s / (1 + exp(-xi_heat));
            C_out(i) = C_at_heat_s * exp(-(ti - t_heat) / tau_rec);
        end
    end

    alpha_s = -log(tau_min);
    tau_out = exp(-alpha_s .* C_out);
end