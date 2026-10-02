% =========================================================
% Hybrid & Fuzzy Control Systems Assignment
% Implementation of Mamdani FIS and Sugeno (TSK) Inference
% =========================================================
clc; clear; close all;

choice = 3; 
% Select operating mode:
% choice = 1: Mamdani Fuzzy Controller & Rule Evaluation (Exercises 1-2)
% choice = 2: Batch Evaluation of Measured Inputs & Centroid Defuzzification (Exercise 3)
% choice = 3: Sugeno Fuzzy Inference System (TSK Model) & Scenario Analysis (Exercise 4)

% =========================================================
% CHOICE 1: Mamdani Fuzzy Controller (Exercises 1 & 2)
% =========================================================
if choice == 1
    
    % A) Environment Brightness
    brightness = 0:0.1:100;
    b_low = trapmf(brightness, [0 0 10 20]);   % Dark
    b_med = trimf(brightness, [15 40 65]);     % Medium
    b_high = trapmf(brightness, [60 80 100 100]); % High 

    figure; hold on; 
    plot(brightness, b_low, 'LineWidth', 2); 
    plot(brightness, b_med, 'LineWidth', 2);
    plot(brightness, b_high, 'LineWidth', 2);
    xlabel('Brightness'); ylabel('Membership Degree'); 
    title('Membership Functions for Environment Brightness'); 
    legend('Low', 'Medium', 'High'); grid on; 

    % B) Rate of Change of Brightness
    rateofchange = -10:0.1:10;
    rateofchange_low = trapmf(rateofchange, [-10 -10 -5 -1]);
    rateofchange_none = trimf(rateofchange, [-2 0 2]);
    rateofchange_high = trapmf(rateofchange, [1 5 10 10]);

    figure; hold on; 
    plot(rateofchange, rateofchange_low, 'LineWidth', 2); 
    plot(rateofchange, rateofchange_none, 'LineWidth', 2);
    plot(rateofchange, rateofchange_high, 'LineWidth', 2);
    xlabel('Rate of Change'); ylabel('Membership Degree'); 
    title('Membership Functions for Rate of Change'); 
    legend('Low', 'None', 'High'); grid on; 

    % C) Bulb Brightness (Output)
    bulb_brightness = 0:0.1:10;
    bulb_brightness_low = trapmf(bulb_brightness, [0 0 2 5]);
    bulb_brightness_med = trimf(bulb_brightness, [3 5 7]);
    bulb_brightness_high = trapmf(bulb_brightness, [5 8 10 10]);

    figure; hold on; 
    plot(bulb_brightness, bulb_brightness_low, 'LineWidth', 2); 
    plot(bulb_brightness, bulb_brightness_med, 'LineWidth', 2);
    plot(bulb_brightness, bulb_brightness_high, 'LineWidth', 2);
    xlabel('Bulb Brightness'); ylabel('Membership Degree'); 
    title('Membership Functions for Bulb Brightness'); 
    legend('Low', 'Medium', 'High'); grid on; 

    % Inputs and Firing Strengths
    b_in = trapmf(brightness, [10 30 40 60]);
    rateofchange_in = sigmf(rateofchange, [0.5, 5]);

    w_b_low = max(min(b_in, b_low));
    w_b_med = max(min(b_in, b_med));
    w_b_high = max(min(b_in, b_high));

    w_rateofchange_low = max(min(rateofchange_in, rateofchange_low));
    w_rateofchange_none = max(min(rateofchange_in, rateofchange_none));
    w_rateofchange_high = max(min(rateofchange_in, rateofchange_high));

    % Rule Base Evaluation
    Rule_1 = min(min(w_b_low, (w_rateofchange_low).^2), (bulb_brightness_high).^2);
    Rule_2 = min(min(w_b_low, w_rateofchange_none), (bulb_brightness_high).^0.5);
    Rule_3 = min(min(w_b_high, w_rateofchange_none), (bulb_brightness_low).^2);
    Rule_4 = min(min(w_b_med, w_rateofchange_none), bulb_brightness_med);
    Rule_5 = min(min(w_b_med, w_rateofchange_high), bulb_brightness_low);
    Rule_6 = min(min(w_b_med, w_rateofchange_low), min(bulb_brightness_med, (bulb_brightness_high).^0.5));

    % Aggregation using MAX operator
    F = max(max(max(Rule_1, Rule_2), max(Rule_3, Rule_4)), max(Rule_5, Rule_6));

    figure; hold on;
    plot(bulb_brightness, Rule_1, 'DisplayName', 'Rule 1');
    plot(bulb_brightness, Rule_2, 'DisplayName', 'Rule 2');
    plot(bulb_brightness, Rule_3, 'DisplayName', 'Rule 3');
    plot(bulb_brightness, Rule_4, 'DisplayName', 'Rule 4');
    plot(bulb_brightness, Rule_5, 'DisplayName', 'Rule 5');
    plot(bulb_brightness, Rule_6, 'DisplayName', 'Rule 6');
    plot(bulb_brightness, F, 'k--', 'LineWidth', 2, 'DisplayName', 'Aggregated Output');
    xlabel('Bulb Brightness'); ylabel('Membership Degree');
    title('6-Rule Fuzzy Controller Output');
    legend; grid on;

% =========================================================
% CHOICE 2: Batch Evaluation of Measured Inputs (Exercise 3)
% =========================================================
elseif choice == 2
    brightness = 0:0.1:100;
    b_low = trapmf(brightness, [0 0 10 20]); 
    b_med = trimf(brightness, [15 40 65]); 
    b_high = trapmf(brightness, [60 80 100 100]); 

    rateofchange = -10:0.1:10;
    rateofchange_low = trapmf(rateofchange, [-10 -10 -5 -1]);
    rateofchange_none = trimf(rateofchange, [-2 0 2]);
    rateofchange_high = trimf(rateofchange, [1 5 10 10]);

    bulb_brightness = 0:0.1:10;
    bulb_brightness_low = trapmf(bulb_brightness, [0 0 2 5]);
    bulb_brightness_med = trimf(bulb_brightness, [3 5 7]);
    bulb_brightness_high = trapmf(bulb_brightness, [5 8 10 10]);

    measured_brightness = [10, 13, 43.2, 53, 29, 33, 78, 90];
    measured_rateofchange = [-2, 3.2, 0, -3, 5, 4.3, -7.8, 9];
    
    for i = 1:length(measured_brightness)
        w_b_low = interp1(brightness, b_low, measured_brightness(i));
        w_b_med = interp1(brightness, b_med, measured_brightness(i));
        w_b_high = interp1(brightness, b_high, measured_brightness(i));

        w_rateofchange_low = interp1(rateofchange, rateofchange_low, measured_rateofchange(i));
        w_rateofchange_none = interp1(rateofchange, rateofchange_none, measured_rateofchange(i));
        w_rateofchange_high = interp1(rateofchange, rateofchange_high, measured_rateofchange(i));

        Rule_1 = min(min(w_b_low, (w_rateofchange_low).^2), (bulb_brightness_high).^2);
        Rule_2 = min(min(w_b_low, w_rateofchange_none), (bulb_brightness_high).^0.5);
        Rule_3 = min(min(w_b_high, w_rateofchange_none), (bulb_brightness_low).^2);
        Rule_4 = min(min(w_b_med, w_rateofchange_none), bulb_brightness_med);
        Rule_5 = min(min(w_b_med, w_rateofchange_high), bulb_brightness_low);
        Rule_6 = min(min(w_b_med, w_rateofchange_low), min(bulb_brightness_med, (bulb_brightness_high).^0.5));

        mf = max(max(max(Rule_1, Rule_2), max(Rule_3, Rule_4)), max(Rule_5, Rule_6));

        % Defuzzification using Centroid Method
        out(i) = sum(mf .* bulb_brightness) / sum(mf);
        fprintf('Measurement Set %d -> Crisp Bulb Brightness: %.2f\n', i, out(i));
    end

% =========================================================
% CHOICE 3: Sugeno Fuzzy Inference System (Exercise 4)
% =========================================================
elseif choice == 3
    x = linspace(0, 100, 1000); % Ground Humidity
    y = linspace(0, 40, 1000);  % Temperature
    [XX, YY] = meshgrid(x, y);

    % Membership Functions
    x_dry = trapmf(XX, [0 0 25 50]);
    x_med = trapmf(XX, [30 40 60 70]);
    x_wet = trapmf(XX, [50 85 100 100]);

    y_low = trapmf(YY, [0 0 15 20]);
    y_med = trapmf(YY, [18 22 25 29]);
    y_high = trapmf(YY, [27 32 40 40]);

    % Rule Weights
    wg1 = x_dry; wg3 = x_wet;
    wt1 = y_low; wt3 = y_high;

    % Sugeno Consequent Functions
    z1 = 0.3 * (100 - XX) + (0.2 * YY) + 2;
    z2 = 0.35 * (100 - XX) + 1;
    z3 = 0.1 * (100 - XX) - 0.1 * YY + 2;
    z4 = -0.2 * (100 - XX) - 0.2 * YY + 5;

    wr1 = wg1 .* wt3;
    wr2 = wg1 .* wt1;
    wr3 = wg3 .* wt1;
    wr4 = wg3.^2 + wt1.^2 - (wg3.^2) .* (wt1.^2);

    % Sugeno Output Aggregation
    Z = (wr1.*z1 + wr2.*z2 + wr3.*z3 + wr4.*z4) ./ (wr1 + wr2 + wr3 + wr4);

    figure;
    surf(XX, YY, Z, 'EdgeColor', 'none');
    colormap jet; shading interp;
    xlabel('Ground Humidity (%)'); ylabel('Temperature (°C)'); zlabel('Control Output z');
    title('Sugeno FIS Control Surface (General)');
    colorbar; grid on;

    % Scenario 1: Humidity = 25%, Temperature = 32°C
    g1 = trapmf(25, [0 0 25 50]);
    g3 = trapmf(25, [50 85 100 100]);
    t1 = trapmf(32, [0 0 15 20]);
    t3 = trapmf(32, [27 32 40 40]);

    r1 = g1 * t3; r2 = g1 * t1; r3 = g3 * t1;
    r4 = g1^2 + t1^2 - (g1^2)*(t1^2); % Adjusted for scalar scenario evaluation
    
    Znew = (r1*z1 + r2*z2 + r3*z3 + r4*z4) / (r1 + r2 + r3 + r4 + eps);

    figure;
    surf(XX, YY, Znew, 'EdgeColor', 'none');
    colormap jet; shading interp;
    xlabel('Ground Humidity (%)'); ylabel('Temperature (°C)'); zlabel('Control Output z');
    title('Sugeno FIS Scenario: Humidity = 25%, Temp = 32°C');
    colorbar; grid on;

    % Scenario 2: Humidity = 75%, Temperature = 14°C
    g1_75 = trapmf(75, [0 0 25 50]);
    g3_75 = trapmf(75, [50 85 100 100]);
    t1_14 = trapmf(14, [0 0 15 20]);
    t3_14 = trapmf(14, [27 32 40 40]);

    r1_b = g1_75 * t3_14; r2_b = g1_75 * t1_14; r3_b = g3_75 * t1_14;
    r4_b = g3_75^2 + t1_14^2 - (g3_75^2)*(t1_14^2);

    Znewnew = (r1_b*z1 + r2_b*z2 + r3_b*z3 + r4_b*z4) / (r1_b + r2_b + r3_b + r4_b + eps);

    figure;
    surf(XX, YY, Znewnew, 'EdgeColor', 'none');
    colormap jet; shading interp;
    xlabel('Ground Humidity (%)'); ylabel('Temperature (°C)'); zlabel('Control Output z');
    title('Sugeno FIS Scenario: Humidity = 75%, Temp = 14°C');
    colorbar; grid on;
end
