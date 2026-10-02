% =========================================================
% Hybrid Systems: Custom Adam Optimizer & ANFIS (3-bit XOR)
% =========================================================
clc; clear; close all;

% =========================================================
% PART 1: Custom Adam Optimizer Implementation for Neural Networks
% =========================================================
% Note: The following section demonstrates the core logic of the 
% Adam optimizer applied to gradient-based network training (Weights W).

% Dummy initialization for demonstration structure:
% W = {rand(4,3), rand(1,4)}; 
% trainX = ...; trainY = ...; epochs = 50; learningRate = 0.01;
% error_vec = zeros(1, epochs); accuracy_vec = zeros(1, epochs);

% Initialization of Adam moment vectors (m for mean, v for uncentered variance)
% m = {zeros(size(W{1})), zeros(size(W{2}))}; 
% v = {zeros(size(W{1})), zeros(size(W{2}))}; 
% t = 1;

% Training loop structure with Adam
% for ep = 1:epochs
%     [error, out, gradients] = forward_back_propagation(W, trainX, trainY);
%     
%     % Apply Adam optimization step for each layer weight
%     [adam_grad1, m{1}, v{1}] = adam_optimizer(gradients{1}, m{1}, v{1}, t);
%     [adam_grad2, m{2}, v{2}] = adam_optimizer(gradients{2}, m{2}, v{2}, t);
%     t = t + 1;
%     
%     % Update weights
%     W{1} = W{1} - learningRate * adam_grad1;
%     W{2} = W{2} - learningRate * adam_grad2;
%     
%     error_vec(ep) = error;
%     
%     % Compute training accuracy
%     [~, trainPredictions] = max(out);
%     [~, trainLabels] = max(trainY);
%     trainAccuracy = mean(trainPredictions == trainLabels) * 100;
%     accuracy_vec(ep) = trainAccuracy;
% end


% =========================================================
% PART 2: ANFIS Modeling for 3-bit XOR Problem
% =========================================================
clc; clear; close all;

% 1. Define Dataset for 3-bit XOR Problem
X = [0 0 0; 
     0 0 1; 
     0 1 0; 
     0 1 1; 
     1 0 0; 
     1 0 1; 
     1 1 0; 
     1 1 1];

Y = [0; 1; 1; 0; 1; 0; 0; 1];

% 2. FIS Generation using Grid Partitioning
opt = genfisOptions('GridPartition');
opt.NumMembershipFunctions = [2 2 2]; % 2 Gaussian MFs for each of the 3 inputs
opt.InputMembershipFunctionType = ["gaussmf" "gaussmf" "gaussmf"]; 
opt.OutputMembershipFunctionType = "constant"; % Sugeno-type fuzzy system (Zero-order)
fis = genfis(X, Y, opt);

% 3. Train the ANFIS Model
trainData = [X, Y];
anfis_opt = anfisOptions('InitialFIS', fis, 'EpochNumber', 100, 'DisplayErrorValue', 1);
[trainFIS, trainFISError] = anfis(trainData, anfis_opt);

% 4. Visualize Training Error Convergence
figure('Name', 'ANFIS Training Error', 'Position', [100, 100, 600, 400]);
plot(trainFISError, 'LineWidth', 2, 'Color', 'b');
title('ANFIS Training Error Convergence (3-bit XOR)');
xlabel('Epochs');
ylabel('Root Mean Squared Error (RMSE)');
grid on;

% 5. Evaluate the Trained Neuro-Fuzzy Model
predictedOutput = evalfis(trainFIS, X);
disp('--- ANFIS 3-bit XOR Evaluation ---');
disp('Actual Output (Y) vs Predicted Output (ANFIS):');
disp([Y, predictedOutput]);


% =========================================================
% HELPER FUNCTIONS
% =========================================================
function [adam_grad, m, v] = adam_optimizer(grad, m, v, t)
    % ADAM_OPTIMIZER Computes adaptive learning rates using first and second moments.
    beta1 = 0.9;
    beta2 = 0.999;
    epsilon = 1e-8;
    
    % Update biased first and second moment estimates
    m = beta1 * m + (1 - beta1) * grad;
    v = beta2 * v + (1 - beta2) * (grad .^ 2);
    
    % Compute bias-corrected estimates
    m_hat = m / (1 - beta1^t);
    v_hat = v / (1 - beta2^t);
    
    % Compute final adaptive gradient step
    adam_grad = m_hat ./ (sqrt(v_hat) + epsilon);
end
