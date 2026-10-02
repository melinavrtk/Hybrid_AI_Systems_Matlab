% =========================================================
% Hybrid Systems Assignment: Regression & Optimization
% Dataset: Boston Housing (Polynomial Regression, GD, Adam, GA, SA)
% =========================================================
clc; clear; close all;

choice = 2; % Select Exercise / Operating Mode (1 to 4)

% =========================================================
% EXERCISE 1: Polynomial Regression, Regularization & Ridge
% =========================================================
if choice == 1
    
    data = readtable('BostonHousing.csv');    
    dataMatrix = table2array(data); 
    N = size(dataMatrix, 1);
    rand_per = randperm(N); 

    X = dataMatrix(rand_per, 6);       % Feature: Number of rooms
    Y = dataMatrix(rand_per, end-1);   % Target: House prices      

    % 1. Train/Test Split (70% - 30%)
    tr = 0.7; 
    ts = floor(tr * length(Y)); 
    Xtrain = X(1:ts);
    Ytrain = Y(1:ts);
    Xtest = X(ts+1:end);
    Ytest = Y(ts+1:end);

    disp(['Training Samples: ', num2str(length(Xtrain)), ' | Testing Samples: ', num2str(length(Xtest))]);

    % 2. Polynomial Regression with Degree K = 2
    K = 2;
    biastrain = ones(length(Xtrain), 1);
    biastest = ones(length(Xtest), 1);

    Atrain = biastrain;
    Atest = biastest;
    for i = 1:K
        Atrain = [Atrain, Xtrain.^i];
        Atest = [Atest, Xtest.^i];
    end

    % Normal Equation for Optimal Theta
    theta = (Atrain' * Atrain) \ (Atrain' * Ytrain);

    ypredtrain = Atrain * theta;
    ypredtest = Atest * theta;
    msetrain = mean((ypredtrain - Ytrain).^2);
    msetest = mean((ypredtest - Ytest).^2);

    figure; hold on;
    scatter(Xtest, Ytest, 'filled');
    [Xsorted, idx] = sort(Xtest);
    plot(Xsorted, ypredtest(idx), 'r', 'LineWidth', 2);
    grid on;
    title('Polynomial Regression (K = 2)');
    xlabel('Number of Rooms'); ylabel('House Prices');
    legend('Test Data', 'Regression Fit (K=2)');

    disp(['MSE Training Set (K=2): ', num2str(msetrain)]);
    disp(['MSE Test Set (K=2): ', num2str(msetest)]);

    % 3. High-Degree Polynomial (K = 10) with Normalization & Ridge
    K2 = 10;
    A2train = biastrain;
    A2test = biastest;
    for i = 1:K2
        A2train = [A2train, Xtrain.^i];
        A2test = [A2test, Xtest.^i];
    end

    theta2 = (A2train' * A2train) \ (A2train' * Ytrain);
    ypredtest2 = A2test * theta2;

    % Feature Normalization
    X_mean = mean(Xtrain);
    X_std = std(Xtrain);
    Xtrain_norm = (Xtrain - X_mean) / X_std;
    Xtest_norm = (Xtest - X_mean) / X_std;    

    A2_train_norm = biastrain;
    A2_test_norm = biastest;
    for i = 1:K2
        A2_train_norm = [A2_train_norm, Xtrain_norm.^i];
        A2_test_norm = [A2_test_norm, Xtest_norm.^i];
    end

    theta_norm = (A2_train_norm' * A2_train_norm) \ (A2_train_norm' * Ytrain);
    ypredtest_norm = A2_test_norm * theta_norm;
    msetest2_norm = mean((A2_test_norm * theta_norm - Ytest).^2);

    % Ridge Regression (L2 Regularization)
    l = 0.1;
    n_params = K2 + 1;
    Ridge_matrix = (A2_train_norm' * A2_train_norm + l * eye(n_params));
    theta_ridge = Ridge_matrix \ (A2_train_norm' * Ytrain);
    ypredtest_ridge = A2_test_norm * theta_ridge;
    msetest_ridge = mean((ypredtest_ridge - Ytest).^2);

    disp(['MSE Test K=10 with Normalization: ', num2str(msetest2_norm)]);
    disp(['MSE Test Ridge K=10 (\lambda=', num2str(l), '): ', num2str(msetest_ridge)]);

    % Plot Comparison for K=10
    figure; hold on;
    scatter(Xtest, Ytest, 'filled');
    plot(Xsorted, ypredtest2(idx), 'm--', 'LineWidth', 1.5);
    plot(Xsorted, ypredtest_norm(idx), 'k', 'LineWidth', 1.5);
    plot(Xsorted, ypredtest_ridge(idx), 'r', 'LineWidth', 2);
    title('Polynomial Regression (K = 10) Comparison');
    xlabel('Number of Rooms'); ylabel('House Prices');
    legend('Data', 'K=10 (Unnormalized)', 'K=10 (Normalized)', 'Ridge Regression');
    grid on;

% =========================================================
% EXERCISE 2: Gradient Descent, Adam, & Mini-Batch Optimizers
% =========================================================
elseif choice == 2
    
    erwthma = 5; % Select sub-exercise (1 to 5)
    
    data = readtable('BostonHousing.csv');
    dataMatrix = table2array(data);
    rand_per = randperm(size(dataMatrix, 1));
    X = dataMatrix(rand_per, 6);
    Y = dataMatrix(rand_per, end);

    trainRatio = 0.7;
    trainSize = floor(trainRatio * length(Y));
    K = 5;

    X_train = X(1:trainSize);
    Y_train = Y(1:trainSize);
    X_test = X(trainSize+1:end);
    Y_test = Y(trainSize+1:end);

    m_val = mean(X_train);
    s_val = std(X_train);

    X_train_norm = (X_train - m_val) / s_val;
    X_test_norm = (X_test - m_val) / s_val;

    X_train_poly = X_train_norm;
    X_test_poly = X_test_norm;

    for k = 2:K
        X_train_poly = [X_train_poly, X_train_norm.^k];    
        X_test_poly = [X_test_poly, X_test_norm.^k];    
    end
    X_train = [ones(size(X_train_poly, 1), 1), X_train_poly];
    X_test = [ones(size(X_test_poly, 1), 1), X_test_poly];

    dlX_train = dlarray(X_train);
    dlY_train = dlarray(Y_train);
    dlX_test = dlarray(X_test);
    dlY_test = dlarray(Y_test);

    [m_samples, n_features] = size(X_train);    
    max_iter = 1000;
    tol = 1e-6;

    if erwthma == 1 || erwthma == 2 || erwthma == 4 || erwthma == 5
        % Standard Gradient Descent
        theta_gd = dlarray(zeros(n_features, 1));    
        learning_rate_gd = 0.0001;    
        loss_history_gd_train = [];
        loss_history_gd_test = [];

        theta = theta_gd;
        for iter = 1:max_iter
            [loss_train, grad_train] = dlfeval(@mse_loss, dlX_train, dlY_train, theta);    
            loss_test = dlfeval(@mse_loss, dlX_test, dlY_test, theta);    
            
            theta = theta - learning_rate_gd * grad_train;    

            loss_history_gd_train = [loss_history_gd_train, extractdata(loss_train)];
            loss_history_gd_test = [loss_history_gd_test, extractdata(loss_test)];

            if norm(extractdata(grad_train)) < tol
                break;
            end
        end
        fprintf('Gradient Descent Training Completed.\n');
    end

    if erwthma >= 3
        % Adam Optimizer Setup
        learning_rate_adam = 0.2; 
        theta_adam = dlarray(zeros(n_features, 1)); 
        beta1 = 0.9; beta2 = 0.999; epsilon = 1e-8; 
        m_adam = dlarray(zeros(n_features, 1)); 
        v_adam = dlarray(zeros(n_features, 1)); 
        
        theta = theta_adam;
        for iter = 1:max_iter
            [loss_train, grad_train] = dlfeval(@mse_loss, dlX_train, dlY_train, theta);

            m_adam = beta1 * m_adam + (1 - beta1) * grad_train;
            v_adam = beta2 * v_adam + (1 - beta2) * (grad_train .^ 2);
            
            m_hat = m_adam / (1 - beta1^iter);
            v_hat = v_adam / (1 - beta2^iter);

            theta = theta - learning_rate_adam * (m_hat ./ (sqrt(v_hat) + epsilon));

            if norm(extractdata(grad_train)) < tol
                break;
            end
        end
        fprintf('Adam Optimizer Training Completed.\n');
    end

% =========================================================
% EXERCISE 3: Genetic Algorithms (GA) Optimization
% =========================================================
elseif choice == 3
    
    data = readtable('BostonHousing.csv');
    dataMatrix = table2array(data);
    rand_per = randperm(size(dataMatrix, 1));
    X = dataMatrix(rand_per, 6);
    Y = dataMatrix(rand_per, end);

    trainRatio = 0.7;
    trainSize = floor(trainRatio * length(Y));
    K = 5;

    X_train = X(1:trainSize);
    Y_train = Y(1:trainSize);
    X_test = X(trainSize+1:end);
    Y_test = Y(trainSize+1:end);

    m_val = mean(X_train);
    s_val = std(X_train);
    X_train_norm = (X_train - m_val) / s_val;
    X_test_norm = (X_test - m_val) / s_val;

    X_train_poly = X_train_norm;
    X_test_poly = X_test_norm;
    for k = 2:K
        X_train_poly = [X_train_poly, X_train_norm.^k];    
        X_test_poly = [X_test_poly, X_test_norm.^k];    
    end
    X_train = [ones(size(X_train_poly, 1), 1), X_train_poly];
    X_test = [ones(size(X_test_poly, 1), 1), X_test_poly];
    
    N_params = K + 1;
    LowerBound = -10 * ones(1, N_params);    
    UpperBound = 10 * ones(1, N_params);    

    options = optimoptions('ga', ...
                           'Display', 'iter', ...    
                           'PopulationSize', 200, ...    
                           'Generations', 10);

    [theta_ga, train_mse_ga] = ga(@(theta) msee_cost(theta, X_train, Y_train), ...
                                  N_params, [], [], [], [], ...    
                                  LowerBound, UpperBound, [], options);

    fprintf('Results of Genetic Algorithm (GA):\n');
    fprintf('  Final MSE on Training Set: %f\n', train_mse_ga);
    disp(theta_ga);

    H_test = X_test * theta_ga';    
    Error_test = H_test - Y_test;
    test_mse_ga = mean(Error_test.^2);
    fprintf('  Final MSE on Test Set: %f\n', test_mse_ga);

% =========================================================
% EXERCISE 4: Simulated Annealing (SA) Optimization
% =========================================================
elseif choice == 4
    
    data = readtable('BostonHousing.csv');
    dataMatrix = table2array(data);
    rand_per = randperm(size(dataMatrix, 1));
    X = dataMatrix(rand_per, 6);
    Y = dataMatrix(rand_per, end);

    trainRatio = 0.7;
    trainSize = floor(trainRatio * length(Y));
    K = 5;

    X_train = X(1:trainSize);
    Y_train = Y(1:trainSize);
    X_test = X(trainSize+1:end);
    Y_test = Y(trainSize+1:end);

    m_val = mean(X_train);
    s_val = std(X_train);
    X_train_norm = (X_train - m_val) / s_val;
    X_test_norm = (X_test - m_val) / s_val;

    X_train_poly = X_train_norm;
    X_test_poly = X_test_norm;
    for k = 2:K
        X_train_poly = [X_train_poly, X_train_norm.^k];    
        X_test_poly = [X_test_poly, X_test_norm.^k];    
    end
    X_train = [ones(size(X_train_poly, 1), 1), X_train_poly];
    X_test = [ones(size(X_test_poly, 1), 1), X_test_poly];
    
    N_params = K + 1;
    LowerBound = -10 * ones(1, N_params);    
    UpperBound = 10 * ones(1, N_params);    
    theta_initial = randn(N_params, 1);

    options = optimoptions('simulannealbnd', ...
                           'Display', 'iter', ...    
                           'MaxIterations', 2000, ...
                           'InitialTemperature', 100, ...
                           'TemperatureFcn', @temperaturefast);

    cost_function_sa = @(theta) mse_cost(theta, X_train, Y_train);

    [theta_sa, train_mse_sa] = simulannealbnd(cost_function_sa, ...
                                               theta_initial, ...
                                               LowerBound, UpperBound, options);

    fprintf('Simulated Annealing Results:\n');
    fprintf('Final MSE on Training Set: %f\n', train_mse_sa);

    H_test = X_test * theta_sa';    
    Error_test = H_test - Y_test;
    test_mse_sa = mean(Error_test.^2);
    fprintf('Final MSE on Test Set: %f\n', test_mse_sa);
end

% =========================================================
% HELPER FUNCTIONS
% =========================================================
function mse = msee_cost(theta, X, Y)
    m = length(Y);    
    H = X * theta';    
    Error = H - Y;
    mse = (1/m) * sum(Error.^2);
end

function [loss, grad, m_adam, v_adam] = adam_optimizer(lossFcn, X, Y, theta, iter, m_adam, v_adam, lr)
    % Helper function for Mini-batch Adam optimization
    [loss, grad] = dlfeval(lossFcn, X, Y, theta);
    beta1 = 0.9; beta2 = 0.999; epsilon = 1e-8;
    
    if isempty(m_adam)
        m_adam = dlarray(zeros(size(grad)));
        v_adam = dlarray(zeros(size(grad)));
    end
    
    m_adam = beta1 * m_adam + (1 - beta1) * grad;
    v_adam = beta2 * v_adam + (1 - beta2) * (grad .^ 2);
    
    m_hat = m_adam / (1 - beta1^iter);
    v_hat = v_adam / (1 - beta2^iter);
    
    grad = m_hat ./ (sqrt(v_hat) + epsilon);
end
