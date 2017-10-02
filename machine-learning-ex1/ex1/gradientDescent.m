function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

disp(theta);
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % X is of the form first column: 1 and second column values of X
    % So when we multiple by theta of the form (theta0, theta1) than we are actually computing h(x) as follows:
    % h(x) = theta0 + theta1*x



    theta0 = alpha * (1/m) * sum(X*theta - y);

    theta1 = alpha * (1/m) * sum((X*theta - y) .* X(:,2));

    theta = [theta(1, 1) - theta0; theta(2, 1) - theta1];



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end