function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


%%% FORWARD PROPOGATION

% a1 (input)
a1 = [ones(size(X), 1) X];
z2 = sigmoid(a1*Theta1'); %'

% a2 (hidden)
a2 = [ones(size(a1), 1) z2];

% a3 (output) = z3
a3 = sigmoid(a2*Theta2'); %'

vectorizedY = zeros(num_labels, 1);

thetaTrans = Theta2'; %'

for ex = 1:m,


	%%% COMPUTING CUMULATIVE COST OF ALL TRAINING EXAMPLES

	vectorizedY(y(ex)) = 1;
	predictedY = a3(ex, :)'; %'

	% We take a newly created vector for the expected output y
	% And we take a row representing the outputs for each training example
	% We transpose the row (sucht hat it corresponds with a y value)
	% And then use our cost function as expected.
    
    J += sum((-vectorizedY) .* log(predictedY) - (1 - vectorizedY) .* (log(1 - predictedY)));





	%%% USING BACK PROPOGATION FOR COMPUTING GRADIENTS

	% a3 is the result of forward propogation, it is a 5000 x 10 matrix, 
	% so each row represents the output of a single training example.
	% We have already built up a vector representation of the current Y and the current predictedY values

	a2Current = a2(ex, :);

	% The last layer
	delta3 = predictedY - vectorizedY;

	% The second last layer, each row j represents the deltaValue for the node in the jth node position
	delta2 = thetaTrans * delta3 .* (a2Current .* (1 - a2Current))'; %'
	delta2 = delta2(2:end);


	% delta3 = 10x1, 10 nodes in last layer
	% delta2 = 25x1, 25 nodes in hidden layer

	% input into last layer is of size 1x26 (including bias)
	% input into hidden layer is of size 1x401 (including bias)

	Theta2_grad = Theta2_grad + delta3 * a2Current; %'
	Theta1_grad = Theta1_grad + delta2 * a1(ex, :); %'

	vectorizedY(y(ex)) = 0;

end

% Theta2_grad is 10x25
% Theta1_grad is 25x400

Theta2_grad = (Theta2_grad) / m;
Theta1_grad = (Theta1_grad) / m;

Theta2_grad += [zeros(size(Theta2_grad), 1) Theta2(:, 2:end)*(lambda/m) ];
Theta1_grad += [zeros(size(Theta1_grad), 1) Theta1(:, 2:end)*(lambda/m) ];




J = sum(J) / m;

% Removing the first row which is for the bias
J += ((sum(sum(Theta1(1:end, 2:end)(:) .^ 2))) + (sum(sum(Theta2(1:end, 2:end)(:) .^ 2)))) * (lambda / (2 * m));

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
