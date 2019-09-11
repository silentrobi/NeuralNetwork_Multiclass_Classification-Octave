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
# hidden_layer_size * (input_layer_size + 1) is 25 * 401
# Theta 1 is 25 x 401
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                hidden_layer_size, (input_layer_size + 1)); # nn_params is thetaVector and need to reshape.
# Theta 2 is 10 x 26
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
X = [ones(m, 1) X]; 

% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% +++++++ Feedforward Propogation Algorithm ++++++++
% +++++++Vectorize Implementation +++++++++

%Z2 = Theta1 X   Theta1 = 25 x 401 and X = 5000 x 401
Z2= Theta1 * X'; # Z2= 25 x 5000 return Z2 for 5000 training sets

A2= sigmoid(Z2); # return 25 x 5000 matrix

A2= [ones(1,m); A2]; # add bias term 1 to to A2 for 5000 training data , 26 x 5000
% Theta2 has size 10 x 26 

Z3= Theta2 * A2; # 10 x 5000 

A3 = sigmoid(Z3); # 10 x 5000 

% +++++++ Creating k x m matrix of output Where Each column represent training data output and each row  is class label ++++++++

# lso, recall that whereas the original labels
# (in the variable y) were 1, 2, ..., 10, for the purpose of training a neural
# network, we need to recode the labels as vectors containing only values 0 or 1
Y= zeros(num_labels, size(y)); # k x m matrix, where m is size of y, (10X 5000) matrix
for col=1:size(y),
   for row=1: num_labels,
     if row == y(col,1),
       Y(row,col) = 1;
     endif
   endfor
endfor

% ++++++ Computation_part does column wise multiplication and addition +++++++++++ 
Computation_part= -(Y .* log(A3) + (1- Y) .* log(1- A3));  #(k x m) = (10 x 5000) matrix
% ++++++ Does K summation part of the cost function of J +++++++++++ 
COST = sum(Computation_part); # column wise sum.  (1 x 5000) matrix

% ++++++ Does m summation part of the cost function of J +++++++++++
J = sum(COST)/m; # row wise sum
% Finding cost J
% +++++++ Calculating regularization term ++++++++++++
Theta1_reg = Theta1(:,2:end);  # Theta1_reg doesn't have first column. Theta1_reg = 25 x 400
Theta2_reg = Theta2(:,2:end);  # Theta2_reg doesn't have first column. Theta2_reg = 10 x 25

# left Sum Term: inner sum add 400 elements column wise, outer  sum add 25 elements row wise.  
# Right Sum Term: inner sum add 25 elements column wise, outer  sum add 10 elements row wise. 
reg= lambda/(2*m) * (sum(sum(Theta1_reg.^2 ,2)) + sum(sum(Theta2_reg.^2,2)));
% adding the regularization term to cost function J

J= J + reg; 

Theta1_grad = zeros(size(Theta1));  # 25x 401
Theta2_grad = zeros(size(Theta2));  #10 x 26

%++++++++++BackPropogation Algorithm+++++++++++++++
% +++++++Vectorize Implementation +++++++++
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
Delta1 = zeros(size(Theta1)); # 25 x 401
Delta2 = zeros(size(Theta2)); # 10 x 26

Sigma3 = A3 - Y; # gives Sigma3 , (10 x 5000) matrix

Sigma2 = (Theta2' * Sigma3) .* [ones(1,m);sigmoidGradient(Z2)];  # gives Sigma2 , (26 x 5000) matrix

# Note no Delta1
C = Sigma2(2:end,:) * X;
Delta1 = Delta1 + C; 
C= Sigma3 * A2';
Delta2 = Delta2 + C;

#grad_reg_Theta1 = lambda * Theta1 ;
# regularization term doesn't count first column. So, first column is set to zero
Theta1_grad = BigDelta1/m  + (lambda/m)* [zeros(size(Theta1,1),1), Theta1(:,2:end)];
Theta2_grad = BigDelta2/m + (lambda/m)* [zeros(size(Theta2,1),1), Theta2(:,2:end)];


%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
