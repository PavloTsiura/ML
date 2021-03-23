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

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


m = size(X, 1);
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


%Feedforward
h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');


for i = 1:m
   temp_y = zeros(1, num_labels);
   temp_y(y(i)) = 1;
   J = J-sum(temp_y.*log(h2(i,:))+(1-temp_y) .* log(1-h2(i,:)));
end

J = J/m+lambda*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)))/(2*m);

%backpropagation
for t = 1:m
    a_1 = [1; X(t,:)'];
    z_2 = Theta1 * a_1;
    
    a_2 = [1;sigmoid(z_2)];
    a_3 = sigmoid(Theta2 * a_2);
    
    temp_y = zeros(num_labels,1);
    temp_y(y(t)) = 1;
    
    delta_3 = a_3 - temp_y;
    
    delta_2 = (Theta2'*delta_3).*[1;sigmoidGradient(z_2)];
    delta_2 = delta_2(2:end);
    
    Theta1_grad = Theta1_grad+delta_2*a_1';
	Theta2_grad = Theta2_grad+delta_3*a_2';
end

Theta1_grad = Theta1_grad/m;
Theta2_grad = Theta2_grad/m;


Theta1_grad = Theta1_grad + lambda*([zeros(size(Theta1,1),1),Theta1(:,2:end)])/m;
Theta2_grad = Theta2_grad + lambda*([zeros(size(Theta2,1),1),Theta2(:,2:end)])/m;

grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
