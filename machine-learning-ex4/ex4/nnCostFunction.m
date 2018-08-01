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

X = [ones(m, 1) X];
a1=X;
z2=X*Theta1';
a2temp=sigmoid(z2);
a2 =[ones(size(a2temp,1),1) a2temp];
z3=a2*Theta2';
H=sigmoid(z3);

Y=zeros(m,num_labels);

for i =1:m
  Y(i,y(i))=1;
endfor


YH=-Y.*log(H);
oneY=ones(size(Y))-Y;
oneH=ones(size(H))-H;
oneYH=-oneY.*log(oneH);
sumYH=oneYH+YH;
tmepJ1=(sum(sum(sumYH)))/m;
 
thetasqr=0;
for j=1:size(Theta1,1)
  for k=2:size(Theta1,2)
    thetasqr = thetasqr + Theta1(j,k)*Theta1(j,k);
  endfor
endfor
for j=1:size(Theta2,1)
  for k=2:size(Theta2,2)
    thetasqr = thetasqr + Theta2(j,k)*Theta2(j,k);
  endfor
endfor

tempJ2=(thetasqr*lambda)/(2*m);
J=tempJ2+tmepJ1;

delta2 = zeros(size(Theta2));
delta1 = zeros(size(Theta1));
for i=1:m
  a_1=(a1(i,1:size(a1,2)))';
  z_2=(z2(i,1:size(z2,2)))';
  a_2=(a2(i,1:size(a2,2)))';
  z_3=(z3(i,1:size(z3,2)))';
  a_3=(H(i,1:size(H,2)))';
  y_i=(Y(i,1:size(Y,2)))';
  
  d_3=(a_3-y_i);
  d_2=((Theta2')*d_3) .* [1;sigmoidGradient(z_2)];
  
  tempdelta2=d_3*a_2';
  delta2=delta2+tempdelta2;
  
  tempdelta1=d_2(2:size(d_2,1))*(a_1)';
  delta1=delta1+tempdelta1;
  
endfor

Theta1_grad=(delta1)/m;
Theta2_grad=(delta2)/m;


for i=1:size(Theta1,1)
  Theta1(i,1)=0;
endfor

for i=1:size(Theta2,1)
  Theta2(i,1)=0;
endfor

Theta1_grad=Theta1_grad+((lambda)*(Theta1))/m;
Theta2_grad=Theta2_grad+((lambda)*(Theta2))/m;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
