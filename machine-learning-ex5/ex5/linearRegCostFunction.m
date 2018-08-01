function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

H=X*theta;
Diff=H-y;
DiffSquare= Diff.^2;
DiffSquareSum=sum(DiffSquare);
tempJ=(DiffSquareSum)/(2*m);

theta2=theta.^2;
tempTheta= sum(theta2)-theta2(1,1);
tempTheta1=tempTheta*lambda;
tempTheta2=(tempTheta1)/(2*m);

J=tempJ+tempTheta2;

tempGrad=(sum((Diff.*X)))/m;
for i = 2:size(tempGrad,2)
  tempGrad(1,i)=tempGrad(1,i)+(((lambda)*theta(i,1))/m);
endfor

grad=tempGrad;


% =========================================================================

grad = grad(:);

end
