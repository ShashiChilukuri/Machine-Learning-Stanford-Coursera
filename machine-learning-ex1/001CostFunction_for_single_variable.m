


%First load data from the file into Octave
  data = load('ex1data1.txt');

% data is in 97 x 2 matrix
% select X and y
  X = data(: , 1); 
  y = data(:, 2);
  
%define 'm'
 m = length(y);
 
%add x0 column to X values 
  X = [ones(m,1),  data(:,1)];

% define theta values
  theta = zeros(2,1);

now define the function for standered cost function
  J = 1 / (2 * m) * sum(((X * theta) - y).^2);  

% to calculate cost call the function like this
  computeCost(X, y, theta)


% ================Cut below section to make it a function==============================================================


function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.


J = 1 / (2 * m) * sum(((X * theta) - y).^2);


% =========================================================================

end