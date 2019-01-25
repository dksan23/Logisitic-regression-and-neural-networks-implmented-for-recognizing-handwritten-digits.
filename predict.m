function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);
X=[ones(m,1) X];

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);


a2=sigmoid((Theta1*X')');
a2=[ones(size(a2,1),1) a2];
h=sigmoid((Theta2*a2')');
[pi,ix]=max(h,[],2);
p=ix;







% =========================================================================


end
