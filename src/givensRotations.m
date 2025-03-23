%returns the Givens matrix used to eliminate the element of A at position (j,i)
function [cos,sin] = givensRotations(A,i,j)
cos= abs(A(i,i)) / sqrt(A(i,i)^2+A(j,i)^2);
sin= -sign(A(j,i)/A(i,i)) * abs(A(j,i))/sqrt(A(i,i)^2+A(j,i)^2);
