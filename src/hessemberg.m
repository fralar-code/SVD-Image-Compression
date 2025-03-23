function [H, A] = hessemberg(A)
n=size(A,2);
H=eye(n);
for k=1:n-2
    % apply the transformation excluding the first element of the k-th column, starting from the index at position k+1
    sigma=sign(A(k+1,k))*norm(A(k+1:n,k));
    %v is the k-th column with the first element perturbed by sigma.
    v =[sigma+A(k+1,k);A(k+2:n,k)]; 
    beta =1/(sigma*(sigma+A(k+1,k))); 

    %Transforms the lower part of the matrix A, that is, the transformation HA.
    for j=k:n
        tau = (v'*A(k+1:n,j))*beta;
        A(k+1:n,j)= A(k+1:n,j)-tau*v;
    end
    %Transforms the upper part of the matrix A, that is, the transformation AH.
    for j=1:n
        tau = (A(j,k+1:n)*v)*beta;
        A(j,k+1:n)= A(j,k+1:n)-tau*v';
        %We keep track of the transformation performed, that is, the AH..Hn-3Hn-2, the matrix that post-multiplies A
        tau = (H(j,k+1:n)*v)*beta;
        H(j,k+1:n)= H(j,k+1:n)-tau*v';
    end
end