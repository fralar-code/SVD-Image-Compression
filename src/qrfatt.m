function [Q,R] = qrfatt(A,toll)
[P,H]= hessemberg(A);
n=length(H);
%Q contains the eigenvectors.
Q=P;
G=zeros(n-1,2);
error=toll+1;
while error > toll
    %shift s=H(n,n);
    delta=(H(n-1,n-1)-H(n,n))/2;
    s=H(n,n)-sign(delta)*H(n,n-1)^2/(sign(delta)+sqrt(delta^2+H(n,n-1)^2)); 
    H=H-s*eye(n);
    for k=1:n-1
        %It takes the Givens rotations of the Gij matrix, specific to eliminate the subdiagonal.
        [G(k,1),G(k,2)]= givensRotations(H,k,k+1);
        %Apply the rotation to H, that is, perform the product Gij * H.
        H(k:k+1, k:n) = [G(k,1) -G(k,2); G(k,2) G(k,1)]*H(k:k+1, k:n); 
    end

    for k=1:n-1
        %We post-multiply by the transposed Givens matrices to obtain Hi+1 = G * A * G', 
        %the Hessenberg matrix for the next iteration.
        H(1:k+1, k:k+1) = H(1:k+1, k:k+1)*[G(k,1) G(k,2); -G(k,2) G(k,1)]; 
    
        %Q keeps track of all the transformations performed: Q = G12' * G23' * G34'...
        % Q=Q*G;
        Q(:, k:k+1) = Q(:, k:k+1)*[G(k,1) G(k,2); -G(k,2) G(k,1)]; 
    end
    H=H+s*eye(n);

    %As the method converges, the element h(n,n-1) will approach 0.
    error=abs(H(n,n-1));
end
R=H;




