function [U,S,V] = my_svd(A)
[m,n]=size(A);
[Q,R]=qrfatt(A*A',10^3);
%S is defined as the square root of the ordered eigenvalues.
S=sqrt(diag(R));
V=Q;
%The first column of V represents the eigenvector associated with the first eigenvalue of S. 
%when sorting the eigenvalues, we must maintain the correspondence.
if ~issorted(S,'descend')
    [S, indexs]=sort(S,'descend');
    V=V(:,indexs);
end
%We can derive U from the relation A = U * S * V'. To do so, we need the inverse singular values.
S2=1./S(S~=0);
S2(n)=0;
U=A*V*diag(S2);