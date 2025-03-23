A = imread('baboon.tiff');
[n,m,d]=size(A);
A=double(A);
R = A(:,:,1); 
G = A(:,:,2); 
B = A(:,:,3);
%I apply the SVD to each of the 3 channels.
[U_r,S_r,V_r] = my_svd(R);
[U_g,S_g,V_g] = my_svd(G);
[U_b,S_b,V_b] = my_svd(B);

K=[22, 55, 100, 320, 500];
for i=1:size(K,2)
    k=K(i);
    %I reconstruct each channel by considering the first k singular values.
    R_compr= U_r(:,1:k)*diag(S_r(1:k))*V_r(:,1:k)';
    G_compr= U_g(:,1:k)*diag(S_g(1:k))*V_g(:,1:k)';
    B_compr= U_b(:,1:k)*diag(S_b(1:k))*V_b(:,1:k)';
    %I reconstruct the entire image using the first k singular values.
    img=cat(3,R_compr(:,:,1),G_compr(:,:,1),B_compr(:,:,1));
    %Compression ratio
    ratio=(n*m)/(n*k+k+m*k);
    subplot(1,5,i),imshow(uint8(img)),title(['k:' num2str(k) '   Ratio:' num2str(ratio)]);hold  on
end

