function [ etak ] = passo(wk,Ek,gradk,sk,Xt,Yt,paramP,k,nt)
% Calcula o comprimento do passo
% 
 if paramP==1
     etak = backtrackingArmijo(wk,Ek,gradk,sk, Xt,Yt);
 end
 
 if paramP==2
     etak=0.1;
 end
 
 if paramP==3
     etak=1;
     if mod(k,nt)==0
         d=k/nt;
         etak=1/(10^d);  %min(1/10^d,1e-6)
     end
 end

end

