function [ valE ] = funE(w,Xt,Yt)
%fun��o objetivo E (= fun��o custo E) 

I=length(w)-1;  %grau do polinomio;
nt=length(Xt);

valE=0;
for i=1:nt
    valE=valE + (funphi(w,Xt(i),I)-Yt(i))^2;
end
valE=valE/(2*nt);


end

