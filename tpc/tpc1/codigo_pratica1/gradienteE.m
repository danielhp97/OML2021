function [ GradE ] = gradienteE(w,Xt,Yt,paramG)
% vetor gradiente de E  

I=length(w)-1;  %grau do polinomio;
nt=length(Xt);
GradE=zeros(1,I+1); % inicializar com o vetor nulo

if paramG==1  %gradiente completo
    for i=1:nt
        GradE=GradE + (funphi(w,Xt(i),I)-Yt(i))*power(Xt(i),0:I);
    end
    GradE=GradE/nt;
end

if paramG==2  %gradiente estocástico
    i=randi(nt);  %escolhe aleatoriamente um indice na gama de 1:nt
    GradE=(funphi(w,Xt(i),I)-Yt(i))*power(Xt(i),0:I);
end

if paramG==3   %gradiente mini-batch
    nb=round(0.05*nt);   %selecionar 5% do Dt
    p=randperm(nt,nb);    %gera aleatoriamente nb numeros inteiros do intervalo 1:nt
    for i=1:nb
        GradE=GradE + (funphi(w,Xt(p(i)),I)-Yt(p(i)))*power(Xt(p(i)),0:I);
    end
    GradE=GradE/nb;
    
end
GradE=GradE';   %tem que swer vetor coluna
end

