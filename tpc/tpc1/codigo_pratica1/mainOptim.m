
%ler o data set  data1.xlsx
Data=xlsread('data1.xlsx');
N=length(Data);
 
X=Data(:,1);
Y=Data(:,2);

%gr�fico do data set 
figure (1)
plot(X,Y,'bo','Markersize',1);
hold on
xdata=(0:0.01:1)';
np=length(xdata);
%-----------------------------


%Data treino - 80% do Data -- selecionado aleatoriamente
p = randperm(N); %returns a row vector containing a random permutation of the integers from 1 to N inclusive.
nt=0.8*N;
Xt=Data(p(1:nt),1);
Yt=Data(p(1:nt),2);

%Data para valida��o - 20% do Data -- selecionado aleatoriamente 
Xv=Data(p(nt+1:N),1);
Yv=Data(p(nt+1:N),2);


%polinomio de grau I a ajustar
I=7;  %grau do polin�mio I=2, I=3, I=4, I=5, I=6, I=7

%M�todo do gradiente completo correr com paramG=1 
%M�todo do gradiente estoc�tico correr com paramG=2
%M�todo do gradiente mini-bacth estoc�stico correr com paramG=3

paramP=1;  %ParamP=1-usa alg. backtracking com cond. Armijo;  ParamP=2 - passo constante; ParamP=3-reduz ao fim de uma �poca
paramG=2;  %ParamG=1-calcula o gradiente completo;  ParamG=2-calcula o gradiente estoc�stico;  ParamG=3- calcula o gradiente estoc�stico mini-batch

%-------
tol=1e-4;
%---------
maxit=10*nt;  % �poca � nt 
k=1;
wk=zeros(I+1,1); %ponto inicial
%--
E=[];

while ( norm(gradienteE(wk,Xt,Yt,1))> tol && k<=maxit)  % -- no estoc�stico e mini-batch usar apenas n�mero de itera��es!!
    
    %calcular o gradiente no ponto wk
    gradk=gradienteE(wk,Xt,Yt,paramG);
    %calcular a dire��o de procura
    sk=-gradk; 
    
    %calcular o comprimento do passo/deslocamento
    Ek=funE(wk,Xt,Yt)  %calcular a fun��o objetivo/custo no ponto wk
    gradFullk=gradienteE(wk,Xt,Yt,1); %calcula o gradiente completo no ponto wk
    etak=passo(wk,Ek,gradFullk,sk,Xt,Yt,paramP,k,nt);  
    %novo ponto
    wk=wk+etak*sk;
    k=k+1;
    
    %----  para fazer o gr�fico de Ek
    E=[E,Ek];
    %-- para fazer o gr�fico do polinomio ao longo do processo iterativo
%     for i=1:np
%       ydata(i)=phi(wk,xdata(i),I);  %calcula o polinomio no pto xdata(i)
%     end
%     plot(xdata,ydata,'.-m');
end



%---
fprintf('solu��o �tima w*: \n'); 
wk %solu��o �tima: a �ltima a  sair do processo iterativo

EDt=funE(wk,Xt,Yt);  % valor �timo == in-sample error  (erro no com data de treino)
fprintf('\n in-sample error E(wopt,Dt): %.12e ', EDt);


EDv=funE(wk,Xv,Yv);   % out-sample error == erro com o data de valida��o
fprintf('\n out-sample error E(wopt,Dv): %.12e', EDv);

%polinomio com os par�metros �timos 
for i=1:np
    ydata(i)=funphi(wk,xdata(i),I);  %calcula o polinomio no ponto xdata(i)
end


plot(xdata,ydata,'.-g');


hold off

%valor da funE (fun��o objectivo/custo) ao longo do processo iterativo
figure(2)
plot(E)

%

