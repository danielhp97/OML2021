
%ler o data set  data1.xlsx
Data=xlsread('data1.xlsx');
N=length(Data);
 
X=Data(:,1);
Y=Data(:,2);

%gráfico do data set 
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

%Data para validação - 20% do Data -- selecionado aleatoriamente 
Xv=Data(p(nt+1:N),1);
Yv=Data(p(nt+1:N),2);


%polinomio de grau I a ajustar
I=7;  %grau do polinómio I=2, I=3, I=4, I=5, I=6, I=7

%Método do gradiente completo correr com paramG=1 
%Método do gradiente estocático correr com paramG=2
%Método do gradiente mini-bacth estocástico correr com paramG=3

paramP=1;  %ParamP=1-usa alg. backtracking com cond. Armijo;  ParamP=2 - passo constante; ParamP=3-reduz ao fim de uma época
paramG=2;  %ParamG=1-calcula o gradiente completo;  ParamG=2-calcula o gradiente estocástico;  ParamG=3- calcula o gradiente estocástico mini-batch

%-------
tol=1e-4;
%---------
maxit=10*nt;  % época é nt 
k=1;
wk=zeros(I+1,1); %ponto inicial
%--
E=[];

while ( norm(gradienteE(wk,Xt,Yt,1))> tol && k<=maxit)  % -- no estocástico e mini-batch usar apenas número de iterações!!
    
    %calcular o gradiente no ponto wk
    gradk=gradienteE(wk,Xt,Yt,paramG);
    %calcular a direção de procura
    sk=-gradk; 
    
    %calcular o comprimento do passo/deslocamento
    Ek=funE(wk,Xt,Yt)  %calcular a função objetivo/custo no ponto wk
    gradFullk=gradienteE(wk,Xt,Yt,1); %calcula o gradiente completo no ponto wk
    etak=passo(wk,Ek,gradFullk,sk,Xt,Yt,paramP,k,nt);  
    %novo ponto
    wk=wk+etak*sk;
    k=k+1;
    
    %----  para fazer o gráfico de Ek
    E=[E,Ek];
    %-- para fazer o gráfico do polinomio ao longo do processo iterativo
%     for i=1:np
%       ydata(i)=phi(wk,xdata(i),I);  %calcula o polinomio no pto xdata(i)
%     end
%     plot(xdata,ydata,'.-m');
end



%---
fprintf('solução ótima w*: \n'); 
wk %solução ótima: a última a  sair do processo iterativo

EDt=funE(wk,Xt,Yt);  % valor ótimo == in-sample error  (erro no com data de treino)
fprintf('\n in-sample error E(wopt,Dt): %.12e ', EDt);


EDv=funE(wk,Xv,Yv);   % out-sample error == erro com o data de validação
fprintf('\n out-sample error E(wopt,Dv): %.12e', EDv);

%polinomio com os parâmetros ótimos 
for i=1:np
    ydata(i)=funphi(wk,xdata(i),I);  %calcula o polinomio no ponto xdata(i)
end


plot(xdata,ydata,'.-g');


hold off

%valor da funE (função objectivo/custo) ao longo do processo iterativo
figure(2)
plot(E)

%

