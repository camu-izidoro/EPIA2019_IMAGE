%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Autor: Clodoaldo A M Lima
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [A,B,veterro]=rna(Xtr,Ytr,Xval,Yval,h)
Ntr=size(Xtr,1);
Nval=size(Xval,1);
Xtr=[Xtr,ones(Ntr,1)];
Xval=[Xval,ones(Nval,1)];
ne = size(Xtr,2);
ns = size(Ytr,2);
%Inicializa os pesos
A = rands(h,ne)/5;
B = rands(ns,h+1)/5;
% Calcula a saida da rede para o treinamento
Yr_tr = calc_saida(A,B,Xtr,ne,ns,h,Ntr);
% Calcula o erro de treinamento
erro_tr = Yr_tr - Ytr;
% Calcula o MSE para treinamento
EQM_tr = 1/Ntr*sum(sum(erro_tr.*erro_tr));
veterro=[];
veterro =[ veterro, EQM_tr];

% Calcula a saida da rede para  validação
Yr_val = calc_saida(A,B,Xval,ne,ns,h,Nval);
% Calcula o erro
erro_val = Yr_val - Yval;
% Calcula o MSE para validação
EQM_val_atual = 1/Nval*sum(sum(erro_val.*erro_val));
EQM_val_ant = inf;
nepocas =0;
nepocasmax = 5000;
alfa =0.9;
while nepocas<nepocasmax & EQM_tr>1e-5 %& EQM_val_atual<=EQM_val_ant
    %Atualiza a epocas
    nepocas = nepocas+1
    %Calcula o gradiente
    [dJdA,dJdB]=calc_grad(A,B,Xtr,Ytr,ne,ns,h,Ntr);
    %Defini a direção
    dir = [-dJdA(:);-dJdB(:)];
    %Calcula a taxa de aprendizado
    alfa = calc_alpha(A,B,Xtr,Ytr,dJdA, dJdB,ne,ns,h,Ntr);
    %Atualiza os pesos
    A = A - alfa*dJdA;
    B = B - alfa*dJdB;
    % Calcula a saida da rede para o treinamento
    Yr_tr = calc_saida(A,B,Xtr,ne,ns,h,Ntr);
    % Calcula o erro de treinamento
    erro_tr = Yr_tr - Ytr;
    % Calcula o MSE para treinamento
    EQM_tr = 1/Ntr*sum(sum(erro_tr.*erro_tr));
    % Calcula a saida da rede para  validação
    Yr_val = calc_saida(A,B,Xval,ne,ns,h,Nval);
    % Calcula o erro
    erro_val = Yr_val - Yval;
    % Calcula o MSE para validação
    EQM_val_ant = EQM_val_atual
    EQM_val_atual = 1/Nval*sum(sum(erro_val.*erro_val))
    veterro =[ veterro, EQM_tr];

end
end

function  alfa_m = calc_alpha(A,B,Xtr,Ytr,dJdA, dJdB,ne,ns,h,Ntr)
% algoritmo da bissecao
alfa_l=0;
alfa_u = rand;
Aaux = A - alfa_u*dJdA;
Baux = B - alfa_u*dJdB;
[dJdAaux,dJdBaux]=calc_grad(Aaux,Baux,Xtr,Ytr,ne,ns,h,Ntr);
grad = [dJdAaux(:); dJdBaux(:)];
dir = [-dJdA(:); -dJdB(:)];
hl = grad'*dir;

while hl<0
    alfa_u  = 2*alfa_u;
    Aaux = A - alfa_u*dJdA;
    Baux = B - alfa_u*dJdB;
    [dJdAaux,dJdBaux]=calc_grad(Aaux,Baux,Xtr,Ytr,ne,ns,h,Ntr);
    grad = [dJdAaux(:); dJdBaux(:)];
    hl = grad'*dir;
end

kmax = ceil(log((alfa_u-alfa_l)/1e-3));
k=0;
alfa_m = alfa_u;

while abs(hl)>1e-5 & k<kmax
    k = k+1;
    alfa_m = (alfa_l+alfa_u)/2;
    Aaux = A - alfa_m*dJdA;
    Baux = B - alfa_m*dJdB;
    [dJdAaux,dJdBaux]=calc_grad(Aaux,Baux,Xtr,Ytr,ne,ns,h,Ntr);
    grad = [dJdAaux(:); dJdBaux(:)];
    hl = grad'*dir;
    if hl>0
        alfa_u = alfa_m;
    elseif hl==0
        break;
    else 
        alfa_l = alfa_m;
    end
end
end

function [dJdA,dJdB]=calc_grad(A,B,Xtr,Ytr,ne,ns,h,N)
Zin = zeros(N,h);
Z = zeros(N,h+1);
Yin = zeros(N,ns);
Y = zeros(N,ns);
Zin = Xtr*A';
Z = 1./(exp(-Zin)+1);
Z = [Z,ones(N,1)];
Yin = Z*B';
Y = 1./(exp(-Yin)+1);
erro = Y - Ytr;

dJdB = (erro.*((1-Y).*Y))'*Z;
dJdB = dJdB/N;
dJdZ = erro.*((1-Y).*Y)*B(:,1:h);            
Z1 = Z(:,1:end-1);
dJdA = (dJdZ.*((1-Z1).*Z1))'*Xtr;
dJdA = dJdA/N;

% for n=1:N
%     for i=1:h
%         for j=1:ne
%            Zin(n,i)=Zin(n,i)+A(i,j)*X(n,j);
%         end
%         Z(n,i)=1/(exp(-Zin(n,i))+1);
%     end
%     Z(n,h+1)=1;
% end

% for n=1:N
%     for i=1:ns
%         for j=1:h+1
%            Yin(n,i)=Yin(n,i)+B(i,j)*Z(n,j);
%         end
%         Y(n,i)=1/(exp(-Yin(n,i))+1);
%     end
% end

%  dJdB1 = zeros(ns,h+1);
%  for n=1:N,
%      for k=1:ns,
%          for i=1:h+1,
%              dJdB1(k,i) =dJdB1(k,i)+ erro(n,k)*(1-Y(n,k))*Y(n,k)*Z(n,i);
%          end
%      end
%  end
 
 dJdZ = zeros(N,h);
 for n=1:N
     for i=1:ha
         for k=1:ns,
             dJdZ(n,i)=dJdZ(n,i)+erro(n,k)*(1-Y(n,k))*Y(n,k)*B(k,i);
         end
     end
 end

            
 dJdA1 = zeros(h,ne);
 for n=1:N,
     for k=1:h,
         for i=1:ne,
             dJdA1(k,i) =dJdA1(k,i)+ dJdZ(n,k)*(1-Z(n,k))*Z(n,k)*Xtr(n,i);
         end
     end
 end

 dJdA-dJdA1/N
 pause
 
end

function Yr=calc_saida(A,B,X,ne,ns,h,N)
Zin = zeros(N,h);
Z = zeros(N,h+1);
Yin = zeros(N,ns);
Y = zeros(N,ns);
Zin = X*A';
Z = 1./(exp(-Zin)+1);
Z = [Z,ones(N,1)];
Yin = Z*B';
Yr = 1./(exp(-Yin)+1);

% for n=1:N
%     for i=1:h
%         for j=1:ne
%            Zin(n,i)=Zin(n,i)+A(i,j)*X(n,j);
%         end
%         Z(n,i)=1/(exp(-Zin(n,i))+1);
%     end
%     Z(n,h+1)=1;
% end

% for n=1:N
%     for i=1:ns
%         for j=1:h+1
%            Yin(n,i)=Yin(n,i)+B(i,j)*Z(n,j);
%         end
%         Y(n,i)=1/(exp(-Yin(n,i))+1);
%     end
% end
end

