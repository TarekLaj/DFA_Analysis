
clear;
clc;
load('/home/karim/DFA_Analysis/packages/fractaldata.mat')

X=cumsum(multifractal-mean(multifractal));
X=transpose(X);
scale=[16,32,64,128,256,512,1024];
m=1;
for ns=1:length(scale)
  segments(ns)=floor(length(X)/scale(ns));
  for v=1:segments(ns)

    Idx_start=((v-1)*scale(ns))+1;
    Idx_stop=v*scale(ns);
    Index{v,ns}=Idx_start:Idx_stop;
    X_Idx=X(Index{v,ns});
    
    
    C=polyfit(Index{v,ns},X(Index{v,ns}),m);
    fit{v,ns}=polyval(C,Index{v,ns});
    RMS{ns}(v)=sqrt(mean((X_Idx-fit{v,ns}).^2));
  end
  F(ns)=sqrt(mean(RMS{ns}.^2));
end


C=polyfit(log2(scale),log2(F),1);
H=C(1);
RegLine=polyval(C,log2(scale));