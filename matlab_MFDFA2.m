

clear;
clc;
load('/home/karim/DFA_Analysis/packages/fractaldata.mat')

X=cumsum(multifractal-mean(multifractal));
X=transpose(X);
scale_small=[7,9,11,13,15,17];
halfmax=floor(max(scale_small)/2);
Time_index=halfmax+1:length(X)-halfmax;
scale=[16,32,64,128,256,512,1024];
q=[-5,-3,-1,0,1,3,5];
m=1;


for ns=1:length(scale_small),
    halfseg=floor(scale_small(ns)/2);
    for v=halfmax+1:length(X)-halfmax;
        T_index=v-halfseg:v+halfseg;
        C=polyfit(T_index,X(T_index),m);
        fit=polyval(C,T_index);
        RMS{ns}(v)=sqrt(mean((X(T_index)-fit).^2));
    end
    for nq=1:length(q),
        qRMS{nq,ns}=RMS{ns}.^q(nq);
        Fq(nq,ns)=mean(qRMS{nq,ns}).^(1/q(nq));
    end
  
    Fq(q==0,ns)=exp(0.5*mean(log(RMS{ns}.^2)));
end



C=polyfit(log2(scale),log2(Fq(q==0,:)),1);
Regfit=polyval(C,log2(scale_small));
maxL=length(X);
for ns=1:length(scale_small);
    RMSt=RMS{ns}(Time_index);
    resRMS{ns}=Regfit(ns)-log2(RMSt);
    logscale(ns)=log2(maxL)-log2(scale_small(ns));
    Ht(ns,:)=resRMS{ns}./logscale(ns)+Hq(q==0);
end

Ht_row=Ht(:);
BinNumb=round(sqrt(length(Ht_row)));
[freq,Htbin]=hist(Ht_row,BinNumb);
Ph=freq./sum(freq);
Ph_norm=Ph./max(Ph);
Dh=1-(log(Ph_norm)./-log(mean(scale)));