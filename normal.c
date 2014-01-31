#include <math.h>
#include <stdlib.h>
static double pi=3.14159265358979323;
double phi(double); // phi(x)
double Phi(double); // Phi(x)
double Phi1(double); // 1-Phi(x)=Phi(-x)
double Rphi(double); // phi(x)/Phi(x)
double Rphi1(double); // phi(x)/(1-Phi(x))=phi(x)/Phi(-x)
double normal(void); // random normal variable
static double Phi_0(double);
static double Phi_1(double);
double phi(double x){return exp(-x*x/2)/sqrt(2*pi);}
double Phi(double x){
 if(x<0)return Phi1(-x);
 if(x<3.4)return Phi_0(x);
 return 1-exp(-x*x/2)/sqrt(2*pi)/Phi_1(x);
}
double Phi1(double x){
 if(x<0)return Phi(-x);
 if(x<3.4)return 1-Phi_0(x);
 return exp(-x*x/2)/sqrt(2*pi)/Phi_1(x);
}
double Rphi(double x){
 if(x<0)return Rphi1(-x);
 return exp(-x*x/2)/sqrt(2*pi)/Phi(x);
}
double Rphi1(double x){
 if(x<0)return Rphi(-x);
 if(x<3.4)return exp(-x*x/2)/sqrt(2*pi)/(1-Phi_0(x));
 return Phi_1(x);
}
double Phi_0(double x){
 double s,y,p;int n;
 s=.5;y=x*x;p=x/sqrt(2*pi);for(n=0;n<42;n++){s+=p/(2*n+1);p=-p*y/(2*(n+1));}
 return s;
}
double Phi_1(double x){
 double s;int n;
 s=x;for(n=42;n>=1;n--)s=x+n/s;
 return s;
}
double normal(void){
  static int ph=0;
  static double x,y;
  double r,th;
  if(ph==1){ph=0;return y;}
  th=drand48()*2*pi;
  r=sqrt(-2*log(drand48()));
  x=r*cos(th);y=r*sin(th);
  ph=1;return x;
}
