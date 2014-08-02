#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <getopt.h>
#include <time.h>

// Experiments with Itay Hen's planted solutions

// Chimera graph, C_n:
// Vertices are (x,y,o,i)  0<=x,y<n, 0<=o<2, 0<=i<4
// Edge from (x,y,o,i) to (x',y',o',i') if
// (x,y)=(x',y'), o!=o', OR
// |x-x'|=1, y=y', o=o'=0, i=i', OR
// |y-y'|=1, x=x', o=o'=1, i=i'
// Encode (x,y,o,i) as 8*(n*x+y)+4*o+i
#define enc(x,y,o,i) (8*(n*(x)+(y))+4*(o)+(i))
#define decx(v) ((v)/(8*n))
#define decy(v) (((v)/8)%n)
#define deco(v) (((v)>>2)&1)
#define deci(v) ((v)&3)

// Edge (x,y,i,j)_i joins (x,y,0,i) with (x,y,1,j)
// Edge (x,y,o,i)_e joins (x,y,o,i) with (x+1-o,y+o,o,i) (may not exist)
// Encode (x,y,i,j)_i as 24*(n*x+y)+4*i+j
// Encode (x,y,o,i)_e as 24*(n*x+y)+16+4*o+i
#define enci(x,y,i,j) (24*(n*(x)+(y))+4*(i)+(j))
#define ence(x,y,o,i) (24*(n*(x)+(y))+16+4*(o)+(i))

// Isolate random number generator in case we need to replace it with something better
void initrand(int seed){srandom(seed);}
int randbit(void){return (random()>>16)&1;}
int randsign(void){return randbit()*2-1;}
int randnib(void){return (random()>>16)&15;}
int randnum(void){return random();}
int randint(int n){return random()%n;}

int n,N,NE,ANE;
int minloop,maxloop;
double al;

void setupgraph(int(*vl)[6],int(*el)[6],int(*evl)[2]){
  int i,j,o,x,y,v0,v1;
  int ed[NE];
  // Setup Chimera graph
  // If vertex=v, direction=d, then unoriented edge el[v][d] connects vertex v with vertex vl[v][d]
  // evl[edge][i] = vertex at the i^th end of the edge (i=0,1)
  for(i=0;i<N;i++)for(j=0;j<6;j++)vl[i][j]=el[i][j]=-1;
  for(x=0;x<n;x++)for(y=0;y<n;y++)for(o=0;o<2;o++)for(i=0;i<4;i++){
    v0=enc(x,y,o,i);
    for(j=0;j<4;j++){
      v1=enc(x,y,1-o,j);vl[v0][j]=v1;
      if(o==0)el[v0][j]=enci(x,y,i,j); else el[v0][j]=enci(x,y,j,i);
    }
    if(o==0){
      if(x>0){v1=enc(x-1,y,o,i);vl[v0][4]=v1;el[v0][4]=ence(x-1,y,o,i);}
      if(x<n-1){v1=enc(x+1,y,o,i);vl[v0][5]=v1;el[v0][5]=ence(x,y,o,i);}
    }else{
      if(y>0){v1=enc(x,y-1,o,i);vl[v0][4]=v1;el[v0][4]=ence(x,y-1,o,i);}
      if(y<n-1){v1=enc(x,y+1,o,i);vl[v0][5]=v1;el[v0][5]=ence(x,y,o,i);}
    }
  }
  for(i=0;i<NE;i++)evl[i][0]=evl[i][1]=-1;
  for(i=0;i<N;i++)for(j=0;j<6;j++)if(vl[i][j]>=0){
    if(evl[el[i][j]][0]<0)evl[el[i][j]][0]=i; else evl[el[i][j]][1]=i;
  }

  // Checks
  memset(ed,0,sizeof(ed));
  for(i=0;i<N;i++)for(j=0;j<6;j++)if(vl[i][j]>=0){
    ed[el[i][j]]++;
    int k,l;
    k=vl[i][j];
    for(l=0;l<6;l++)if(vl[k][l]==i){assert(el[k][l]==el[i][j]);goto ok0;}
    assert(0);
    ok0:;
  }
  for(i=0,j=0;i<NE;i++){assert(ed[i]==2||ed[i]==0);j+=ed[i]/2;}
  assert(j==ANE);
  for(i=0;i<NE;i++)for(j=0;j<2;j++)assert((ed[i]==2)^(evl[i][j]<0));
}

int genloop(int(*vl)[6],int(*el)[6],int*bd,int*rvl,int*rel){
  // Input: bd = workspace array, initialised to 0
  // Output: Return value = loop length
  //         rvl = array of vertices (incl start point, excl end point)
  //         rel = array of edges (ditto)
  //         bd unchanged (still 0)
  int d,i,j,k,l,r,pr,tvl[N],tel[N];
  while(1){
    r=randint(N);i=0;
    pr=-1;
    while(1){
      do d=randint(6); while(vl[r][d]<0||vl[r][d]==pr);
      tvl[i]=r;tel[i]=el[r][d];
      i++;bd[r]=i;
      pr=r;r=vl[r][d];
      if(bd[r])break;
    }
    j=bd[r]-1;
    for(k=0;k<i;k++)bd[tvl[k]]=0;
    l=i-j;
    if(l>=minloop&&l<=maxloop){memcpy(rvl,tvl+j,l*sizeof(int));memcpy(rel,tel+j,l*sizeof(int));return l;}
  }
}

void makeinstance(int(*vl)[6],int(*el)[6]){
  int d,e,i,j,l,o,r,x,y,en,v0,v1,sp[N],jj[NE],vv[N],ee[N],bd[N];
  int nl=al*N+.5;// Number of loops
  for(i=0;i<N;i++)sp[i]=randsign();// Choose planted solution
  memset(jj,0,sizeof(jj));
  memset(bd,0,sizeof(bd));
  en=0;
  for(i=0;i<nl;i++){
    l=genloop(vl,el,bd,vv,ee);
    r=randint(l);
    for(j=0;j<l;j++)jj[ee[j]]+=-sp[vv[j]]*sp[vv[(j+1)%l]]*(1-2*(j==r));
    en+=2-l;
  }
  printf("INSTANCE\n");
  printf("%d %d %d\n",n,n,en);
  for(x=0;x<n;x++)for(y=0;y<n;y++)for(o=0;o<2;o++)for(i=0;i<4;i++)for(d=0;d<6;d++){
    v0=enc(x,y,o,i);
    e=el[v0][d];if(e<0||jj[e]==0)continue;
    v1=vl[v0][d];if(v0>v1)continue;
    printf("%2d %2d %d %d   %2d %2d %d %d   %8d\n",x,y,o,i,decx(v1),decy(v1),deco(v1),deci(v1),jj[e]);
  }
}

void loopsizestats(int(*vl)[6],int(*el)[6]){
  int i,l,it,vv[N],ee[N],bd[N],stats[N+1];
  memset(stats,0,sizeof(stats));
  memset(bd,0,sizeof(bd));
  for(it=0;it<10000000;it++){
    l=genloop(vl,el,bd,vv,ee);
    stats[l]++;
  }
  printf("%d loop%s sampled\n",it,it==1?"":"s");
  printf("Loopsize    Probability\n");
  for(i=0;i<=N;i++)if(stats[i])printf("   %5d   %12g\n",i,stats[i]/(double)it);
  double s0,s1;
  s0=s1=0;
  for(i=0;i<=N;i++)if(stats[i]){s0+=stats[i];s1+=i*stats[i];}
  printf("Average loop size = %g\n",s1/s0);
}

// Simple proportion of frustrated edges as a function of N_loops/N_qubits
void simplefrustnum(int(*vl)[6],int(*el)[6]){
  int maxnl=al*N+.5;
  int i,l,r,it,nl,nfr,vv[N],ee[N],bd[N],jj[NE];
  double s1[maxnl];
  memset(bd,0,sizeof(bd));
  for(i=0;i<maxnl;i++)s1[i]=0;
  for(it=0;it<10000;it++){
    memset(jj,0,sizeof(jj));
    nfr=0;
    for(nl=0;nl<maxnl;nl++){// nl=num loops
      l=genloop(vl,el,bd,vv,ee);
      r=randint(l);
      for(i=0;i<l;i++){
        nfr-=(jj[ee[i]]<0);
        jj[ee[i]]+=1-2*(i==r);
        nfr+=(jj[ee[i]]<0);
      }
      s1[nl]+=nfr;
    }
  }
  double maxs1,bal;
  maxs1=-1e9;bal=-1;
  for(nl=0;nl<maxnl;nl++){
    printf("%8.5f   %8.5f\n",(nl+1)/(double)N,s1[nl]/it/ANE);
    if(s1[nl]>maxs1){maxs1=s1[nl];bal=(nl+1.)/N;}
  }
  printf("Hardest N_loops/N_qubits %g\n",bal);
}

// Weighted proportion of frustrated edges as a function of N_loops/N_qubits
void weightedfrustnum(int(*vl)[6],int(*el)[6],int(*evl)[2]){
  int maxnl=al*N+.5;
  int e,i,j,l,r,it,na,nl,bd[N],vv[N],ee[N],jj[NE],ael[6*N],aea[NE];
  double nfr,s1[maxnl];
  memset(bd,0,sizeof(bd));
  // ael = affected edge list, na=number affected
  memset(aea,0,sizeof(aea));// Affected edge array
  for(i=0;i<maxnl;i++)s1[i]=0;

  double weight(int e){
    if(jj[e]>=0)return 0;
    int i,j,k,m,s,v,i0,i1,j0,j1,e1,n1,s1,en,ben,nfr,nuf,jl[5],jd[2][32],md[2][32],nd[2];
    double f0,f1;
    int cmpi(const void*p,const void*q){return (*(int*)p)-(*(int*)q);}
    for(i=0;i<2;i++){// end of edge
      v=evl[e][i];assert(v>=0);
      for(j=0,n1=0;j<6;j++){
        e1=el[v][j];
        if(e1>=0&&e1!=e)jl[n1++]=abs(jj[e1]);
      }
      assert(n1<=5);
      for(j=0,s1=0;j<n1;j++)s1+=jl[j];
      for(j=0;j<(1<<n1);j++){
        for(k=0,s=0;k<n1;k++)s+=(1-2*((j>>k)&1))*jl[k];
        jd[i][j]=s;
      }
      qsort(jd[i],1<<n1,sizeof(int),cmpi);
      //for(j=0;j<(1<<n1);j++)printf("%d ",jd[i][j]);printf("\n");
      for(j=1,nd[i]=0,m=1;j<=(1<<n1);j++){
        if(j==(1<<n1)||jd[i][j]!=jd[i][j-1]){jd[i][nd[i]]=jd[i][j-1];md[i][nd[i]++]=m;m=0;}
        m++;
      }
      //for(j=0;j<nd[i];j++)printf("%d*%d ",md[i][j],jd[i][j]);printf("\n");
      //printf("\n");
    }
    f0=f1=0;
    for(i0=0;i0<nd[0];i0++)for(i1=0;i1<nd[1];i1++){
      ben=1000000000;
      for(j0=-1;j0<=1;j0+=2)for(j1=-1;j1<=1;j1+=2){
        en=jd[0][i0]*j0+j0*jj[e]*j1+j1*jd[1][i1];
        if(en<ben){ben=en;nfr=nuf=0;}
        if(en==ben){nfr+=(j0==j1);nuf+=(j0!=j1);}
      }
      f0+=md[0][i0]*md[1][i1];
      f1+=md[0][i0]*md[1][i1]*nfr/(double)(nfr+nuf);
    }
    return f1/f0;
  }

  for(it=0;it<10000;it++){
    memset(jj,0,sizeof(jj));
    nfr=0;
    for(nl=0;nl<maxnl;nl++){// nl=num loops
      l=genloop(vl,el,bd,vv,ee);
      r=randint(l);
      na=0;// Optimisation: only reevaluate possibly affected edges
      for(i=0;i<l;i++)for(j=0;j<6;j++){
        e=el[vv[i]][j];
        if(e>=0){
          if(aea[e]==0){ael[na++]=e;aea[e]=1;}
        }
      }
      for(i=0;i<na;i++)nfr-=weight(ael[i]);
      for(i=0;i<l;i++)jj[ee[i]]+=1-2*(i==r);
      for(i=0;i<na;i++)nfr+=weight(ael[i]);
      for(i=0;i<na;i++)aea[ael[i]]=0;
      s1[nl]+=nfr;
    }
  }
  double maxs1,bal;
  maxs1=-1e9;bal=-1;
  for(nl=0;nl<maxnl;nl++){
    printf("%8.5f   %8.5f\n",(nl+1.)/N,s1[nl]/it/ANE);
    if(s1[nl]>maxs1){maxs1=s1[nl];bal=(nl+1.)/N;}
  }
  printf("Hardest N_loops/N_qubits %g\n",bal);
}

int main(int ac,char**av){
  int mode,opt,seed;
  n=8;mode=1;minloop=8;maxloop=1000000;seed=time(0);al=2;
  while((opt=getopt(ac,av,"a:b:m:n:p:s:"))!=-1){
    switch(opt){
    case 'a':minloop=atoi(optarg);break;
    case 'b':maxloop=atoi(optarg);break;
    case 'm':mode=atoi(optarg);break;
    case 'n':n=atoi(optarg);break;
    case 'p':al=atof(optarg);break;
    case 's':seed=atoi(optarg);break;
    default:
      fprintf(stderr,"Usage: %s [OPTIONS]\n",av[0]);
      fprintf(stderr,"       -a   minlooplength (default 8)\n");
      fprintf(stderr,"       -b   maxlooplength (default 1000000)\n");
      fprintf(stderr,"       -n   Chimera dimensions (default 8)\n");
      fprintf(stderr,"       -m   mode of operation:\n");
      fprintf(stderr,"            0   Generate an instance\n");
      fprintf(stderr,"            1   Loop size stats\n");
      fprintf(stderr,"            2   Proportion of simple frustration count\n");
      fprintf(stderr,"            3   Proportion of weighted frustration count\n");
      fprintf(stderr,"       -p   N_loops/N_qubits, or max N_loops/N_qubits, according to context\n");
      fprintf(stderr,"       -s   seed (default time)\n");
      exit(1);
    }
  }
  N=8*n*n;// Number of vertices (qubits)
  NE=24*n*n;// Edge numbers go up to NE for convenience, but some of these edges don't exist
  ANE=8*n*(3*n-1);// Actual number of edges
  int vl[N][6],el[N][6];// vl=vertex at end of edge, el=edge number of corresponding edge; -1 means non-existent
  int evl[NE][2];// map from edge to vertices at endpoints (-1 means non-existent edge)

  printf("Chimera dimensions %d\n",n);
  printf("Chimera vertices %d\n",N);
  printf("Chimera edges %d\n",ANE);
  printf("Min loop size %d\n",minloop);
  printf("Max loop size %d\n",maxloop);
  printf("N_loop/N_qubits %g\n",al);
  printf("Mode %d\n",mode);
  printf("Seed %d\n",seed);initrand(seed);
  printf("\n");

  setupgraph(vl,el,evl);

  switch(mode){
  case 0:makeinstance(vl,el);break;
  case 1:loopsizestats(vl,el);break;
  case 2:simplefrustnum(vl,el);break;
  case 3:weightedfrustnum(vl,el,evl);break;
  default:fprintf(stderr,"Unrecognised mode %d\n",mode);exit(1);
  }

  return 0;
}
