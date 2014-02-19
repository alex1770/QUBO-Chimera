#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <getopt.h>
#include <assert.h>
#ifdef PARALLEL
#include <omp.h>
#endif

#include "normal.c"

// QUBO solver
// Solves QUBO problem:
// Minimise sum_{i,j} Q_ij x_i x_j over choices of x_i
// i,j corresponds to an edge from i to j in the "Chimera" graph C_N.
// The sum is taken over both directions (i,j and j,i) and includes the diagonal terms
// (configurable using option -w).
// x_i can take the values statemap[0] and statemap[1] (default 0,1).
// This includes the case described in section 3.2 of http://www.cs.amherst.edu/ccm/cf14-mcgeoch.pdf
//
// This now includes the union of all historical test code. To illustrate any particular
// techinque probably only requires a small subset of this program. In other words, it
// could do with a clean out and separating into different programs.
// 
// Chimera graph, C_N:
// Vertices are (x,y,o,i)  0<=x,y<N, 0<=o<2, 0<=i<4
// Edge from (x,y,o,i) to (x',y',o',i') if
// (x,y)=(x',y'), o!=o', OR
// |x-x'|=1, y=y', o=o'=0, i=i', OR
// |y-y'|=1, x=x', o=o'=1, i=i'
// 
// x,y are the horizontal,vertical co-ords of the K4,4
// o=0..1 is the "orientation" (0=horizontally connected, 1=vertically connected)
// i=0..3 is the index within the "semi-K4,4"="bigvertex"
// There is an involution given by {x<->y o<->1-o}
//

#define NV (8*N*N)        // Num vertices
#define NE (8*N*(3*N-1))  // Num edges (not used)
#define NBV (2*N*N)       // Num "big" vertices (semi-K4,4s)
#define NBE (N*(3*N-2))   // Num "big" edges (not used)
#define enc(x,y,o) ((o)+((N*(x)+(y))<<1))
#define encp(x,y,o) ((x)>=0&&(x)<N&&(y)>=0&&(y)<N?enc(x,y,o):NBV) // bounds-protected version
#define decx(p) (((p)>>1)/N)
#define decy(p) (((p)>>1)%N)
#define deco(p) ((p)&1)
// encI is the same as enc but incorporates the involution x<->y, o<->1-o
#define encI(inv,x,y,o) (((inv)^(o))+((N*(x)+(y)+(inv)*(N-1)*((y)-(x)))<<1))
//#define encI(inv,x,y,o) ((inv)?enc(y,x,1-(o)):enc(x,y,o))
//#define encI(inv,x,y,o) (enc(x,y,o)+(inv)*(enc(y,x,1-(o))-enc(x,y,o)))
#define enc2(x,y) (N*(x)+(y))
#define enc2p(x,y) ((x)>=0&&(x)<N&&(y)>=0&&(y)<N?enc2(x,y):N*N) // bounds-protected version

int (*Q)[4][7]; // Q[NBV][4][7]
                // Weights: Q[r][i][d] = weight of i^th vertex of r^th big vertex in direction d
                // Directions 0-3 corresponds to intra-K_4,4 neighbours, and
                // 4 = Left or Down, 5 = Right or Up, 6 = self
int (*adj)[4][7][2]; // adj[NBV][4][7][2]
                     // Complete adjacency list, including both directions along an edge and self-loops
                     // adj[p][i][d]={q,j} <-> d^th neighbour of encoded vertex p, index i, is 
                     //                        encoded vertex q, index j                    
                     // d as above
int (*okv)[4]; // okv[NBV][4] list of working vertices
int *XBplus; // XBplus[(N+2)*N*2]
int *XBa; // XBa[NBV]
          // XBa[enc(x,y,o)] = State (0..15) of big vert
          // Allow extra space to avoid having to check for out-of-bounds accesses
          // (Doesn't matter that they wrap horizontally, since the weights will be 0 for these edges.)
int (*QBa)[3][16][16]; // QBa[NBV][3][16][16]
                       // Weights for big verts (derived from Q[])
                       // QBa[enc(x,y,o)][d][s0][s1] = total weight from big vert (x,y,o) in state s0
                       //                              to the big vert in direction d in state s1
                       // d=0 is intra-K_4,4, d=1 is Left/Down, d=2 is Right/Up
int (*ok)[16]; // ok[NBV+1][16]   ok[enc(x,y,o)][s] = s^th allowable state in cell x,y,o (list)
int *nok;      // nok[NBV+1]      nok[enc(x,y,o)] = number of allowable states in x,y,o
               // The last entry is single state entry which is used when things go outside the grid
int (*ok2)[256];// ok2[N*N+1][256]  ok2[enc2(x,y)][s] = s^th allowable state in K44 x,y (list)
int *nok2;      // nok2[N*N+1]      nok2[enc2(x,y)] = number of allowable states in K44 x,y
#define QB(x,y,o,d,s0,s1) (QBa[enc(x,y,o)][d][s0][s1])
#define QBI(inv,x,y,o,d,s0,s1) (QBa[encI(inv,x,y,o)][d][s0][s1])// Involution-capable addressing of QB
#define XB(x,y,o) (XBa[enc(x,y,o)])
#define XBI(inv,x,y,o) (XBa[encI(inv,x,y,o)])// Involution-capable addressing of XB

int N;// Size of Chimera graph
int statemap[2];// statemap[0], statemap[1] are the two possible values that the state variables take
int deb;// verbosity
int seed;
double ext;

#define MAXNGP 100
int ngp;// Number of general parameters
double genp[MAXNGP]={0}; // General parameters

typedef long long int int64;// gcc's 64-bit type
typedef unsigned char UC;

#define NTB 1024
int ps[NTB][256];
#define MIN(x,y) ((x)<(y)?(x):(y))
#define MAX(x,y) ((x)>(y)?(x):(y))

// Isolate random number generator in case we need to replace it with something better
void initrand(int seed){srandom(seed);}
int randbit(void){return (random()>>16)&1;}
int randsign(void){return randbit()*2-1;}
int randnib(void){return (random()>>16)&15;}
int randint(int n){return random()%n;}
double randfloat(void){return (random()+.5)/(RAND_MAX+1.0);}

double cpu(){return clock()/(double)CLOCKS_PER_SEC;}

void initgraph(int wn){
  int d,i,j,o,p,t,u,x,y,z;
  for(p=0;p<NBV;p++)for(i=0;i<4;i++)for(d=0;d<7;d++)adj[p][i][d][0]=adj[p][i][d][1]=-1;// Set "non-existent" flag
  for(x=0;x<N;x++)for(y=0;y<N;y++)for(o=0;o<2;o++){
    p=enc(x,y,o);
    for(i=0;i<4;i++){
      for(j=0;j<4;j++){adj[p][i][j][0]=enc(x,y,1-o);adj[p][i][j][1]=j;}
      z=o?y:x;
      if(z>0){adj[p][i][4][0]=enc(x-1+o,y-o,o);adj[p][i][4][1]=i;}
      if(z<N-1){adj[p][i][5][0]=enc(x+1-o,y+o,o);adj[p][i][5][1]=i;}
      adj[p][i][6][0]=p;adj[p][i][6][1]=i;
    }
  }
  // Choose random subset of size wn to be the working nodes
  t=wn;u=NV;
  for(p=0;p<NBV;p++)for(i=0;i<4;i++){okv[p][i]=(randint(u)<t);t-=okv[p][i];u--;}
}

void getbigweights1(void){// Get derived weights on "big graph" QB[] from Q[]
  // Optimised version of getbigweights()
  // This (messier) version is just here to show that the setup time can be more-or-less negligible.
  // Could be faster if we optimised for the case that statemap={0,1} or {-1,1}, but it's fast enough for now.
  int i,j,o,p,q,v,x,y,po,s0,s1,x0,x1,x00,x0d,dd,dd2;
  memset(QBa,0,NBV*3*16*16*sizeof(int));
  x0=statemap[0];x1=statemap[1];
  x00=x0*x0;x0d=x0*(x1-x0);dd=(x1-x0)*(x1-x0);dd2=x1*x1-x0*x0;
  for(x=0;x<N;x++)for(y=0;y<N;y++){
    int (*QBal)[16],vv[16][16];
    p=enc(x,y,0);po=enc(x,y,1);
    for(i=0,v=0;i<4;i++)for(j=0;j<4;j++){vv[i][j]=Q[p][i][j]+Q[po][j][i];v+=vv[i][j];}
    for(i=0;i<4;i++)v+=Q[p][i][6]+Q[po][i][6];
    v*=x00;
    QBal=QBa[p][0];
    QBal[0][0]=v;
    for(i=0;i<4;i++){
      QBal[1<<i][0]=v+(vv[i][0]+vv[i][1]+vv[i][2]+vv[i][3])*x0d+Q[p][i][6]*dd2;
      QBal[0][1<<i]=v+(vv[0][i]+vv[1][i]+vv[2][i]+vv[3][i])*x0d+Q[po][i][6]*dd2;
    }
    for(i=1;i<4;i++)for(s0=(1<<i)+1;s0<(1<<(i+1));s0++){
      QBal[0][s0]=QBal[0][1<<i]+QBal[0][s0-(1<<i)]-v;
      QBal[s0][0]=QBal[1<<i][0]+QBal[s0-(1<<i)][0]-v;
    }
    for(i=0;i<4;i++)for(j=0;j<4;j++)QBal[1<<i][1<<j]=QBal[1<<i][0]+QBal[0][1<<j]-v+dd*vv[i][j];
    for(i=0;i<4;i++)for(s0=(1<<i);s0<(1<<(i+1));s0++){
      for(j=0;j<4;j++){
        QBal[s0][1<<j]=QBal[1<<i][1<<j]+QBal[s0-(1<<i)][1<<j]-QBal[0][1<<j];
        for(s1=(1<<j)+1;s1<(1<<(j+1));s1++){
          QBal[s0][s1]=QBal[s0][1<<j]+QBal[s0][s1-(1<<j)]-QBal[s0][0];
        }
      }
    }
    for(o=0;o<2;o++)if((o?y:x)<N-1){
      int aa[16],oo[16];
      p=enc(x,y,o);
      q=enc(x+1-o,y+o,o);
      for(i=0,v=0;i<4;i++)v+=Q[p][i][5]+Q[q][i][4];
      v*=x00;
      aa[0]=0;oo[0]=v;
      for(i=0;i<4;i++){
        aa[1<<i]=(Q[p][i][5]+Q[q][i][4])*x1*(x1-x0);
        oo[1<<i]=v+(Q[p][i][5]+Q[q][i][4])*x0*(x1-x0);
        for(s0=(1<<i)+1;s0<(1<<(i+1));s0++){
          aa[s0]=aa[1<<i]+aa[s0-(1<<i)];
          oo[s0]=oo[1<<i]+oo[s0-(1<<i)]-v;
        }
      }
      for(s0=0;s0<16;s0++)for(s1=0;s1<16;s1++)QBa[p][2][s0][s1]=aa[s0&s1]+oo[s0|s1];
    }
  }// x,y
  for(x=0;x<N;x++)for(y=0;y<N;y++){
    if(x<N-1)memcpy(QBa[enc(x+1,y,0)][1],QBa[enc(x,y,0)][2],256*sizeof(int));
    if(y<N-1)memcpy(QBa[enc(x,y+1,1)][1],QBa[enc(x,y,1)][2],256*sizeof(int));
    p=enc(x,y,1);q=enc(x,y,0);
    for(s0=0;s0<16;s0++)for(s1=0;s1<16;s1++)QBa[p][0][s0][s1]=QBa[q][0][s1][s0];
  }
}

void getbigweights(void){// Get derived weights on "big graph" QB[] from Q[]
  // Intended so that the energy is calculated by summing over each big-edge exactly once,
  // not forwards and backwards.  See val() below.
  // That means that the off-diagonal bit of Q[][] has to be replaced by Q+Q^T, not
  // (1/2)(Q+Q^T) as would happen if you later intended to sum over both big-edge directions.
  // The self-loops are incorporated (once) into the intra-K_4,4 terms, QB(*,*,*,0,*,*).
  int d,i,j,k,o,p,q,x,y,po,s0,s1,x0,x1;
  getbigweights1();return;// Call equivalent optimised version and return.
  // Simple version below retained because it makes it clearer what is going on.
  for(x=0;x<N;x++)for(y=0;y<N;y++)for(o=0;o<2;o++)for(s0=0;s0<16;s0++)for(s1=0;s1<16;s1++){
    for(k=0;k<3;k++)QB(x,y,o,k,s0,s1)=0;
    p=enc(x,y,o);po=enc(x,y,1-o);
    for(i=0;i<4;i++)for(d=0;d<7;d++){
      q=adj[p][i][d][0];j=adj[p][i][d][1];
      if(q>=0){
        x0=statemap[(s0>>i)&1];x1=statemap[(s1>>j)&1];
        if(d<4)QB(x,y,o,0,s0,s1)+=(Q[p][i][j]+Q[po][j][i])*x0*x1;
        if(d==6)QB(x,y,o,0,s0,s1)+=Q[p][i][6]*x0*x0+Q[po][j][6]*x1*x1;
        if(d==4)QB(x,y,o,1,s0,s1)+=(Q[p][i][4]+Q[q][j][5])*x0*x1;
        if(d==5)QB(x,y,o,2,s0,s1)+=(Q[p][i][5]+Q[q][j][4])*x0*x1;
      }
    }
  }
}

int val(void){// Calculate value (energy)
  int v,x,y;
  v=0;
  for(x=0;x<N;x++)for(y=0;y<N;y++){
    v+=QB(x,y,0,0,XB(x,y,0),XB(x,y,1));
    v+=QB(x,y,0,2,XB(x,y,0),XB(x+1,y,0));
    v+=QB(x,y,1,2,XB(x,y,1),XB(x,y+1,1));
  }
  return v;
}

int stripval(int d,int c0,int c1){
  // If d=0, get value of columns c0..(c1-1), not including external edges
  // If d=1 then same for rows c0..(c1-1)
  int v,x,y;
  v=0;
  for(x=c0;x<c1;x++)for(y=0;y<N;y++){
    v+=QBI(d,x,y,0,0,XBI(d,x,y,0),XBI(d,x,y,1));
    if(x<c1-1)v+=QBI(d,x,y,0,2,XBI(d,x,y,0),XBI(d,x+1,y,0));
    v+=QBI(d,x,y,1,2,XBI(d,x,y,1),XBI(d,x,y+1,1));
  }
  return v;
}

void initweights(int weightmode){// Randomly initialise a symmetric weight matrix
  // weightmode
  // 0           All of Q_ij independently +/-1
  // 1           As 0, but diagonal not allowed
  // 2           Upper triangular
  // 3           All of Q_ij allowed, but constrained symmetric
  // 4           Constrained symmetric, diagonal not allowed
  // 5           Start with J_ij (i<j) and h_i IID {-1,1} and transform back to Q (ignoring constant term)
  int d,i,j,p,q,r;
  for(p=0;p<NBV;p++)for(i=0;i<4;i++)for(d=0;d<7;d++)Q[p][i][d]=0;
  for(p=0;p<NBV;p++)for(i=0;i<4;i++)for(d=0;d<7;d++){
    q=adj[p][i][d][0];j=adj[p][i][d][1];
    if(!(q>=0&&okv[p][i]&&okv[q][j]))continue;
    switch(weightmode){
    case 0:Q[p][i][d]=randsign();break;
    case 1:if(d<6)Q[p][i][d]=randsign();break;
    case 2:if((d<4&&deco(p)==0)||d==5)Q[p][i][d]=randsign();break;
    case 3:if((d<4&&deco(p)==0)||d==5)Q[p][i][d]=2*randsign(); else if(d==6)Q[p][i][d]=randsign(); break;
    case 4:if((d<4&&deco(p)==0)||d==5)Q[p][i][d]=2*randsign();break;
    case 5:if((d<4&&deco(p)==0)||d==5){r=randsign();Q[p][i][d]=4*r;Q[p][i][6]-=2*r;Q[q][j][6]-=2*r;}
      else if(d==6)Q[p][i][d]+=2*randsign();
      break;
    case 6:if((d<4&&deco(p)==0)||d==5){r=randsign()*(10+(seed%32)*(d>=4));Q[p][i][d]=2*r;Q[p][i][6]-=r;Q[q][j][6]-=r;}
      break;
      // mode 7 is "noextfield" (J_{ij}=+/-1, h_i=0) in QUBO form
    case 7:if((d<4&&deco(p)==0)||d==5){r=randsign();Q[p][i][d]=2*r;Q[p][i][6]-=r;Q[q][j][6]-=r;}
      break;
      // mode 8 is a test mode
    case 8:if((d<4&&deco(p)==0)||d==5){int n=100+20*(seed%10)*(d>=4);r=randint(2*n+1)-n;Q[p][i][d]=2*r;Q[p][i][6]-=r;Q[q][j][6]-=r;}
      break;
      // mode 9 is candidate for most difficult class of instances at N=8 (tested with strat 13)
    case 9:if((d<4&&deco(p)==0)||d==5){int n=100+100*(d>=4);r=randint(2*n+1)-n;Q[p][i][d]=2*r;Q[p][i][6]-=r;Q[q][j][6]-=r;}
      break;
      // mode 10 is candidate for most difficult class of instances at N=16 (tested with strat 14)
    case 10:if((d<4&&deco(p)==0)||d==5){int n=100+120*(d>=4);r=randint(2*n+1)-n;Q[p][i][d]=2*r;Q[p][i][6]-=r;Q[q][j][6]-=r;}
      break;
      // mode 11 is uniform on {-n,...,-1,1,...,n} (converted to QUBO form) to mimic "range" of http://arxiv.org/abs/1401.2910
    case 11:if((d<4&&deco(p)==0)||d==5){int n=7;r=randint(2*n)-n;r+=(r>=0);Q[p][i][d]=2*r;Q[p][i][6]-=r;Q[q][j][6]-=r;}
      break;
    }
  }
  getbigweights();
}

void writeweights(char *f){
  int d,i,j,p,q;
  FILE *fp;
  fp=fopen(f,"w");assert(fp);
  fprintf(fp,"%d %d\n",N,N);
  for(p=0;p<NBV;p++)for(i=0;i<4;i++)for(d=0;d<7;d++){
    q=adj[p][i][d][0];j=adj[p][i][d][1];
    if(q>=0&&Q[p][i][d]!=0)fprintf(fp,"%d %d %d %d   %d %d %d %d   %8d\n",
                                   decx(p),decy(p),deco(p),i,
                                   decx(q),decy(q),deco(q),j,
                                   Q[p][i][d]);
  }
  fclose(fp);
}

int readweights(char *f){
  int d,i,n,p,w,v0,v1,x0,y0,o0,i0,e0,x1,y1,o1,i1,nx,ny,wn,gtr;
  char l[1000];
  FILE *fp;
  printf("Reading weight matrix from file \"%s\"\n",f);
  fp=fopen(f,"r");assert(fp);
  while(fgets(l,1000,fp))if(l[0]!='#')break;
  n=sscanf(l,"%d %d %d",&nx,&ny,&gtr);assert(n>=2);if(n==2)gtr=1000000;// gtr=ground truth (not currently used)
  assert(nx==N&&ny==N);
  // Ensure weights=0 for edges that go out of bounds
  for(p=0;p<NBV;p++)for(i=0;i<4;i++){okv[p][i]=0;for(d=0;d<7;d++)Q[p][i][d]=0;}
  while(fgets(l,1000,fp)){
    if(l[0]=='#')continue;
    assert(sscanf(l,"%d %d %d %d %d %d %d %d %d",
                  &x0,&y0,&o0,&i0,
                  &x1,&y1,&o1,&i1,
                  &w)==9);
    if(x1==x0&&y1==y0){
      if(o0==o1)e0=6; else e0=i1;
    }else{
      if(abs(x1-x0)==1&&y1==y0&&o0==0&&o1==0){e0=4+(x1-x0+1)/2;}else
        if(x1==x0&&abs(y1-y0)==1&&o0==1&&o1==1){e0=4+(y1-y0+1)/2;}else
          {fprintf(stderr,"Unexpected edge in line: %s",l);assert(0);}
    }
    v0=enc(x0,y0,o0);
    v1=enc(x1,y1,o1);
    Q[v0][i0][e0]=w;
    if(w)okv[v0][i0]=okv[v1][i1]=1;
  }
  fclose(fp);
  for(p=0,wn=0;p<NBV;p++)for(i=0;i<4;i++)wn+=okv[p][i];
  getbigweights();
  return wn;
}

void prstate(FILE*fp,int style,int*X0){
  // style = 0: hex grid xored with X0 (if supplied)
  // style = 1: hex grid xored with X0 (if supplied), "gauge-fixed"
  // style = 2: list of vertices
  int nb[16];
  int i,j,o,p,t,x,xor;
  x=xor=0;
  if(style==1){
    xor=-1;
    if(statemap[0]==-1){
      for(i=1,nb[0]=0;i<16;i++)nb[i]=nb[i>>1]+(i&1);
      for(i=0,t=0;i<N;i++)for(j=0;j<N;j++)for(o=0;o<2;o++)t+=nb[XB(i,j,o)^(X0?X0[enc(i,j,o)]:0)];
      x=(t>=NV/2?15:0);
    }
  }
  if(style<2){
    for(j=N-1;j>=0;j--){
      for(i=0;i<N;i++){fprintf(fp," ");for(o=0;o<2;o++)fprintf(fp,"%X",XB(i,j,o)^((X0?X0[enc(i,j,o)]:0)&xor)^x);}
      fprintf(fp,"\n");
    }
  }else{
    for(p=0;p<NBV;p++)for(i=0;i<4;i++)fprintf(fp,"%d %d %d %d  %4d\n",decx(p),decy(p),deco(p),i,statemap[(XBa[p]>>i)&1]);
  }
}

void writestate(char *f){
  FILE *fp;
  fp=fopen(f,"w");assert(fp);
  prstate(fp,2,0);
  fclose(fp);
}

void readstate(char *f){
  int s,x,y,o,i;
  char l[1000];
  FILE *fp;
  printf("Reading state from file \"%s\"\n",f);
  fp=fopen(f,"r");assert(fp);
  memset(XBa,0,NBV*sizeof(int));
  while(fgets(l,1000,fp)){
    assert(sscanf(l,"%d %d %d %d %d",&x,&y,&o,&i,&s)==5);
    assert(s==statemap[0]||s==statemap[1]);
    XB(x,y,o)|=(s==statemap[1])<<i;
  }
  fclose(fp);
}

void shuf(int*a,int n){
  int i,j,t;
  for(i=0;i<n-1;i++){
    j=i+randint(n-i);t=a[i];a[i]=a[j];a[j]=t;
  }
}

void inittiebreaks(){
  int i,j;
  for(i=0;i<NTB;i++){
    for(j=0;j<256;j++)ps[i][j]=j;
    shuf(ps[i],256);
  }
}

int stripexhaust(int d,int c0,int c1,int upd){
  // If d=0 exhaust columns c0..(c1-1), if d=1 exhaust rows c0..(c1-1)
  // Comments and variable names are as if in the column case (d=0)
  // upd=1 <-> the optimum is written back into the global state
  // Returns value of strip only

  int c,r,s,t,v,x,bc,sh,wid,smin,vmin;
  int64 b,M,bl,br;
  short*vv,v1[16];// Map from boundary state to value of interior
  wid=c1-c0;
  M=1LL<<4*wid;
  int ps[wid][16],vc[wid][16],h1[16];
  UC (*hc)[wid][M]=0,// Comb history: hc[r][x][b] = opt value of (c0+x,r,0) given (c0+y,r,I(y<=x))=b   (y=0,...,wid-1)
    (*hs)[wid][M]=0; // Strut history: hs[r][x][b] = opt value of (c0+x,r,1) given (c0+y,r+I(y<=x),1)=b   (y=0,...,wid-1)
  vv=(short*)malloc(M*sizeof(short));
  if(upd){
    hc=(UC (*)[wid][M])malloc(N*wid*M);
    hs=(UC (*)[wid][M])malloc(N*wid*M);
  }
  if(!(vv&&(!upd||(hc&&hs)))){fprintf(stderr,"Couldn't allocate %gGiB in stripexhaust()\n",
                                      M*(sizeof(short)+(!!upd)*2.*N*wid)/(1<<30));return 1;}
  // Break ties randomly. Sufficient to choose fixed tiebreaker outside r,b loops
  for(x=0;x<wid;x++){for(s=0;s<16;s++)ps[x][s]=s;if(upd)shuf(ps[x],16);}
  memset(vv,0,M*sizeof(short));
  // Encoding of boundary into b is that the (c0+y,?,?) term corresponds to nibble y
  for(r=0;r<N;r++){
    // Comb exhaust
    // At this point: vv maps (*,r,1) to value of (*,r,1), (*,<r,*)
    //      
    //        *b0       *b1       *b2
    //       /         /         /
    //      /         /         /
    //     *---------*---------*
    //     s0        s1        s2
    //
    // b2{
    //   s1{ vc1[s1]=min_{s2} Q(s3ext,s2)+Q(s2,b2)+Q(s2,s1) }
    //   b1{
    //     s0{ vc0[s0]=min_{s1} vc1[s1]+Q(s1,b1)+Q(s1,s0) }
    //     b0{
    //       vv[b2][b1][b0]+=min_{s0} vc0[s0]+Q(s0,b0)+Q(s0,s-1ext)
    //     }
    //   }
    // }

    for(s=0;s<16;s++)vc[wid-1][s]=QBI(d,c1-1,r,0,2,s,XBI(d,c1,r,0));
    x=wid-1;b=0;
    while(x<wid){
      // At this point b has (at least) x*4 zeros at the end of its binary expansion
      if(x==0){
        t=XBI(d,c0-1,r,0);
        vmin=1000000000;smin=-1;
        for(s=0;s<16;s++){// s=s_x
          v=(vc[0][s]+QBI(d,c0,r,0,0,s,b&15)+QBI(d,c0,r,0,1,s,t))<<4|ps[x][s];
          if(v<vmin){vmin=v;smin=s;}
        }
        vv[b]+=vmin>>4;
        if(upd)hc[r][x][b]=smin;
        b++;
        while((b>>x*4&15)==0)x++;
      }else{
        for(t=0;t<16;t++){// t=s_{x-1}
          vmin=1000000000;smin=-1;
          for(s=0;s<16;s++){// s=s_x
            v=(vc[x][s]+QBI(d,c0+x,r,0,0,s,b>>x*4&15)+QBI(d,c0+x,r,0,1,s,t))<<4|ps[x][s];
            if(v<vmin){vmin=v;smin=s;}
          }
          vc[x-1][t]=vmin>>4;
          if(upd)hc[r][x][b+t]=smin;
        }
        x--;
      }
    }
    // At this point vv maps (*,r,1) to value of (*,<=r,*)

    // Strut exhaust
    //
    //     *         *b1       *b2       *b3
    //     |         |         |         |
    //     |         ^         |         |
    //     |         |         |         |
    //     |         |         |         |
    //     *b0       *s1       *         *  
    //
    // (c=1 picture)
    
    for(x=wid-1;x>=0;x--){
      c=c0+x;
      // At this point vv maps (<=c,r,1), (>c,r+1,1) to value below these vertices
      sh=x*4;
      for(br=0;br<M;br+=1LL<<(sh+4)){// br = state of (>c,r+1,1)
        if(r==N-1&&br>0)continue;
        for(bl=0;bl<1LL<<sh;bl++){// bl = state of (<c,r,1)
          b=bl+br;
          for(bc=0;bc<16;bc++){
            //if(r==N-1&&bc>0)continue;// This optimisation appears to slow it down
            vmin=1000000000;smin=-1;
            for(s=0;s<16;s++){// s = state of (c,r,1)
              v=((vv[b+((int64)s<<sh)]+QBI(d,c,r,1,2,s,bc))<<4)|ps[x][s];
              if(v<vmin){vmin=v;smin=s;}
            }
            v1[bc]=vmin>>4;
            h1[bc]=smin;
          }
          for(bc=0;bc<16;bc++){
            int64 b1=b+((int64)bc<<sh);
            vv[b1]=v1[bc];
            if(upd)hs[r][x][b1]=h1[bc];
          }
        }
      }
    }
    // Now vv maps (*,r+1,1) to value of (*,r+1,1),(*,<=r,*)
  }//r
  if(upd){
    b=0;
    for(r=N-1;r>=0;r--){
      // Now b = opt value of (*,r+1,1)
      for(x=0;x<wid;x++){
        sh=x*4;
        s=hs[r][x][b];XBI(d,c0+x,r,1)=s;
        b=(b&~(15LL<<sh))|(int64)s<<sh;
      }
      // Now b = opt value of (*,r,1)
      s=0;
      for(x=0;x<wid;x++){
        sh=x*4;
        s=hc[r][x][(b&(-(1LL<<sh)))+s];
        XBI(d,c0+x,r,0)=s;
      }      
    }
    free(hs);free(hc);
  }
  v=vv[0];
  free(vv);
  return v;//+stripval(d,0,c0)+stripval(d,c1,N);
}

int lineexhaust(int d,int c,int upd){return stripexhaust(d,c,c+1,upd);}

int k44exhaust(int x,int y){
  // Exhausts big vertex (x,y)
  // Writes optimum value back into the global state
  int t,v,s0,s1,v0,vmin,tv[16];
  t=randint(NTB);// Random tiebreaker
  for(s1=0;s1<16;s1++)tv[s1]=QB(x,y,1,1,s1,XB(x,y-1,1))+QB(x,y,1,2,s1,XB(x,y+1,1));
  vmin=1000000000;
  for(s0=0;s0<16;s0++){
    v0=QB(x,y,0,1,s0,XB(x-1,y,0))+QB(x,y,0,2,s0,XB(x+1,y,0));
    for(s1=0;s1<16;s1++){
      v=((QB(x,y,0,0,s0,s1)+v0+tv[s1])<<8)|ps[t][s0+(s1<<4)];
      if(v<vmin){vmin=v;XB(x,y,0)=s0;XB(x,y,1)=s1;}
    }
  }
  return vmin>>8;
}

void init_state(void){// Initialise state randomly
  int x,y,o;
  for(x=0;x<N;x++)for(y=0;y<N;y++)for(o=0;o<2;o++)XB(x,y,o)=randnib();
}

void pertstate(double p){
  int o,x,y;
  for(x=0;x<N;x++)for(y=0;y<N;y++)if(randfloat()<p)for(o=0;o<2;o++)XB(x,y,o)=randnib();
}

int tree1exhaust(int d,int p,int r0,int upd){
  // If d=0 exhaust the (induced) tree consisting of all columns of parity p,
  // the o=1 (vertically connected) verts of the other columns, and row r0.
  // If d=1 then same with rows <-> columns.
  // Comments and variable names are as if in the column case (d=0)
  // upd=1 <-> the optimum is written back into the global state
  // Returns value of tree, which is global value because tree contains or is adjacent to every edge
  int b,c,f,r,s,v,dir,smin,vmin,ps[16],v0[16],v1[16],v2[16],v3[16],v4[16],hc[N][N][16],hs[N][N][16],hr[N][16];
  // v0[s] = value of current column fragment given that (c,r,1) = s
  // v2[s] = value of current column (apart from (c,r0,0)) given that (c,r0,1) = s
  // v3[s] = value of tree to left of column c given that (c,r0,0) = s

  for(s=0;s<16;s++)v3[s]=0;
  for(c=0;c<N;c++){
    for(s=0;s<16;s++)ps[s]=s;
    if(upd)shuf(ps,16);

    for(s=0;s<16;s++)v2[s]=0;
    for(dir=0;dir<2;dir++){// dir=0 <-> increasing r, dir=1 <-> decreasing r
      for(s=0;s<16;s++)v0[s]=0;
      for(r=dir*(N-1);r!=r0;r+=1-2*dir){
        // Here v0[b] = value of (c,previous,*) given that (c,r,1)=b
        if((c-p)&1){
          for(b=0;b<16;b++){// b = state of (c,r,1)
            v1[b]=v0[b]+QBI(d,c,r,0,0,XBI(d,c,r,0),b);
          }
        } else {
          for(b=0;b<16;b++){// b = state of (c,r,1)
            vmin=1000000000;smin=0;
            for(s=0;s<16;s++){// s = state of (c,r,0)
              v=(QBI(d,c,r,0,0,s,b)+
                 QBI(d,c,r,0,1,s,XBI(d,c-1,r,0))+
                 QBI(d,c,r,0,2,s,XBI(d,c+1,r,0)))<<4|ps[s];
              if(v<vmin){vmin=v;smin=s;}
            }
            v1[b]=v0[b]+(vmin>>4);
            hc[c][r][b]=smin;
          }
        }
        for(b=0;b<16;b++){// b = state of (c,r+1-2*dir,1)
          vmin=1000000000;smin=0;
          for(s=0;s<16;s++){// s = state of (c,r,1)
            v=(v1[s]+QBI(d,c,r,1,2-dir,s,b))<<4|ps[s];
            if(v<vmin){vmin=v;smin=s;}
          }
          v0[b]=vmin>>4;
          hs[c][r][b]=smin;
        }
      }//r
      for(s=0;s<16;s++)v2[s]+=v0[s];
    }//dir

    for(b=0;b<16;b++){// b = state of (c,r0,0)
      vmin=1000000000;smin=0;
      for(s=0;s<16;s++){// s = state of (c,r0,1)
        v=(v2[s]+QBI(d,c,r0,1,0,s,b))<<4|ps[s];
        if(v<vmin){vmin=v;smin=s;}
      }
      v4[b]=v3[b]+(vmin>>4);
      hc[c][r0][b]=smin;
    }

    for(b=0;b<16;b++){// b = state of (c+1,r0,0)
      vmin=1000000000;smin=0;
      for(s=0;s<16;s++){// s = state of (c,r0,0)
        v=(v4[s]+QBI(d,c,r0,0,2,s,b))<<4|ps[s];
        if(v<vmin){vmin=v;smin=s;}
      }
      v3[b]=vmin>>4;
      hr[c][b]=smin;
    }
  }//c

  if(upd){
    for(c=N-1;c>=0;c--){
      f=!((c-p)&1);
      XBI(d,c,r0,0)=hr[c][c==N-1?0:XBI(d,c+1,r0,0)];
      XBI(d,c,r0,1)=hc[c][r0][XBI(d,c,r0,0)];
      for(r=r0+1;r<N;r++){
        XBI(d,c,r,1)=hs[c][r][XBI(d,c,r-1,1)];
        if(f)XBI(d,c,r,0)=hc[c][r][XBI(d,c,r,1)];
      }
      for(r=r0-1;r>=0;r--){
        XBI(d,c,r,1)=hs[c][r][XBI(d,c,r+1,1)];
        if(f)XBI(d,c,r,0)=hc[c][r][XBI(d,c,r,1)];
      }
    }
  }

  return v3[0];
}

double tree1gibbs(int d,int p,int r0,double beta){
  // If d=0 sample the (induced) tree consisting of all verts of columns of parity p,
  //               the o=1 (vertically connected) verts of the other columns, and row r0.
  // If d=1 then same with rows <-> columns.
  // Comments and variable names are as if in the column case (d=0)
  // Updates tree to new sample and returns log(Z) of tree
  int b,c,f,r,s,dir,hc[N][N][16],hs[N][N][16],hr[N][16];
  double w,Z,max,lp[16],pr[16],W0[16],W1[16],W2[16],W3[16],W4[16];
  // W0[s] = log(Z) of current column fragment given that (c,r,1) = s
  // W2[s] = log(Z) of current column (apart from (c,r0,0)) given that (c,r0,1) = s
  // W3[s] = log(Z) of tree to left of column c given that (c,r0,0) = s

  for(s=0;s<16;s++)W3[s]=0;
  for(c=0;c<N;c++){

    for(s=0;s<16;s++)W2[s]=0;
    for(dir=0;dir<2;dir++){// dir=0 <-> increasing r, dir=1 <-> decreasing r
      for(s=0;s<16;s++)W0[s]=0;
      for(r=dir*(N-1);r!=r0;r+=1-2*dir){
        // Here W0[b] = log(Z) of (c,previous,*) given that (c,r,1)=b
        if((c-p)&1){
          for(b=0;b<16;b++){// b = state of (c,r,1)
            W1[b]=W0[b]-beta*QBI(d,c,r,0,0,XBI(d,c,r,0),b);
          }
        } else {
          for(b=0;b<16;b++){// b = state of (c,r,1)
            for(s=0,max=-1e9;s<16;s++){// s = state of (c,r,0)
              lp[s]=-beta*(QBI(d,c,r,0,0,s,b)+
                           QBI(d,c,r,0,1,s,XBI(d,c-1,r,0))+
                           QBI(d,c,r,0,2,s,XBI(d,c+1,r,0)));
              if(lp[s]>max)max=lp[s];
            }
            for(s=0,Z=0;s<16;s++){pr[s]=exp(lp[s]-max);Z+=pr[s];}
            for(w=randfloat()*Z,s=0;s<16;s++){w-=pr[s];if(w<=0)break;}
            assert(s<16);
            W1[b]=W0[b]+max+log(Z);
            hc[c][r][b]=s;
          }
        }
        for(b=0;b<16;b++){// b = state of (c,r+1-2*dir,1)
          for(s=0,max=-1e9;s<16;s++){// s = state of (c,r,1)
            lp[s]=W1[s]-beta*QBI(d,c,r,1,2-dir,s,b);
            if(lp[s]>max)max=lp[s];
          }
          for(s=0,Z=0;s<16;s++){pr[s]=exp(lp[s]-max);Z+=pr[s];}
          for(w=randfloat()*Z,s=0;s<16;s++){w-=pr[s];if(w<=0)break;}
          assert(s<16);
          W0[b]=max+log(Z);
          hs[c][r][b]=s;
        }
      }//r
      for(s=0;s<16;s++)W2[s]+=W0[s];
    }//dir

    for(b=0;b<16;b++){// b = state of (c,r0,0)
      for(s=0,max=-1e9;s<16;s++){// s = state of (c,r0,1)
        lp[s]=W2[s]-beta*QBI(d,c,r0,1,0,s,b);
        if(lp[s]>max)max=lp[s];
      }
      for(s=0,Z=0;s<16;s++){pr[s]=exp(lp[s]-max);Z+=pr[s];}
      for(w=randfloat()*Z,s=0;s<16;s++){w-=pr[s];if(w<=0)break;}
      assert(s<16);
      W4[b]=W3[b]+max+log(Z);
      hc[c][r0][b]=s;
    }

    for(b=0;b<16;b++){// b = state of (c+1,r0,0)
      for(s=0,max=-1e9;s<16;s++){// s = state of (c,r0,0)
        lp[s]=W4[s]-beta*QBI(d,c,r0,0,2,s,b);
        if(lp[s]>max)max=lp[s];
      }
      for(s=0,Z=0;s<16;s++){pr[s]=exp(lp[s]-max);Z+=pr[s];}
      for(w=randfloat()*Z,s=0;s<16;s++){w-=pr[s];if(w<=0)break;}
      assert(s<16);
      W3[b]=max+log(Z);
      hr[c][b]=s;
    }
  }//c


  for(c=N-1;c>=0;c--){
    f=!((c-p)&1);
    XBI(d,c,r0,0)=hr[c][c==N-1?0:XBI(d,c+1,r0,0)];
    XBI(d,c,r0,1)=hc[c][r0][XBI(d,c,r0,0)];
    for(r=r0+1;r<N;r++){
      XBI(d,c,r,1)=hs[c][r][XBI(d,c,r-1,1)];
      if(f)XBI(d,c,r,0)=hc[c][r][XBI(d,c,r,1)];
    }
    for(r=r0-1;r>=0;r--){
      XBI(d,c,r,1)=hs[c][r][XBI(d,c,r+1,1)];
      if(f)XBI(d,c,r,0)=hc[c][r][XBI(d,c,r,1)];
    }
  }

  return W3[0];
}

double bigvertexgibbs(int d,int x,int y,double beta){// Does Gibbs iteration to a single "bigvertex" (4 spins)
  // If d=0 then v=the bigvertex (x,y,0)
  // If d=1 then v=the bigvertex (y,x,1)
  // Replaces bigvertex v (4 spins) with a random value given by the Gibbs
  // distribution at inverse temperature beta conditioned on the rest of the graph.
  // Comments and variable names are as if in the horizontally-connected case d=0.
  // This is the basic unit of the simple Monte Carlo sweep
  int s;
  double z,Z,pr[16];
  for(s=0,Z=0;s<16;s++){
    pr[s]=exp(-beta*(QBI(d,x,y,0,0,s,XBI(d,x,y,1))+
                     QBI(d,x,y,0,1,s,XBI(d,x-1,y,0))+
                     QBI(d,x,y,0,2,s,XBI(d,x+1,y,0))));
    Z+=pr[s];
  }
  for(z=randfloat()*Z,s=0;s<16;s++){z-=pr[s];if(z<=0)break;}
  assert(s<16);
  XBI(d,x,y,0)=s;
  return Z;
}

typedef int treestriptype;// Use int if range of values exceeds 16 bits, or use short to save memory on a very wide exhaust.
int treestripexhaust(int d,int w,int ph,int upd,int fixedrow){
  // w=width, ph=phase (0,...,w)
  // If d=0 exhaust the (induced) treewidth w subgraph consisting of:
  //   all columns, c, s.t. c+ph is congruent to 0,1,...,w-1 mod w+1 ("full"),
  //   the o=1 (vertically connected) verts of columns where c+ph is congruent to w mod w+1 ("spike"),
  //   and horizontal joiners of randomly chosen height either side of each c+ph=w mod w+1 column.
  // If d=1 then same with rows <-> columns.
  // Comments and variable names are as if in the column case (d=0)
  // upd=1 <-> the optimum is written back into the global state
  // E.g., w=2, N=8:
  // ph=0: FFSFFSFF  \ F=full, S=spike (o=1)
  // ph=1: FSFFSFFS  |
  // ph=2: SFFSFFSF  /
  int b,c,f,i,r,s,x,v,nf,b1,s0,s1,bc,lw,dir,inc,mul,phl,smin,vmin,pre2[16][16],ps[N][16];
  int64 bi,br,bm,size0,size1;
  int jr0,jr1,jv0[16],jv1[16];// join row, value.
  treestriptype*v0,*v1,*v2,*vold,*vnew;
  double t0,t1,t2;
  size0=1LL<<4*w;
  size1=16*(size0-1)/15;
  v0=(treestriptype*)malloc(size0*sizeof(treestriptype));
  v1=(treestriptype*)malloc(size1*sizeof(treestriptype));
  v2=(treestriptype*)malloc(size0*sizeof(treestriptype));
  nf=(N-1+ph)/(w+1)-(ph+1)/(w+1)+1;// Number of full (non-spike) exhausts (the end ones could be narrow, but still called full)
  bm=1LL<<4*(w-1);
  // Spike labels are f = 0...nf-1 or 0...nf; Full labels are f = 0...nf-1
  int jr[N];// jr[c] = join row associated to column c
  UC hs0[nf+1][N][16],hs1[nf+1][16],(*hf0)[N][w][bm*16],(*hf1)[N][w][bm][16],(*hf2)[w][bm][16],(*hf3)[w][bm][16];
  hf0=0;hf1=0;hf2=hf3=0;
  if(upd){
    hf0=(UC(*)[N][w][bm*16])malloc(nf*N*w*bm*16);
    hf1=(UC(*)[N][w][bm][16])malloc(nf*N*w*bm*16);
    hf2=(UC(*)[w][bm][16])malloc(nf*w*bm*16);
    hf3=(UC(*)[w][bm][16])malloc(nf*w*bm*16);
  }
  if(!(v0&&v1&&v2&&(upd==0||(hf0&&hf1&&hf2&&hf3)))){
    fprintf(stderr,"Couldn't allocate %gGiB in treestripexhaust()\n",
            (double)((size0*2+size1)*sizeof(treestriptype)+!!upd*(nf*(2*N+2)*w*bm*16))/(1<<30));return 1;}
  t0=t1=t2=0;
  for(c=0;c<N;c++){for(s=0;s<16;s++)ps[c][s]=s;if(upd)shuf(ps[c],16);}
  jr0=randint(N);for(i=0;i<16;i++)jv0[i]=0;
  if(fixedrow>=0)jr0=fixedrow;
  for(c=f=0;c<N;){// f=full exhaust no.
    // jv0[s] = value of stuff to the left given that (c-1,jr0,0)=s
    phl=(c+ph)%(w+1);
    jr[c]=jr0;
    if(phl==w){// Spike
      t0-=cpu();
      jr1=jr0;
      for(s=0;s<16;s++)v2[s]=0;
      for(dir=0;dir<2;dir++){
        for(s=0;s<16;s++)v0[s]=0;
        for(r=dir*(N-1);r!=jr0;r+=1-2*dir){
          // Here v0[b] = value of (c,previous,*) given that (c,r,1)=b
          for(b=0;b<16;b++)v1[b]=v0[b]+QBI(d,c,r,0,0,XBI(d,c,r,0),b);// b = state of (c,r,1)
          for(b=0;b<16;b++){// b = state of (c,r+1-2*dir,1)
            vmin=1000000000;smin=0;
            for(s=0;s<16;s++){// s = state of (c,r,1)
              v=(v1[s]+QBI(d,c,r,1,2-dir,s,b))<<4|ps[c][s];
              if(v<vmin){vmin=v;smin=s;}
            }
            v0[b]=vmin>>4;
            if(upd)hs0[f][r][b]=smin;
          }
        }//r
        for(s=0;s<16;s++)v2[s]+=v0[s];
      }//dir
      
      for(b=0;b<16;b++){// b = state of (c,jr0,0)
        vmin=1000000000;smin=0;
        for(s=0;s<16;s++){// s = state of (c,jr0,1)
          v=(v2[s]+QBI(d,c,jr0,1,0,s,b))<<4|ps[c][s];
          if(v<vmin){vmin=v;smin=s;}
        }
        v0[b]=jv0[b]+(vmin>>4);
        if(upd)hs0[f][jr0][b]=smin;
      }

      for(b=0;b<16;b++){// b = state of (c+1,jr0,0)
        vmin=1000000000;smin=0;
        for(s=0;s<16;s++){// s = state of (c,jr0,0)
          v=(v0[s]+QBI(d,c,jr0,0,2,s,b))<<4|ps[c][s];
          if(v<vmin){vmin=v;smin=s;}
        }
        jv1[b]=vmin>>4;
        if(upd)hs1[f][b]=smin;
      }
      c++;
      t0+=cpu();

    }else{// Full
      assert(f<nf);
      lw=MIN(w-phl,N-c);assert(lw>=1&&lw<=w);// Local width
      jr1=randint(N);
      if(fixedrow>=0)jr1=fixedrow;
      jr[c+lw-1]=jr1;
      // Width lw exhaust, incoming jv0[] at row jr0, outgoing jv1[] at row jr1

      memset(v2,0,size0*sizeof(treestriptype));
      for(dir=0;dir<2;dir++){
        memset(v0,0,size0*sizeof(treestriptype));
        for(r=dir*(N-1);r!=jr1;r+=1-2*dir){
          // Comb exhaust
          // At this point: v0 maps (*,r,1) to value of (*,r,1), (*,<r,*)
          //      
          //             *b0       *b1       *b2       *b3
          //            /         /         /         /
          //           /         /         /         /
          //    ------*---------*---------*---------*------
          //          s0        s1        s2        s3
          //
          // vc3[s3] = Qext(s3,X4)
          // vc2[s2,b3] = min_{s3} vc3[s3]+Q(s3,b3)+Q(s3,s2)
          // vc1[s1,b2,b3] = min_{s2} vc2[s2,b3]+Q(s2,b2)+Q(s2,s1)
          // vc0[s0,b1,b2,b3] = min_{s1} vc1[s1,b2,b3]+Q(s1,b1)+Q(s1,s0) (variable names s0,s1,b1 correspond to this x=1 case)
          // v0[b0,b1,b2,b3] += min_{s0} vc0[s0,b1,b2,b3]+Q(s0,b0)+{Qext(X_{-1},s0), or jv0[s0] if r=jr0}
          t1-=cpu();
          vold=v1;
          for(s=0;s<16;s++)vold[s]=QBI(d,c+lw-1,r,0,2,s,XBI(d,c+lw,r,0));// right boundary interaction
          for(x=lw-1,bm=1;x>=0;bm*=16,x--){
            // Loop over br = (b_{x+1},...,b_{lw-1}) { // the irrelevant parameters
            //   Loop over s_{x-1} {
            //     Loop over b_x {
            //       vc[x-1][s_{x-1},b_x,br] = min over s_x of vc[x][s_x,br]+Q(s_x,b_x)+Q(s_x,s_{x-1})
            //     }
            //   }
            // }
            int ql[16];// left boundary interaction
            if(r==jr0)memcpy(ql,jv0,16*sizeof(int)); else for(s=0;s<16;s++)ql[s]=QBI(d,c,r,0,1,s,XBI(d,c-1,r,0));
            vnew=vold+16*bm;
            mul=(x>0?16:1);
            for(br=0;br<bm;br++){// br is state of (c+x+1,r,1),...,(c+lw-1,r,1)
              for(s0=0;s0<mul;s0++){// s0 is state of (c+x-1,r,0) (doesn't exist if x=0)
                for(b1=0;b1<16;b1++){// b1 is state of (c+x,r,1)
                  vmin=1000000000;smin=0;
                  for(s1=0;s1<16;s1++){// s1 is state of (c+x,r,0)
                    v=(vold[s1+16*br]+QBI(d,c+x,r,0,0,s1,b1)+(x>0?QBI(d,c+x,r,0,1,s1,s0):ql[s1]))<<4|ps[c][s1];
                    if(v<vmin){vmin=v;smin=s1;}
                  }
                  bi=s0+mul*(b1+16*br);
                  if(x>0)vnew[bi]=vmin>>4; else v0[bi]+=vmin>>4;
                  //printf("comb c=%d f=%d r=%d x=%d br=%lld s0=%d b1=%d bi=%lld smin=%d vmin=%d\n",c,f,r,x,br,s0,b1,bi,smin,vmin>>4);
                  if(upd)hf0[f][r][x][bi]=smin;
                }
              }
            }
            vold=vnew;
          }//x
          assert(vnew-v1<=size1);
          if(lw==w)assert(vnew-v1==size1);
          // At this point v0 maps (*,r,1) to value of (*,<=r,*)
          t1+=cpu();
          
          // Strut exhaust
          //
          //     *b0       *bc       *         *
          //     |         |         |         |
          //     |         ^         |         |
          //     |         |         |         |
          //     *         *s        *b2       *b3
          //
          // (c=1 dir=0 picture)
          t2-=cpu();
          bm=1LL<<4*(lw-1);
          for(x=0;x<lw;x++){
            if(x&1){vold=v1;vnew=v0;} else {vold=v0;vnew=v1;}
            // At this point vold maps (>=c+x,r,1), (<c+x,r+1,1) to the value below these vertices
            // (r+1 corresponds to dir=0; r-1 and mutatis mutandis for dir=1)
            for(bc=0;bc<16;bc++){// bc = state of (c+x,r+1,1)
              for(s=0;s<16;s++){// s = state of (c+x,r,1)
                pre2[bc][s]=QBI(d,c+x,r,1,2-dir,s,bc);
              }
            }
            for(br=0;br<bm;br++){// br = state of non-(c+x) columns in cyclic order x+1,...,lw-1,0,...,x-1
              //                         i.e., cols c+x+1,...,c+lw-1 at row r
              //                         then cols c,...,c+x-1 at row r+1
              for(bc=0;bc<16;bc++){// bc = state of (c+x,r+1,1)
                vmin=1000000000;smin=0;
                for(s=0;s<16;s++){// s = state of (c+x,r,1)
                  v=(vold[s+16*br]+pre2[bc][s])<<4|ps[c][s];
                  if(v<vmin){vmin=v;smin=s;}
                }
                vnew[br+bm*bc]=vmin>>4;
                if(upd)hf1[f][r][x][br][bc]=smin;
              }
            }
          }//x
          if(lw&1)memcpy(v0,v1,size0*sizeof(treestriptype));
          // Now v0 maps (*,r+1,1) to value of (*,r+1,1),(*,<=r,*)
          t2+=cpu();
        }//r
        for(br=0;br<size0;br++)v2[br]+=v0[br];
      }//dir

      // Now v2 maps (*,jr1,1) to value of (*,r!=jr1,*)
      // v2[b0,b1,b2,b3]=val(above and below)
      // Think of this as v2[s0,b0,b1,b2,b3] but not depending on s0
      //
      //                            v2
      //               .------------+-------------.
      //              .         .         .        .
      //             *b0       *b1       *b2       *b3
      //            /         /         /         /
      //           /         /         /         /
      // *--------*---------*---------*---------*--------*
      // ext or   s0        s1        s2        s3       jv1[s4]
      // jv0[s0]
      
      for(x=0;x<lw;x++){
        // v0[s_x,b_{x+1},..,b_{lw-1}] = min_{b_x} v2[s_x,b_x,...,b_{lw-1}]
        //                                         + Q(s_x,b_x) + (if x=0) Qext(X_{-1},s_x) or jv0[s_x] if jr1=jr0
        // s_x is (c+x,jr1,0)
        // b_x is (c+x,jr1,1)
        //
        bm=1LL<<4*(lw-1-x);
        for(br=0;br<bm;br++){// br = state of b_{x+1},...,b_{lw-1}
          for(s=0;s<16;s++){// s = state of s_x
            vmin=1000000000;smin=0;
            for(bc=0;bc<16;bc++){// bc = state of b_x
              if(x==0)v=v2[bc+16*br]; else v=v2[s+16*(bc+16*br)];
              v+=QBI(d,c+x,jr1,0,0,s,bc);
              v=(v<<4)|ps[c][bc];
              if(v<vmin){vmin=v;smin=bc;}
            }
            vmin>>=4;
            if(x==0){
              if(jr1==jr0)vmin+=jv0[s]; else vmin+=QBI(d,c,jr1,0,1,s,XBI(d,c-1,jr1,0));
            }
            v0[s+16*br]=vmin;
            if(upd)hf2[f][x][br][s]=smin;
          }//s
        }//br

        // v2[s_{x+1},b_{x+1},...,b_{lw-1}] = min_{s_x} v0[s_x,b_{x+1},...,b_{lw-1}] + Q(s_x,s_{x+1})
        for(br=0;br<bm;br++){// br = state of b_{x+1},...,b_{lw-1}
          for(s1=0;s1<16;s1++){// s = state of s_{x+1}
            vmin=1000000000;smin=0;
            for(s0=0;s0<16;s0++){// s = state of s_x
              v=(v0[s0+16*br]+QBI(d,c+x,jr1,0,2,s0,s1))<<4|ps[c][s0];
              if(v<vmin){vmin=v;smin=s0;}
            }
            v2[s1+16*br]=vmin>>4;
            if(upd)hf3[f][x][br][s1]=smin;
          }//s1
        }//br
      }//x
      for(s=0;s<16;s++)jv1[s]=v2[s];
      c+=lw;f++;
    }
    for(s=0;s<16;s++)jv0[s]=jv1[s];
    jr0=jr1;
  }//c
  assert(f==nf&&c==N);

  if(upd){
    for(c=N;c>0;){
      // Incoming info is state of (c,jr[c-1],0)
      phl=(c+w+ph)%(w+1);
      jr1=jr[c-1];
      if(phl==w){// Came from spike
        c--;
        XBI(d,c,jr1,0)=hs1[f][c<N?XBI(d,c+1,jr1,0):0];
        XBI(d,c,jr1,1)=hs0[f][jr1][XBI(d,c,jr1,0)];
        for(dir=0;dir<2;dir++){
          inc=1-2*dir;
          for(r=jr1-inc;r>=0&&r<N;r-=inc)XBI(d,c,r,1)=hs0[f][r][XBI(d,c,r+inc,1)];
        }
      }else{
        f--;lw=MIN(phl+1,c);c-=lw;jr0=jr[c];
        br=0;
        for(x=lw-1;x>=0;x--){
          XBI(d,c+x,jr1,0)=hf3[f][x][br][XBI(d,c+x+1,jr1,0)];
          XBI(d,c+x,jr1,1)=hf2[f][x][br][XBI(d,c+x,jr1,0)];
          br=(br<<4)|XBI(d,c+x,jr1,1);
        }
        // Info is (c...c+lw-1,jr1,*)
        for(dir=0;dir<2;dir++){
          inc=1-2*dir;
          for(r=jr1-inc;r>=0&&r<N;r-=inc){
            // Info is (c...c+lw-1,r+inc,1)
            // De-strut
            for(x=lw-2,br=0;x>=0;x--)br=(br<<4)|XBI(d,c+x,r+inc,1);
            for(x=lw-1;x>=0;x--){
              XBI(d,c+x,r,1)=hf1[f][r][x][br][XBI(d,c+x,r+inc,1)];
              br=((br<<4)&~(15LL<<4*(lw-1)))|XBI(d,c+x,r,1);
            }
            // Info is (c...c+lw-1,r,1)
            // De-comb
            for(x=lw-1,br=0;x>=0;x--)br=(br<<4)|XBI(d,c+x,r,1);
            for(x=0;x<lw;x++){
              if(x==0)bi=br; else bi=(br<<4)|XBI(d,c+x-1,r,0);
              XBI(d,c+x,r,0)=hf0[f][r][x][bi];
              //printf("de-comb c=%d f=%d r=%d x=%d br=%lld bi=%lld smin=%d\n",c,f,r,x,br,bi,XBI(d,c+x,r,0));
              br>>=4;
            }
          }//r
        }//dir
      }
    }//c
    assert(f==0&&c==0);
    free(hf3);free(hf2);free(hf1);free(hf0);
  }
  free(v2);free(v1);free(v0);
  //printf("Times %.2fs %.2fs %.2fs\n",t0,t1,t2);
  return jv0[0];
}

int stablek44exhaust(int cv){// A round of exhausts on each K44 (big vertex)
  int i,r,v,x,y,ord[N*N];
  r=0;
  for(i=0;i<N*N;i++)ord[i]=i;shuf(ord,N*N);
  while(1){
    for(i=0;i<N*N;i++){
      x=ord[i]%N;y=ord[i]/N;k44exhaust(x,y);
      v=val();assert(v<=cv);
      if(v<cv){cv=v;r=0;}else{r+=1;if(r==N*N)return cv;}
    }
  }
}

int stablestripexhaust(int cv,int wid){// Repeated strip exhausts until no more improvement likely
  int c,i,o,r,v,nc,ord[2*(N-wid+1)];
  nc=N-wid+1;r=0;
  while(1){
    for(i=0;i<2*nc;i++)ord[i]=i;
    //shuf(ord,2*nc);
    shuf(ord,nc);shuf(ord+nc,nc);
    for(i=0;i<2*nc;i++){
      c=ord[i]%nc;o=ord[i]/nc;
      stripexhaust(o,c,c+wid,1);
      v=val();assert(v<=cv);
      if(v<cv){cv=v;r=0;}else{r+=1;if(r==2*nc)return cv;}
    }
  }
}

int stabletreeexhaust(int cv,int wid,int64*ntr){// Repeated tree exhausts until no more improvement likely
  int d,n,ph,ph0,r,v;
  n=0;d=randint(2);ph=ph0=randint(wid+1);r=randint(N);
  while(1){
    if(ntr)(*ntr)++;
    if(wid==1)v=tree1exhaust(d,ph,r,1);//{tree1gibbs(d,ph,r,genp[0]);v=val();}
    else v=treestripexhaust(d,wid,ph,1,r);
    if(v<cv){cv=v;n=0;}else{n+=1;if(n==(wid+1)*2)return cv;}
    r=(r+1)%N;
    ph=(ph+1)%(wid+1);if(ph==ph0)d=1-d;
  }
}

void resizecumsumleft(int64*cst,int size){// insert cst[0..size-1] into right of cst[0..2*size-1] and clear left
  int s;
  for(s=size/2;s>=1;s>>=1){
    memcpy(cst+3*s,cst+s,s*sizeof(int64));
    memset(cst+2*s,0,s*sizeof(int64));
  }
  cst[1]=cst[0];
}
void resizecumsumright(int64*cst,int size){// insert cst[0..size-1] into left of cst[0..2*size-1] and clear right
  int s;
  for(s=size/2;s>=1;s>>=1){
    memcpy(cst+2*s,cst+s,s*sizeof(int64));
    memset(cst+3*s,0,s*sizeof(int64));
  }
  cst[1]=0;
}
void inccumsum(int64*cst,int size,int v){// effectively increment all of [0,v)
  for(v+=size;v>0;v/=2)if(v&1)cst[v/2]++;
}
int64 querycumsumgt(int64*cst,int size,int v){// query how many increments were greater than v
  int64 t;
  if(v<0)return cst[0];
  if(v>=size)return 0;
  for(v+=size,t=0;v>0;v/=2)if(!(v&1))t+=cst[v/2];
  return t;
}
int64 querycumsumle(int64*cst,int size,int v){return cst[0]-querycumsumgt(cst,size,v);}// query how many increments were less than or equal to v
int64 querycumsumlt(int64*cst,int size,int v){return querycumsumle(cst,size,v-1);}// ditto, less than v
int64 querycumsumeq(int64*cst,int size,int v){return querycumsumle(cst,size,v)-querycumsumlt(cst,size,v);}// ditto, equal to v

#define MAXST (1<<18) // For stats.
int opt1(double mint,double maxt,int pr,int tns,double *findtts,int strat){
  //
  // Heuristic optimisation, writing back best value found. Can be used to find TTS, the
  // expected time to find an optimum solution, using the strategy labelled by 'strat'.
  // 'strat' is assumed to be a fixed strategy running forever which does not know what
  // the optimum value is. I.e., it is not allowed to make decisions as to how to search
  // based on outside knowledge of the optimum value. In "findtts" mode, the aim is to get
  // an accurate estimate of the expected time for 'strat' to find its first optimum
  // state.
  // 
  // opt1() returns a pair (presumed optimum, estimate of TTS), the "presumed optimum"
  // being the smallest value found in its searching. If the presumed optimum is wrong
  // (not actually the optimum) then the estimate of TTS is allowed to be anything. If the
  // presumed optimum is correct then the estimate of TTS must be unbiased. So for the
  // purposes of reasoning whether or not opt1 is behaving correctly, we only care about
  // the case when the presumed optimum is the actual optimum. Of course, we also want to
  // make it very likely that the presumed optimum is the actual optimum, but we don't
  // seek to quantify what constitutes very likely here.
  // 
  // Strategies maintain some "state" in addition to the spin configuration. They use
  // information from previous iterations to guide the present iteration. This means that
  // to get unbiased samples of "time to solve", you need to stop the strategy with it
  // hits an optimum and then restart it cleanly, clearing all state.  Only that way can
  // you be sure that you are averaging unbiased runs when taking the average TTS.
  //
  // Notionally it runs as if it is presided over by an oracle that resets its state
  // whenever it hits the optimal value. The wrinkle is that the strategy itself is the
  // thing that is deciding the optimal value. Since the strategy doesn't actually know
  // the optimum value for certain, and is not allowed to use it to change its behaviour,
  // it has to restart itself every time it hits a "presumed optimum", i.e., a
  // (equal-)lowest value found so far. For external convenience, a record of this optimum
  // is carried over from independently restarted runs, but this value is "unauthorised
  // information" - the strategy is not allowed to use it to make a decision. I.e., you
  // have to imagine the run carrying on forever, but the presumed optimum just dictates
  // when to take the time reading (at the point the run finds an equally good value).
  //
  // If the presumed optimum is bettered, so making a new presumed optimum, then all
  // previous statistics have to be discarded, including the current run which found the
  // new presumed optimum. This is because the new presumed optimum was not found under
  // "infinity run" conditions: the early stopping at the old presumed optimum might have
  // biased it. So you actually need to find n+1 optima to generate n samples.
  //
  // S0: Randomise configuration; stablek44exhaust; repeat
  // S1: Randomise configuration; stablelineexhaust; repeat
  // S2: Randomise configuration; hybrid of stablelineexhaust and stable-width2-exhaust
  // S3: Randomise configuration; stabletree1exhaust
  // S4: Randomise configuration; stabletree2exhaust
  // S(10+n): Do Sn but randomly perturb configuration instead of randomise it entirely

  int v,w,bv,lbv,nis,cmin,cv,nv,ns,new,last,reset,ssize,Xbest[NBV],Xlbest[NBV];
  int64 nn,rep,stt,ntr,*stats;
  double ff,t0,t1,t2,tt,w1,now,work[N+1];
  double parms[6][2]={{0.5,0.3},{0.25,0.25},{0.5,0.25},{0.5,0.35},{0.25,0.2},{0.25,0.2}};

  // These are for hybrid strat 2, not currently used:
  // work[wid] counts approx number of QBI references in a stable exhaust of width wid
  for(w=1;w<=N;w++)work[w]=N*(1LL<<4*(w+1))*(w+32/15.)*(N-w+1)*3;
  ff=N*N/40.;
  
  if(pr){
    printf("Target number of presumed optima: %d\n",tns);
    printf("Min time to run: %gs\nMax time to run: %gs\n",mint,maxt);
    printf("Solutions are %sdependent\n",findtts?"in":"");
  }
  bv=1000000000;nn=0;
  t0=cpu();// Initial time
  t1=0;// Elapsed time threshold for printing update
  // "presumed solution" means "minimum value found so far"
  ns=0;// Number of presumed solutions
  ntr=0;// Number of treeexhausts (which is nearly number of sweeps)
  t2=t0;// t2 = Time of last clean start (new minimum value in "independent" mode)
  stats=(int64*)malloc(MAXST*sizeof(int64));assert(stats);
  memset(Xbest,0,NBV*sizeof(int));
  if(N>=2&&strat%10==2&&pr)printf("w1/w2 = %g\n",work[2]/(ff*work[1]));
  reset=1;
  w1=rep=cv=lbv=nis=0;
  do{
    if(reset){
      init_state();nis++;cv=val();
      cmin=1000000000;ssize=1024;assert(ssize<=MAXST);memset(stats,0,ssize*sizeof(int64));// Reset stats
      stt=0;// Total count of values found
      lbv=1000000000;// Local best value (for S2)
      memset(Xlbest,0,NBV*sizeof(int));// Local best state (for S2)
      w1=0;// Work done so far at width 1 since last width 2 exhaust or last presumed solution (for S2)
      rep=0;
      reset=0;
    }
    //printf("%10d %10d %10lld %10lld\n",cv,bv,rep,stt);
    if(rep>=stt*parms[strat%10][0]){
      if(strat<10){init_state();nis++;} else pertstate(parms[strat%10][1]);
      cv=val();rep=0;
    }
    switch(strat%10){
    case 0:
      nv=stablek44exhaust(cv);// Simple "local" strategy
      break;
    case 1:
      nv=stablestripexhaust(cv,1);
      break;
    case 2:// hybrid mode
      nv=stablestripexhaust(cv,1);w1+=work[1];
      if(nv<=lbv){lbv=nv;memcpy(Xlbest,XBa,NBV*sizeof(int));}
      if(N>=2&&ff*w1>=work[2]){
        if(pr>=2){printf("Width 2 exhaust");fflush(stdout);}
        memcpy(XBa,Xlbest,NBV*sizeof(int));
        nv=stablestripexhaust(lbv,2);
        if(pr>=2)printf("\n");
        w1=0;lbv=1000000000;
      }
      break;
    case 3:
      nv=stabletreeexhaust(cv,1,&ntr);
      break;
    case 4:
      nv=stabletreeexhaust(cv,2,&ntr);
      break;
    case 5:
      nv=stabletreeexhaust(cv,3,&ntr);
      break;
    default:
      fprintf(stderr,"Unknown strategy %d\n",strat);exit(1);
    }
    if(nv<cv)rep=0;
    cv=nv;
    if(cmin==1000000000)cmin=cv-ssize/2;
    // [cmin,cmin+ssize) corresponds to [0,ssize) in stats[], as encoded by cumsum tree
    while(cv<cmin){assert(ssize*2<=MAXST);resizecumsumleft(stats,ssize);cmin-=ssize;ssize*=2;}
    while(cv>=cmin+ssize){assert(ssize*2<=MAXST);resizecumsumright(stats,ssize);ssize*=2;}
    if(rep==0){inccumsum(stats,ssize,cv-cmin);stt++;}// possibly change rep=0 condition
    rep+=querycumsumle(stats,ssize,cv-cmin);
    if((pr>=3&&cv<=bv)||pr>=5){printf("\nSTATE cv=%d bv=%d\n",cv,bv);prstate(stdout,0,0);printf("DIFF\n");prstate(stdout,1,Xbest);}
    nn++;
    now=cpu();
    new=(cv<bv);
    if(new){bv=cv;ns=0;ntr=0;memcpy(Xbest,XBa,NBV*sizeof(int));}
    if(cv==bv){
      if(new&&findtts)t2=now; else ns++;
      if(findtts)reset=1;
    }
    tt=now-t0;
    last=(now-t2>=mint&&ns>=tns)||tt>=maxt;
    if(new||tt>=t1||last){
      t1=MAX(tt*1.1,tt+5);
      if(pr>=1){
        if(findtts)printf("%12lld %10d %10d %8.2f %8.2f %10.3g %10.3g %10.3g\n",nn,bv,ns,now-t2,tt,(now-t2)/ns,ntr/(double)nis,nis/(double)ns); else
          printf("%12lld %10d %8.2f\n",nn,bv,tt);
        if(pr>=2){
          printf("Tot stat %lld\n",stt);
          for(v=cmin+ssize-1;v>=cmin;v--){
            int64 n=querycumsumeq(stats,ssize,v-cmin);
            if(n)printf("%6d %12lld\n",v,n);
          }
          if(pr>=4){printf("STATE cv=%d bv=%d\n",cv,bv);prstate(stdout,0,0);printf("DIFF\n");prstate(stdout,1,Xbest);printf("\n");}
        }
        fflush(stdout);
      }
    }
  }while(!last);
  if(findtts)*findtts=(now-t2)/ns;
  memcpy(XBa,Xbest,NBV*sizeof(int));
  free(stats);
  return bv;
}

int cmpint(const void*p,const void*q){return *(int*)p-*(int*)q;}
int cmpd(const void*p,const void*q){double z=*(double*)p-*(double*)q;return (z>0)-(z<0);}

int okinv(int c,int r,int o,int s){// aborts if s isn't on the OK list
  int i;
  if(c<0||c>=N||r<0||r>=N){assert(s==0);return 0;}
  for(i=0;i<nok[enc(c,r,o)];i++)if(ok[enc(c,r,o)][i]==s)return i;
  assert(0);
}

int ok2inv(int c,int r,int s){// returns -1 if s isn't on the OK2 list
  int i;
  if(c<0||c>=N||r<0||r>=N)return s==0?0:-1;
  for(i=0;i<nok2[enc2(c,r)];i++)if(ok2[enc2(c,r)][i]==s)return i;
  return -1;
}

void getrestrictedsets(void){
  int i,j,o,s,v,x,y,bb,s0,s1,s0b,s1b,tt,tt0,v0,x0,x1,y0,y1,max,vmin,tv[16],meet[16][16],ll0,ll[65536];
  UC (*ok0)[16][16];
  ok0=(UC(*)[16][16])malloc(65536*16*16);assert(ok0);
  for(v=0;v<NBV;v++)for(s=0;s<16;s++)ok[v][s]=1;
  tt0=1000000000;
  for(x=0;x<N;x++)for(y=0;y<N;y++){for(i=0;i<256;i++)ok2[enc2(x,y)][i]=i;nok2[enc2(x,y)]=256;}
  while(1){
    tt=0;
    for(x=0;x<N;x++)for(y=0;y<N;y++)for(o=0;o<2;o++){for(s=0;s<16;s++)tt+=ok[enc(x,y,o)][s];}
    if(deb>=1)printf("Total %4d / %4d\n",tt,N*N*2*16);
    if(tt>=tt0)break;
    tt0=tt;
    for(x=0;x<N;x++)for(y=0;y<N;y++){
      bb=0;
      for(x0=0;x0<16;x0++)if((x==0&&x0==0)||(x>0&&ok[enc(x-1,y,0)][x0])){
        for(x1=0;x1<16;x1++)if((x==N-1&&x1==0)||(x<N-1&&ok[enc(x+1,y,0)][x1])){
          for(y0=0;y0<16;y0++)if((y==0&&y0==0)||(y>0&&ok[enc(x,y-1,1)][y0])){
            for(y1=0;y1<16;y1++)if((y==N-1&&y1==0)||(y<N-1&&ok[enc(x,y+1,1)][y1])){
              for(s1=0;s1<16;s1++)tv[s1]=QB(x,y,1,1,s1,y0)+QB(x,y,1,2,s1,y1);
              vmin=1000000000;
              for(s0=0;s0<16;s0++){
                //if(ok[enc(x,y,0)][s0]==0)continue;//possible
                v0=QB(x,y,0,1,s0,x0)+QB(x,y,0,2,s0,x1);
                for(s1=0;s1<16;s1++){
                  //if(ok[enc(x,y,1)][s1]==0)continue;//possible
                  v=QB(x,y,0,0,s0,s1)+v0+tv[s1];
                  if(v<vmin){memset(ok0[bb],0,16*16);vmin=v;}
                  if(v==vmin)ok0[bb][s0][s1]=1;
                }//s1
              }//s0
              bb++;
            }//y1
          }//y0
        }//x1
      }//x0
      //printf("bb=%d\n",bb);
      for(o=0;o<2;o++)for(s=0;s<16;s++)ok[enc(x,y,o)][s]=0;
      for(i=0;i<bb-1;i++)ll[i]=i+1;ll0=0;ll[bb-1]=-1;
      nok2[enc2(x,y)]=0;
      while(1){
        memset(meet,0,sizeof(meet));
        for(i=ll0;i>=0;i=ll[i])for(s0=0;s0<16;s0++)for(s1=0;s1<16;s1++)meet[s0][s1]+=ok0[i][s0][s1];
        //for(s0=0;s0<16;s0++){for(s1=0;s1<16;s1++)printf("%10d ",meet[s0][s1]);printf("\n");}
        s0b=s1b=-1;// To shut compiler up
        for(s0=0,max=0;s0<16;s0++)for(s1=0;s1<16;s1++)if(meet[s0][s1]>max){max=meet[s0][s1];s0b=s0;s1b=s1;}
        // ^ Can use better method. Should include cartesian product of projections first.
        if(max==0)break;
        ok[enc(x,y,0)][s0b]=ok[enc(x,y,1)][s1b]=1;
        ok2[enc2(x,y)][nok2[enc2(x,y)]++]=s0b+(s1b<<4);
        for(i=ll0;i>=0&&ok0[i][s0b][s1b];i=ll[i]);
        if(i<0)break;
        ll0=i;
        while(i>=0){
          for(j=ll[i];j>=0&&ok0[j][s0b][s1b];j=ll[j]);
          ll[i]=j;i=j;
        }
      }// subset-choosing loop
      //printf("%2d %2d:",x,y);for(o=0;o<2;o++){printf("   ");for(s=0;s<16;s++)printf("%d ",ok[enc(x,y,o)][s]);}printf("\n");
    }//x,y
  }
  free(ok0);
  // Convert indicator map to list
  for(v=0;v<NBV;v++){for(s=0,i=0;s<16;s++)if(ok[v][s])ok[v][i++]=s;nok[v]=i;}
  if(deb>=2)for(x=0;x<N;x++)for(y=0;y<N;y++)for(o=0;o<2;o++){
    printf("%d %d %d :",x,y,o);
    for(i=0;i<nok[enc(x,y,o)];i++)printf(" %2d",ok[enc(x,y,o)][i]);
    printf("\n");
  }
  // Sort ok2[] to facilitate fullexhaust()
  for(x=0;x<N;x++)for(y=0;y<N;y++)qsort(ok2[enc2(x,y)],nok2[enc2(x,y)],sizeof(int),cmpint);
  ok[NBV][0]=0;nok[NBV]=1;   // Special entries at the end to cater for off-grid cells
  ok2[N*N][0]=0;nok2[N*N]=1; //
  if(deb>=1){
    for(y=N-1;y>=0;y--){
      for(x=0;x<N;x++){
        for(o=0;o<2;o++){
          v=nok[enc(x,y,o)];
          if(v<16)printf("%x",v); else printf("g");
        }
        printf(" ");
      }
      printf("\n");
    }
    for(y=N-1;y>=0;y--){
      for(x=0;x<N;x++)printf("%3d ",nok2[enc2(x,y)]);
      printf("\n");
    }
  }
}

void applyam(int a,int*XBa0,int(*QBa0)[3][16][16],int(*ok0)[16],int*nok0,int(*ok20)[256],int*nok20){
  // Apply automorphism a=0,1,...,7
  int d,i,o,t,v,x,y,o1,x1,y1,dx,dy,d1,v1,s0,s1;
  for(x=0;x<N;x++)for(y=0;y<N;y++){
    x1=x;y1=y;
    if(a&1){x1=y;y1=x;}
    if(a&2)x1=N-1-x1;
    if(a&4)y1=N-1-y1;
    for(o=0;o<2;o++){
      v=enc(x,y,o);
      o1=o^(a&1);
      v1=enc(x1,y1,o1);
      XBa[v]=XBa0[v1];
      for(d=0;d<3;d++){
        d1=d;
        if(d){
          if(o==0){dx=2*d-3;dy=0;}else{dx=0;dy=2*d-3;}
          if(a&1){t=dx;dx=dy;dy=t;}
          if(a&2){dx=-dx;}
          if(a&4){dy=-dy;}
          if(o1==0){assert(dy==0);d1=(dx+3)/2;}else{assert(dx==0);d1=(dy+3)/2;}
        }
        for(s0=0;s0<16;s0++)for(s1=0;s1<16;s1++)QBa[v][d][s0][s1]=QBa0[v1][d1][s0][s1];
      }
      nok[v]=nok0[v1];
      for(i=0;i<nok[v];i++)ok[v][i]=ok0[v1][i];
    }//o
    v=enc2(x,y);v1=enc2(x1,y1);
    nok2[v]=nok20[v1];
    for(i=0;i<nok2[v];i++)if(a&1){t=ok20[v1][i];ok2[v][i]=(t>>4)|((t&15)<<4);} else ok2[v][i]=ok20[v1][i];
    qsort(ok2[v],nok2[v],sizeof(int),cmpint);
  }//x,y
}

int fullexhaust(){
  // Uses restricted sets to cut down possibilities
  // and full automorphism group to choose best orientation
  int a,c,r,s,v,x,bc,bc0,A,s0,mul0,mul1,offset,
    XBa0[NBV],QBa0[NBV][3][16][16],ok0[NBV][16],nok0[NBV],ok20[N*N][256],nok20[N*N],
    pre[4096][4],pre2[16][16];
  int64 b,br,bm,nc,ns,nc0,tnc,maxc,maxs,maxt,size0,size1;
  double t0,t1,t2,tns,ctns,cost,mincost;
  short*v0,*v1,*vold,*vnew;

  t0=-cpu();
  getrestrictedsets();
  memcpy(XBa0,XBa,sizeof(XBa0));memcpy(QBa0,QBa,sizeof(QBa0));
  memcpy(ok0,ok,sizeof(ok0));memcpy(nok0,nok,sizeof(nok0));
  memcpy(ok20,ok2,sizeof(ok20));memcpy(nok20,nok2,sizeof(nok20));
  mincost=1e100;A=-1;size0=size1=1LL<<60;
  if(deb>=1)printf("                  Memory/GiB   Time(a.u.)  Memory*Time\n");
  for(a=0;a<8;a++){// Loop over automorphisms of C_N to choose best representation to exhaust
    applyam(a,XBa0,QBa0,ok0,nok0,ok20,nok20);
    maxc=maxs=0;tns=0;
    for(r=0;r<N;r++){
      nc0=1;tnc=0;
      for(c=N-1;c>=0;c--){
        tns+=nok[encp(c-1,r,0)]*nok2[enc2(c,r)]*nc0;
        nc0*=nok[encp(c+1,r,1)];
        nc=nc0*nok[enc(c,r,0)];
        tnc+=nc;
      }
      if(tnc>maxc)maxc=tnc;
      for(c=0;c<N;c++){
        ns=1;
        for(x=0;x<c;x++)ns*=nok[encp(x,r+1,1)];
        for(x=c;x<N;x++)ns*=nok[enc(x,r,1)];
        tns+=ns*nok[encp(c,r+1,1)];
        if(ns>maxs)maxs=ns;
      }
    }//r
    maxt=maxs+MAX(maxs,maxc);
    cost=tns*maxt;// Using cost = time * memory
    if(deb>=1){double z=(double)maxt*sizeof(short)/(1<<30);printf("Automorphism %d: %12g %12g %12g\n",a,z,tns,z*tns);}
    if(cost<mincost){mincost=cost;size0=maxs;size1=MAX(maxs,maxc);ctns=tns;A=a;}
  }//a
  applyam(A,XBa0,QBa0,ok0,nok0,ok20,nok20);
  if(deb>=1)printf("Choosing automorphism %d\n",A);
  printf("Size %.1fGiB\n",(double)(size0+size1)*sizeof(short)/(1<<30));
  printf("Time units %g\n",ctns);
  fflush(stdout);
  v0=(short*)malloc(size0*sizeof(short));
  v1=(short*)malloc(size1*sizeof(short));
  if(!(v0&&v1)){fprintf(stderr,"Couldn't allocate %gGiB in fullexhaust()\n",
                        (double)(size0+size1)*sizeof(short)/(1<<30));return 1;}
  t0+=cpu();

  offset=32768/(N*(N+1));
  t1=t2=0;
  memset(v0,0,size0*sizeof(short));
  for(r=0;r<N;r++){
    // Comb exhaust
    // At this point: v0 maps (*,r,1) to value of (*,r,1), (*,<r,*)
    //      
    //        *b0       *b1       *b2       *b3
    //       /         /         /         /
    //      /         /         /         /
    //     *---------*---------*---------*
    //     s0        s1        s2        s3
    //
    // vc3[s3] = 0
    // vc2[s2,b3] = min_{s3} vc3[s3]+Q(s3,b3)+Q(s3,s2)
    // vc1[s1,b2,b3] = min_{s2} vc2[s2,b3]+Q(s2,b2)+Q(s2,s1)
    // vc0[s0,b1,b2,b3] = min_{s1} vc1[s1,b2,b3]+Q(s1,b1)+Q(s1,s0)
    // v0[b0,b1,b2,b3] += min_{s0} vc0[s0,b1,b2,b3]+Q(s0,b0)

    t1-=cpu();
    vold=v1;
    for(c=N-1,bm=1;c>=0;bm*=nok[enc(c,r,1)],c--){
      int np,s0i,s1i,b1i,sb1,sb1i,psb1;
      // Loop over br = (b_{c+1},...,b_{N-1}) { // the irrelevant parameters
      //   Loop over s_{c-1} {
      //     vmin=32767
      //     Loop over s_c+b_c { // s_c being the fast-changing half of s_c,b_c
      //       v=vc[c][s_c,br]+Q(s_c,b_c)+Q(s_c,s_{c-1})
      //       if(v<vmin)vmin=v
      //       if(s_c==<last one given b_c>){vc[c-1][s_{c-1},b_c,br]=vmin;vmin=32767;}
      //     }
      //   }
      // }
      vnew=vold+nok[enc(c,r,0)]*bm;
      if(c==N-1)for(b=0;b<vnew-vold;b++)vold[b]=0;
      np=0;
      for(s0i=0;s0i<nok[encp(c-1,r,0)];s0i++){
        s0=ok[encp(c-1,r,0)][s0i];
        psb1=1000;
        for(sb1i=0;sb1i<nok2[enc2(c,r)];sb1i++){
          sb1=ok2[enc2(c,r)][sb1i];
          s1i=okinv(c,r,0,sb1&15);
          b1i=okinv(c,r,1,sb1>>4);
          pre[np][0]=s1i;
          pre[np][1]=QB(c,r,0,0,sb1&15,sb1>>4)+QB(c,r,0,1,sb1&15,s0);
          pre[np][2]=0;
          if(sb1i>0){assert(np>0);pre[np-1][2]=((sb1>>4)>(psb1>>4));}
          pre[np][3]=s0i+nok[encp(c-1,r,0)]*b1i;
          psb1=sb1;
          np++;
        }
        assert(np>0);pre[np-1][2]=1;
      } 
      mul0=nok[enc(c,r,0)];
      mul1=nok[encp(c-1,r,0)]*nok[enc(c,r,1)];
#ifdef PARALLEL
#pragma omp parallel for
#endif
      for(br=0;br<bm;br++){
        int p,v,vmin;
        vmin=32767;
        for(p=0;p<np;p++){
          v=vold[pre[p][0]+mul0*br]+pre[p][1];
          if(v<vmin)vmin=v;
          if(pre[p][2]){
            if(c>0)vnew[pre[p][3]+mul1*br]=vmin; else v0[pre[p][3]+mul1*br]+=vmin;
            vmin=32767;
          }
        }
      }
      vold=vnew;
    }//c
    // At this point v0 maps (*,r,1) to value of (*,<=r,*)
    t1+=cpu();
    if(0){
      int i,t,v,maxd,nsb[256];
      int64 b,p,b0,b1,b2,np,me,stats[4*N+1];
      double t0,t1;
      for(i=1,nsb[0]=0;i<256;i++)nsb[i]=nsb[i>>1]+(i&1);
      maxd=MIN(4*N,2);
      for(i=0,np=0,b=1;i<=maxd;i++){np+=b;b=(b*(4*N-i))/(i+1);}
      int64 sb[np];
      for(i=0,p=0;i<=maxd;i++)for(b=0;b<size0;b++){
        for(b0=b,t=0;b0;b0>>=8)t+=nsb[b0&255];
        if(t==i)sb[p++]=b;
      }
      assert(p==np);
      for(t=0;t<=maxd;t++)stats[t]=0;
      printf("Row %d\n",r);
      me=0;t0=t1=0;
      for(b0=0;b0<size0;b0++){
        for(b1=1;b1<p;b1++){
          b2=sb[b1];
          v=v0[b0]-v0[b0^b2];
          if(v>0){
            for(b=b2,t=0;b;b>>=8)t+=nsb[b&255];
            if(v>=2*t){
              stats[t]++;t0+=1;t1+=t;
              //printf("%08llx dominated by %08llx. Exor %08llx. valdif %d. Distance %d\n",b0,b0^b2,b2,v,t);
              break;
            }
          }
        }//b1
        if(b1==p){me++;if(0){printf("    %0*llX maximal, value %d\n",N,b0,v0[b0]);fflush(stdout);}}
      }//b0
      for(t=0;t<=maxd;t++)if(stats[t])printf("Num %2d = %lld\n",t,stats[t]);
      printf("%lld maximal element%s\n",me,me==1?"":"s");
      printf("Average distance of dominator: %g\n",t1/t0);
      printf("\n");
      fflush(stdout);
    }

    // Strut exhaust
    //
    //     *b0       *bc       *         *
    //     |         |         |         |
    //     |         ^         |         |
    //     |         |         |         |
    //     *         *s        *b2       *b3
    //
    // (c=1 picture)
    
    t2-=cpu();
    for(c=0;c<N;c++){
      if(c&1){vold=v1;vnew=v0;} else {vold=v0;vnew=v1;}
      // At this point vold maps (>=c,r,1), (<c,r+1,1) to the value below these vertices
      for(x=c+1,bm=1;x<N;x++)bm*=nok[enc(x,r,1)];
      for(x=0;x<c;x++)bm*=nok[encp(x,r+1,1)];
      mul0=nok[enc(c,r,1)];
      mul1=nok[encp(c,r+1,1)];
      assert(bm*MAX(mul0,mul1)<=size0);
      for(bc0=0;bc0<mul1;bc0++){// bc = state of (c,r+1,1)
        bc=ok[encp(c,r+1,1)][bc0];
        for(s0=0;s0<mul0;s0++){// s = state of (c,r,1)
          s=ok[enc(c,r,1)][s0];
          pre2[bc0][s0]=QB(c,r,1,2,s,bc);
        }
      }
#ifdef PARALLEL
#pragma omp parallel for
#endif
      for(br=0;br<bm;br++){// br = state of non-c columns
        int v,vmin,bc0,s0;
        for(bc0=0;bc0<mul1;bc0++){// bc = state of (c,r+1,1)
          vmin=1000000000;
          for(s0=0;s0<mul0;s0++){// s = state of (c,r,1)
            v=vold[s0+mul0*br]+pre2[bc0][s0];
            if(v<vmin)vmin=v;
          }
          vnew[br+bm*bc0]=vmin+offset;// offset keeps the intermediate numbers smaller, allowing bigger range
        }
      }
    }//c
    if(N&1)memcpy(v0,v1,size0*sizeof(short));
    // Now v0 maps (*,r+1,1) to value of (*,r+1,1),(*,<=r,*)
    t2+=cpu();

  }//r

  v=v0[0];
  free(v1);free(v0);
  applyam(0,XBa0,QBa0,ok0,nok0,ok20,nok20);
  printf("Setup time %8.2fs\nComb time  %8.2fs\nStrut time %8.2fs\n",t0,t1,t2);
  return v-N*N*offset;
}

void pr16(int t[16][16]){
  int i,j;
  for(i=0;i<16;i++){
    for(j=0;j<16;j++)printf(" %4d",t[i][j]);
    printf("\n");
  }
}

void combLB2(int r,int (*f)[16][16]){
  // f[N-1][16][16] are the approximators, to be returned
  int c,v,b0,b1,s0,s1,vmin;
  int ex[16][16];// excess
  int t[16][16];
  for(s1=0;s1<16;s1++){// s1 = state of (N-1,r,0)
    for(b1=0;b1<16;b1++){// b1 = state of (N-1,r,1)
      ex[s1][b1]=QB(N-1,r,0,0,s1,b1);
    }
  }
  for(c=N-2;c>=0;c--){// approximating the (c,r,*), (c+1,r,*) part with f[c][][]
    //      
    //                  *b0       *b1
    //                 /         /
    //                /         /
    //    *----------*---------*---- ...
    //    s_hint     s0        s1
    //
    for(s0=0;s0<16;s0++){// s0 = (c,r,0)
      for(b1=0;b1<16;b1++){// b1 = (c+1,r,1)
        vmin=1000000000;
        for(s1=0;s1<16;s1++){// s1 = (c+1,r,0)
          v=QB(c,r,0,2,s0,s1)+ex[s1][b1];
          if(v<vmin)vmin=v;
        }
        t[s0][b1]=vmin;
      }//b1
    }//s0
    for(b0=0;b0<16;b0++){// b0 = (c,r,1)
      for(b1=0;b1<16;b1++){// b1 = (c+1,r,1)
        vmin=1000000000;
        for(s0=0;s0<16;s0++){// s0 = (c,r,0)
          v=QB(c,r,0,1,s0,XB(c-1,r,0))+QB(c,r,0,0,s0,b0)+t[s0][b1];
          if(v<vmin)vmin=v;
        }
        f[c][b0][b1]=vmin;
      }//b1
    }//b0
    for(b0=0;b0<16;b0++){// b0 = (c,r,1)
      for(s0=0;s0<16;s0++){// s0 = (c,r,0)
        vmin=1000000000;
        for(b1=0;b1<16;b1++){// b1 = (c+1,r,1)
          v=QB(c,r,0,0,s0,b0)+t[s0][b1]-f[c][b0][b1];
          if(v<vmin)vmin=v;
        }
        ex[s0][b0]=vmin;
      }//s0
    }//b0
  }//c
  for(b0=0;b0<16;b0++){// b0 = (c,r,1)
    vmin=1000000000;
    for(s0=0;s0<16;s0++){// s0 = (c,r,0)
      if(ex[s0][b0]<vmin)vmin=ex[s0][b0];
    }
    assert(vmin==0);
  }
}

void reducerankLB(int t[16][16],int t0[16],int t1[16]){
  int d,i,j,dd,nd,i1,j1,nmax,dmin,imin,jmin,vmin,n0[16],n1[16];
  vmin=1000000000;imin=jmin=-1;
  for(i=0;i<16;i++)for(j=0;j<16;j++)if(t[i][j]<vmin){vmin=t[i][j];imin=i;jmin=j;}
  for(i=0;i<16;i++){t0[i]=t[i][jmin];t1[i]=t[imin][i]-vmin;}

  nd=0;for(i=0;i<16;i++)n0[i]=n1[i]=0;
  for(i=0;i<16;i++)for(j=0;j<16;j++)if(t0[i]+t1[j]>t[i][j]){n0[i]++;n1[j]++;nd++;}
  while(nd>0){

    if(0){
      printf("             ");for(j=0;j<16;j++)printf(" %4d",n1[j]);printf("\n");
      printf("             ");for(j=0;j<16;j++)printf(" %4d",t1[j]);printf("\n");
      for(i=0;i<16;i++){
        printf("%4d %4d : ",n0[i],t0[i]);
        for(j=0;j<16;j++)printf(" %4d",t0[i]+t1[j]-t[i][j]);
        printf("\n");
      }
      printf("\n");
    }

    nmax=0;dd=-1;
    for(i=0;i<16;i++){
      if(n0[i]>nmax){nmax=n0[i];dd=i;}
      if(n1[i]>nmax){nmax=n1[i];dd=16+i;}
    }
    if(dd<16){
      dmin=1000000000;j1=-1;
      for(j=0;j<16;j++){
        d=(t0[dd]+t1[j])-t[dd][j];
        if(d>0&&d<dmin){dmin=d;j1=j;}
      }
      assert(j1>=0);
      t0[dd]-=dmin;
      for(j=0;j<16;j++)if(t0[dd]+t1[j]==t[dd][j]){n0[dd]--;n1[j]--;nd--;assert(n0[dd]>=0&&n1[j]>=0&&nd>=0);}
    }else{
      dd-=16;
      dmin=1000000000;i1=-1;
      for(i=0;i<16;i++){
        d=(t0[i]+t1[dd])-t[i][dd];
        if(d>0&&d<dmin){dmin=d;i1=i;}
      }
      assert(i1>=0);
      t1[dd]-=dmin;
      for(i=0;i<16;i++)if(t0[i]+t1[dd]==t[i][dd]){n0[i]--;n1[dd]--;nd--;assert(n0[i]>=0&&n1[dd]>=0&&nd>=0);}
    }
  }
}

void combLB(int r,int w,int *f){
  // f thought of as f[N-w+1][16^w] are the approximators, to be returned
  // So f_0(a_0,...,a_{w-1})+f_1(a_1,...,a_w)+...+f_{N-w}(a_{N-w},...,a_{N-1}) <= comb_r(a_0,...,a_{N-1})
  // Multiindices encoded low to high, e.g., (a_0,...,a_{w-1}) <-> a_0+16a_1+16^2a_2+...
  
  int c,v,s0,s1,vmin,vtot,tt[16];
  int64 b,n;
  n=1LL<<(4*w);
  int v0[n],v1[n];
  if(w==1)for(b=0;b<16;b++)v0[b]=0; else {
    // Add in (0,r,0) (0,r,1):
    for(b=0;b<256;b++){// b = (0,r,0) (0,r,1)
      v0[b]=QB(0,r,0,0,b&15,b>>4);
    }
    for(c=1;c<w-1;c++){
      // State: v0[ (c-1,r,0) (0,r,1) (1,r,1) ... (c-1,r,1) ]   (low - high)
      // Add in (c,r,0) and minimise over (c-1,r,0):
      for(b=0;b<(1LL<<(c+1)*4);b++){// b = (c,r,0) (0,r,1) (1,r,1) ... (c-1,r,1)
        vmin=1000000000;
        for(s0=0;s0<16;s0++){// s0=(c-1,r,0)
          v=v0[(b&~15)|s0]+QB(c-1,r,0,2,s0,b&15);
          if(v<vmin)vmin=v;
        }
        v1[b]=vmin;
      }//b
      // Add in (c,r,1):
      for(b=0;b<(1LL<<(c+2)*4);b++){// b = (c,r,0) (0,r,1) (1,r,1) ... (c,r,1)
        v0[b]=v1[b&~(15LL<<(c+1)*4)]+QB(c,r,0,0,b&15,b>>(c+1)*4);
      }
    }//c
  }
  for(c=w-1;c<N;c++){
    // State: v0[ (c-1,r,0) (c-w+1,r,1) (c-w+2,r,1) ... (c-1,r,1) ]   (low - high)
    // Add in (c,r,0) and minimise over (c-1,r,0):
    for(b=0;b<n;b++){// b = (c,r,0) (c-w+1,r,1) (c-w+2,r,1) ... (c-1,r,1)
      vmin=1000000000;
      for(s0=0;s0<16;s0++){// s0=(c-1,r,0)
        v=v0[(b&~15)|s0]+QB(c,r,0,1,b&15,s0);
        if(v<vmin)vmin=v;
      }
      v1[b]=vmin;
    }//b
    // Add in (c,r,1) and minimise over (c,r,0), using (c+1,r,0)_ave (sidebranch)
    for(s0=0;s0<16;s0++){
      vtot=0;
      for(s1=0;s1<16;s1++)vtot+=QB(c,r,0,2,s0,s1);
      tt[s0]=(vtot+8)>>4;
    }
    for(b=0;b<n;b++){// b = (c-w+1,r,1) (c-w+2,r,1) ... (c,r,1)
      vmin=1000000000;
      for(s0=0;s0<16;s0++){// s0=(c,r,0)
        v=v1[((b<<4)&~(15LL<<(4*w)))|s0]+QB(c,r,0,0,s0,b>>(4*(w-1)))+tt[s0];
        if(v<vmin)vmin=v;
      }
      f[(int64)(c-(w-1))<<(4*w)|b]=vmin;
    }//b
    if(c==N-1)break;
    // Add in (c,r,1) again, subtract f[c-(w-1)][] and minimise over (c-w+1,r,1)
    if(w==1){
      for(b=0;b<n;b++){// b = (c,r,0)
        vmin=1000000000;
        for(s0=0;s0<16;s0++){// s0=(c,r,1)
          v=v1[b]+QB(c,r,0,0,b&15,s0)-f[(c<<4)|s0];
          if(v<vmin)vmin=v;
        }
        v0[b]=vmin;
      }//b
    }else{
      for(b=0;b<n;b++){// b = (c,r,0) (c-w+2,r,1) (c-w+3,r,1) ... (c,r,1)
        vmin=1000000000;
        for(s0=0;s0<16;s0++){// s0=(c-w+1,r,1)
          v=v1[(((b&~15)<<4)&~(15LL<<4*w))|(b&15)|(s0<<4)]+QB(c,r,0,0,b&15,b>>4*(w-1))-f[(int64)(c-(w-1))<<(4*w)|(b&~15)|s0];
          if(v<vmin)vmin=v;
        }
        v0[b]=vmin;
      }//b
    }
  }//c
}

int lin2LB(){
  int c,i,j,r,v,vmin,b0,b1,b2,s0,s1;
  int m0[16],m1[16],t[16][16],t0[16],t1[16],u[16][16],f0[N-1][16][16],f1[N-1][16][16];
  for(c=0;c<N-1;c++)for(i=0;i<16;i++)for(j=0;j<16;j++)f0[c][i][j]=0;
  for(r=0;r<N;r++){
    if(0){
      combLB2(r,f1);
      for(c=0;c<N-1;c++)for(i=0;i<16;i++)for(j=0;j<16;j++)f1[c][i][j]+=f0[c][i][j];
    }else{
      combLB(r,2,(int*)f1);
      for(c=0;c<N-1;c++)for(i=0;i<16;i++)for(j=0;j<16;j++)f0[c][i][j]=f0[c][i][j]+f1[c][j][i];
      for(c=0;c<N-1;c++)for(i=0;i<16;i++)for(j=0;j<16;j++)f1[c][i][j]=f0[c][i][j];
    }
    if(r==N-1)break;
    //
    // Struts: build f0 (new row r+1) from f1 (old row r)
    //
    //     *b0     *b1     *b2     *  
    //     |       |       |       |
    //     |       |       ^       |
    //     |       |       |       |
    //     *       *       *s2     *b3
    // (c=2 picture)
    //
    for(b0=0;b0<16;b0++)for(b1=0;b1<16;b1++){
      vmin=1000000000;
      for(s0=0;s0<16;s0++){
        v=f1[0][s0][b1]+QB(0,r,1,2,s0,b0);
        if(v<vmin)vmin=v;
      }
      f0[0][b0][b1]=vmin;
    }
    for(c=1;c<N-1;c++){
      for(b1=0;b1<16;b1++){// b1=(c,r,1)
        for(b0=0;b0<16;b0++)for(b2=0;b2<16;b2++){// b0=(c-1,r,1), b2=(c+1,r,1)
          vmin=1000000000;
          for(s1=0;s1<16;s1++){
            v=f0[c-1][b0][s1]+f1[c][s1][b2]+QB(c,r,1,2,s1,b1);
            if(v<vmin)vmin=v;
          }
          t[b0][b2]=vmin;
        }// b0,b2
        reducerankLB(t,t0,t1);
        for(b0=0;b0<16;b0++)u[b0][b1]=t0[b0];
        for(b2=0;b2<16;b2++)f0[c][b1][b2]=t1[b2];
      }// b1
      for(b0=0;b0<16;b0++)for(b1=0;b1<16;b1++)f0[c-1][b0][b1]=u[b0][b1];
    }// c
    for(b0=0;b0<16;b0++)for(b1=0;b1<16;b1++){
      vmin=1000000000;
      for(s1=0;s1<16;s1++){
        v=f0[N-2][b0][s1]+QB(N-1,r,1,2,s1,b1);
        if(v<vmin)vmin=v;
      }
      u[b0][b1]=vmin;
    }
    for(b0=0;b0<16;b0++)for(b1=0;b1<16;b1++)f0[N-2][b0][b1]=u[b0][b1];
  }// r
  // Result in f1
  for(b0=0;b0<16;b0++)m0[b0]=0;
  for(c=0;c<N-1;c++){
    // m0[] is a map from (c,N-1,1) to minval of (<=c,N-1,1) using the c functions, f1[<c][][]
    for(b0=0;b0<16;b0++){// b0=(c+1,N-1,1)
      vmin=1000000000;
      for(s0=0;s0<16;s0++){// s0=(c,N-1,1)
        v=m0[s0]+f1[c][s0][b0];
        if(v<vmin)vmin=v;
      }
      m1[b0]=vmin;
    }// b0
    for(b0=0;b0<16;b0++)m0[b0]=m1[b0];
  }// c
  vmin=1000000000;
  for(s0=0;s0<16;s0++){// s0=(N-1,N-1,1)
    v=m0[s0];
    if(v<vmin)vmin=v;
  }
  return vmin;
}

int linLB(int w){
  int n,r,v,vmin,s0;
  n=N-w+1;// Using n overlapping width-w functions
  int64 b,c,d,b0,b1,N1,N2,N3,hit;
  N1=n*(1LL<<4*w);
  N2=1LL<<4*(w-1);
  N3=1LL<<4*2*(w-1);
  int f0[N1],f1[N1],t0[N3],t1[N3];
  int64 hi[N2];

  memset(f1,0,N1*sizeof(int));
  for(r=0;r<N;r++){
    // Here f1[] is the bound on (*,<r,1)
    combLB(r,w,f0);
    for(b=0;b<N1;b++)f0[b]+=f1[b];
    if(r==N-1)break;
    // f0 at the bottom (row r) -> construct f1 at the top (row r+1)
    for(b1=0;b1<N2;b1++)for(b0=0;b0<N2;b0++){// b0=(<w-1,r,1) b1=(<w-1,r+1,1)
      v=0;
      for(c=0;c<w-1;c++)v+=QB(c,r,1,2,(b0>>4*c)&15,(b1>>4*c)&15);
      t0[b0+(b1<<4*(w-1))]=v;
    }
    for(c=0;c<n;c++){
      // t0[] maps (c..c+w-2,r,1),(c..c+w-2,r+1,1) to sum_{i<c}(f_i'-f_i) + struts
      for(b1=0;b1<N2;b1++)for(b0=0;b0<N2;b0++){// b0=(c+1..c+w-1,r,1) b1=(c..c+w-2,r+1,1)
        vmin=1000000000;
        for(s0=0;s0<16;s0++){// s0=(c,r,1)
          v=t0[s0+((b0<<4)&~(15LL<<4*(w-1)))+(b1<<4*(w-1))]+f0[s0+(b0<<4)+(c<<4*w)];
          if(v<vmin)vmin=v;
        }
        t1[b0+(b1<<4*(w-1))]=vmin;
      }
      // t1[] maps (c+1..c+w-1,r,1),(c..c+w-2,r+1,1) to sum_{i<c}(f_i'-f_i)+f_c' + struts
      assert(w>=2);//FTM
      for(b=0;b<N2;b++)hi[b]=0;
      for(d=1;d<MIN(w,n-c);d++){
        for(b0=0;b0<1LL<<(w-d)*4;b0++){// b0=(c+d..c+w-1,r,1)
          hit=0;
          for(b1=0;b1<1LL<<d*4;b1++){// b1=(c+w..c+w+d-1,r,1)
            hit+=f0[b0+(b1<<(w-d)*4)+((c+d)<<4*w)];
          }
          for(b1=0;b1<1LL<<(d-1)*4;b1++){// b1=(c+1..c+d-1,r,1)
            hi[b1+(b0<<(d-1)*4)]+=hit<<(w-1-d)*4;
          }
        }
      }
      for(b=0;b<N2;b++)hi[b]=(hi[b]+(1LL<<((w-1)*4-1)))>>(w-1)*4;
      for(b1=0;b1<1LL<<4*w;b1++){// b1=(c+w-1,r,1),(c..c+w-2,r+1,1)
        vmin=1000000000;
        for(b0=0;b0<1LL<<4*(w-2);b0++){// b0=(c+1..c+w-2,r,1)
          v=t1[b0+(b1<<4*(w-2))]+hi[b0+((b1&15)<<4*(w-2))];
          if(v<vmin)vmin=v;
        }
        t0[b1]=vmin;
      }
      // t0[] maps (c+w-1,r,1),(c..c+w-2,r+1,1) to sum_{i<c}(f_i'-f_i)+f_c' + struts
      for(b=0;b<1LL<<4*w;b++){// b=(c..c+w-1,r+1,1)
        vmin=1000000000;
        for(s0=0;s0<16;s0++){// s0=(c+w-1,r,1)
          v=t0[s0+((b<<4)&~(15LL<<4*w))]+QB(c+w-1,r,1,2,s0,b>>4*(w-1));
          if(v<vmin)vmin=v;
        }
        f1[b+(c<<4*w)]=vmin;
      }
      for(b1=0;b1<N2;b1++)for(b0=0;b0<N2;b0++){// b0=(c+1..c+w-1,r,1) b1=(c+1..c+w-1,r+1,1)
        vmin=1000000000;
        for(s0=0;s0<16;s0++){// s0=(c,r+1,1)
          v=t1[b0+((s0+((b1<<4)&~(15LL<<4*(w-1))))<<4*(w-1))]-f1[s0+(b1<<4)+(c<<4*w)];
          if(v<vmin)vmin=v;
        }
        t0[b0+(b1<<4*(w-1))]=vmin+QB(c+w-1,r,1,2,b0>>4*(w-2),b1>>4*(w-2));
      }
    }//c
  }//r
  // Result in f0
  for(b=0;b<N2;b++)t0[b]=0;
  for(c=0;c<n;c++){
    // t0[] is a map from (c..c+w-2,N-1,1) to minval of the sum of the c functions, f0[<c]
    // Add variable (c+w-1,N-1,1) and min over (c,N-1,1)
    for(b=0;b<N2;b++){// b=(c+1..c+w-1,N-1,1)
      vmin=1000000000;
      for(s0=0;s0<16;s0++){// s0=(c,N-1,1)
        v=t0[s0+((b<<4)&~(15LL<<4*(w-1)))]+f0[s0+(b<<4)+(c<<4*w)];
        if(v<vmin)vmin=v;
      }
      t1[b]=vmin;
    }
    for(b=0;b<N2;b++)t0[b]=t1[b];
  }
  vmin=1000000000;
  for(b=0;b<N2;b++)if(t0[b]<vmin)vmin=t0[b];
  return vmin;
}

int main(int ac,char**av){
  int opt,wn,mode,strat,weightmode,numpo;
  double mint,maxt;
  char *inprobfile,*outprobfile,*outstatefile;

  wn=-1;inprobfile=outprobfile=outstatefile=0;seed=time(0);mint=10;maxt=1e10;statemap[0]=0;statemap[1]=1;
  weightmode=7;mode=1;N=8;strat=3;deb=1;ext=1;numpo=500;ngp=0;
  while((opt=getopt(ac,av,"m:n:N:o:O:p:P:s:S:t:T:v:w:x:X:"))!=-1){
    switch(opt){
    case 'm': mode=atoi(optarg);break;
    case 'n': wn=atoi(optarg);break;
    case 'N': N=atoi(optarg);break;
    case 'o': outprobfile=strdup(optarg);break;
    case 'O': outstatefile=strdup(optarg);break;
    case 'p': numpo=atoi(optarg);break;
    case 'P':
      {
        char *l=optarg;
        while(1){
          assert(ngp<MAXNGP);genp[ngp++]=atof(l);
          l=strchr(l,',');if(!l)break;
          l++;
        }
      }
      break;
    case 's': seed=atoi(optarg);break;
    case 'S': strat=atoi(optarg);break;
    case 't': mint=atof(optarg);break;
    case 'T': maxt=atof(optarg);break;
    case 'v': deb=atoi(optarg);break;
    case 'w': weightmode=atoi(optarg);break;
    case 'x': statemap[0]=atoi(optarg);break;
    case 'X': ext=atof(optarg);break;
    default:
      fprintf(stderr,"Usage: %s [OPTIONS] [inputproblemfile]\n",av[0]);
      fprintf(stderr,"       -m   mode of operation:\n");
      fprintf(stderr,"            0   Try to find minimum value by heuristic search\n");
      fprintf(stderr,"            1   Try to find rate of solution generation by repeated heuristic search (default)\n");
      fprintf(stderr,"            2   Try to find expected minimum value by heuristic search\n");
      fprintf(stderr,"            3   (no longer used)\n");
      fprintf(stderr,"            4   Consistency checks\n");
      fprintf(stderr,"            5   Full exhaust (proving)\n");
      fprintf(stderr,"       -n   num working nodes (default all)\n");
      fprintf(stderr,"       -N   size of Chimera graph (default 8)\n");
      fprintf(stderr,"       -o   output problem (weight) file\n");
      fprintf(stderr,"       -O   output state file\n");
      fprintf(stderr,"       -p   target number of presumed optima for -m1\n");
      fprintf(stderr,"       -P   x[,y[,z,...]] general parameters for test code\n");
      fprintf(stderr,"       -s   seed\n");
      fprintf(stderr,"       -S   search strategy for heuristic search (0,1,2)\n");
      fprintf(stderr,"            0      Exhaust K44s repeatedly\n");
      fprintf(stderr,"            1      Exhaust lines repeatedly\n");
      fprintf(stderr,"            3      Exhaust maximal treewidth 1 subgraphs (default)\n");
      fprintf(stderr,"            4      Exhaust maximal treewidth 2 subgraphs\n");
      fprintf(stderr,"            5      Exhaust maximal treewidth 3 subgraphs\n");
      fprintf(stderr,"            10+n   As strategy n but with partial random state init\n");
      fprintf(stderr,"       -t   min run time for some modes\n");
      fprintf(stderr,"       -T   max run time for some modes\n");
      fprintf(stderr,"       -v   0,1,2,... verbosity level\n");
      fprintf(stderr,"       -w   weight creation convention\n");
      fprintf(stderr,"            0   All of Q_ij independently +/-1\n");
      fprintf(stderr,"            1   As 0, but diagonal not allowed\n");
      fprintf(stderr,"            2   Upper triangular\n");
      fprintf(stderr,"            3   All of Q_ij allowed, but constrained symmetric\n");
      fprintf(stderr,"            4   Constrained symmetric, diagonal not allowed - the basic Ising model mode\n");
      fprintf(stderr,"            5   Start with Ising J_ij (i<j) and h_i IID {-1,1} and transform back to QUBO,\n");
      fprintf(stderr,"                ignoring constant term. (Default - meant to be equivalent to McGeoch instances.)\n");
      fprintf(stderr,"            6   Test case\n");
      fprintf(stderr,"            7   Start with Ising J_ij (i<j) IID {-1,1} and transform back to QUBO,\n");
      fprintf(stderr,"                also known as \"no external field\".\n");
      fprintf(stderr,"            8   Start with Ising J_ij (i<j) IID uniform in {-100,-99,...,100} and\n");
      fprintf(stderr,"                transform back to QUBO.\n");
      fprintf(stderr,"           10   Start with Ising J_ij (i<j) IID uniform in {-n,-n+1,...,n} where n=100 intra-K_44,\n");
      fprintf(stderr,"                n=220 inter-K_44, then transform back to QUBO.\n");
      fprintf(stderr,"       -x   set the lower state value\n");
      fprintf(stderr,"            Default 0 corresponds to QUBO state values in {0,1}\n");
      fprintf(stderr,"            Other common option is -1, corresponding to Ising model state values in {-1,1}\n");
      exit(1);
    }
  }
  if(wn<0)wn=NV; else assert(wn<=NV);
  if(optind<ac)inprobfile=strdup(av[optind]);
  printf("N=%d\n",N);
  printf("Mode: %d\n",mode);
  printf("Seed: %d\n",seed);
  printf("Search strategy: %d\n",strat);
  if(ngp>0){int i;printf("General parameters:");for(i=0;i<ngp;i++)printf(" %g",genp[i]);printf("\n");}

  Q=(int(*)[4][7])malloc(NBV*4*7*sizeof(int));
  adj=(int(*)[4][7][2])malloc(NBV*4*7*2*sizeof(int));
  okv=(int(*)[4])malloc(NBV*4*sizeof(int));
  XBplus=(int*)calloc((N+2)*N*2*sizeof(int),1);
  XBa=XBplus+N*2;
  QBa=(int(*)[3][16][16])malloc(NBV*3*16*16*sizeof(int));
  ok=(int(*)[16])malloc((NBV+1)*16*sizeof(int));
  nok=(int*)malloc((NBV+1)*sizeof(int));
  ok2=(int(*)[256])malloc((N*N+1)*256*sizeof(int));
  nok2=(int*)malloc((N*N+1)*sizeof(int));
  inittiebreaks();
  initrand(seed);
  initgraph(wn);
  if(inprobfile){
    wn=readweights(inprobfile);// This overrides current setting of wn
  }else{
    initweights(weightmode);printf("Initialising random weight matrix with %d working node%s\n",wn,wn==1?"":"s");
  }
  if(0){
    int i,j,k;
    for(i=0;i<N;i++)for(j=0;j<16;j++)for(k=0;k<16;k++)printf("QB(%d,0,0,0,%d,%d)=%d\n",i,j,k,QB(i,0,0,0,j,k));
    for(i=0;i<N-1;i++)for(j=0;j<16;j++)for(k=0;k<16;k++)printf("QB(%d,0,0,2,%d,%d)=%d\n",i,j,k,QB(i,0,0,2,j,k));
  }
  printf("%d working node%s out of %d\n",wn,wn==1?"":"s",NV);
  printf("States are %d,%d\n",statemap[0],statemap[1]);
  printf("Weight-choosing mode: %d\n",weightmode);
  if(outprobfile){writeweights(outprobfile);printf("Wrote weight matrix to file \"%s\"\n",outprobfile);}
  int v;
  double t0;
  switch(mode){
  case 0:// Find minimum value using heuristic strategy strat, not worrying about independence of subsequent minima
    opt1(mint,maxt,deb,1,0,strat);
    break;
  case 1:;// Find rate of solution generation, ensuring that minima are independent
    {
      double tts;
      v=opt1(0.5,maxt,deb,numpo,&tts,strat);
      printf("Time to solution %gs, assuming true minimum is %d\n",tts,v);
      break;
    }
  case 2:;// Find average minimum value
    {
      double s0,s1,s2,va;
      s0=s1=s2=0;
      while(1){
        initweights(weightmode);
        v=opt1(0,maxt,0,500,0,strat);
        s0+=1;s1+=v;s2+=v*v;va=(s2-s1*s1/s0)/(s0-1);
        printf("%12g %12g %12g %12g\n",s0,s1/s0,sqrt(va),sqrt(va/s0));
      }
    }
    break;
  case 4:;// Checks
    {
      opt1(mint,maxt,1,1,0,strat);
      printf("Full exhaust %d\n",stripexhaust(0,0,N,0));
      int o,c0,c1;
      for(o=0;o<2;o++)for(c0=0;c0<N;c0++)for(c1=c0+1;c1<=N;c1++){
        v=stripexhaust(o,c0,c1,0)+stripval(o,0,c0)+stripval(o,c1,N);
        printf("Strip %2d %2d %2d   %6d\n",o,c0,c1,v);
      }
      break;
    }
  case 5:// Prove using subset method v2
    {
      printf("Restricted set exhaust\n");
      t0=cpu();
      v=fullexhaust();
      printf("Optimum %d found in %gs\n",v,cpu()-t0);
      break;
    }
  case 6:
    readstate("state");printf("state = %d\n",val());break;
  case 8:// timing tests
    {
      int d,n,r,c0,c1,ph,wid,v0,upd;
      double t0;
      opt1(mint,maxt,1,1,0,strat);
      init_state();
      printf("val=%d\n",val());
      upd=0;
      wid=5;
      for(d=0;d<2;d++)for(ph=0;ph<=wid;ph++)for(r=0;r<N;r++){
        for(n=0,t0=cpu();(n&(n-1))||cpu()-t0<.5;n++)v0=treestripexhaust(d,wid,ph,upd,r);
        printf("treestripexh %d %d %d %2d   %6d   %gs\n",d,wid,ph,r,v0,(cpu()-t0)/n);
        fflush(stdout);
      }
      for(d=0;d<2;d++)for(ph=0;ph<2;ph++)for(r=0;r<N;r++){
        v0=1000000000;
        for(n=0,t0=cpu();(n&(n-1))||cpu()-t0<.5;n++)v0=tree1exhaust(d,ph,r,0);
        printf("tree1 %d %d %2d   %6d   %gs\n",d,ph,r,v0,(cpu()-t0)/n);
        fflush(stdout);
      }
      for(d=0;d<2;d++)for(c0=0;c0<N-wid+1;c0++){
        c1=c0+wid;v0=0;
        for(n=0,t0=cpu();(n&(n-1))||cpu()-t0<.5;n++)v0=stripexhaust(d,c0,c1,upd);
        v0+=stripval(d,0,c0)+stripval(d,c1,N);
        printf("Strip %d %2d %2d   %6d   %gs\n",d,c0,c1,v0,(cpu()-t0)/n);
        if(upd)assert(v0==val());
        fflush(stdout);
      }
    }
    break;
  case 9:// consistency checks
    {
      int c,d,o,w,lw,ph,phl,r,v0,v1;
      //opt1(mint,maxt,1,1,0,strat);
      printf("val=%d\n",val());
      if(0){
        writeweights("prob");
        v0=treestripexhaust(0,1,0,1,0);
        v1=val();
        printf("%6d %6d\n",v0,v1);
        assert(v0==v1);
        exit(0);
      }
      while(1){
        initweights(weightmode);
        init_state();
        for(d=0;d<2;d++)for(ph=0;ph<2;ph++)for(r=0;r<N;r++){
          v0=treestripexhaust(d,1,ph,0,r);
          v1=tree1exhaust(d,ph,r,0);
          printf("tree1 %d %d %2d   %6d   %6d\n",d,ph,r,v0,v1);
          assert(v0==v1);
        }
        for(w=1;w<=3;w++){
          for(d=0;d<2;d++)for(ph=0;ph<=w;ph++)for(r=-1;r<N;r++){
            init_state();
            opt1(0,maxt,0,1,0,strat);
            v0=treestripexhaust(d,w,ph,0,r);
            for(c=0;c<N;){
              phl=(c+ph)%(w+1);
              if(phl==w){c++;continue;}
              lw=MIN(w-phl,N-c);
              stripexhaust(d,c,c+lw,1);
              c+=lw;
            }
            v1=val();
            printf("stripexhcomp %d %d %d %2d   %6d %6d\n",w,d,ph,r,v0,v1);
            assert(v0<=v1);
          }
        }
        for(w=1;w<=3;w++){
          for(d=0;d<2;d++)for(ph=0;ph<=w;ph++){
            init_state();
            opt1(mint,maxt,0,1,0,strat);
            v0=val();
            for(c=0;c<N;c++){
              phl=(c+ph)%(w+1);
              for(o=0;o<2;o++){
                if(!(phl==w&&o==0))for(r=0;r<N;r++)XBI(d,c,r,o)=randnib();
              }
            }
            v1=treestripexhaust(d,w,ph,0,-1);
            printf("stripexhspiketest %d %d %d %2d   %6d %6d\n",w,d,ph,r,v0,v1);
            assert(v1<=v0);
          }
        }
        for(w=1;w<=3;w++){
          for(d=0;d<2;d++)for(ph=0;ph<=w;ph++)for(r=0;r<N;r++){
            init_state();
            v0=treestripexhaust(d,w,ph,1,r);
            v1=val();
            printf("updcomp %d %d %d %2d   %6d %6d\n",w,d,ph,r,v0,v1);
            assert(v0==v1);
          }
        }            
      }
    }
  case 10:
    {
      int c,r,f[N-1][16][16],g[N-1][16][16];
      if(0){
        int XBa0[NBV],QBa0[NBV][3][16][16],ok0[NBV][16],nok0[NBV],ok20[N*N][256],nok20[N*N];
        init_state();
        memcpy(XBa0,XBa,sizeof(XBa0));memcpy(QBa0,QBa,sizeof(QBa0));
        memcpy(ok0,ok,sizeof(ok0));memcpy(nok0,nok,sizeof(nok0));
        memcpy(ok20,ok2,sizeof(ok20));memcpy(nok20,nok2,sizeof(nok20));
        for(r=0;r<N;r++){
          printf("Comb at row %2d\n",r);
          combLB2(r,f);
          applyam(2,XBa0,QBa0,ok0,nok0,ok20,nok20);
          combLB(r,2,(int*)g);
          applyam(0,XBa0,QBa0,ok0,nok0,ok20,nok20);
          if(1)for(c=0;c<N-1;c++){
            printf("MEMCMP %d\n",memcmp(f[N-2-c],g[c],16*16));
            if(0){
              pr16(f[N-2-c]);printf("\n");
              pr16(g[c]);
              printf("\n---------------------\n\n");
            }
          }
          printf("\n");
        }
      }
      {
        int w,v0;
        init_state();
        v0=lin2LB();
        printf("lin2 = %d\n",v0);
        for(w=2;w<=MIN(N,2);w++){
          printf("lin(%d) =",w);fflush(stdout);
          v0=linLB(w);
          printf(" %d\n",v0);
        }
      }
    }
    break;
  case 11:
    {
      int d,i,m,r,w,dmax;
      int64 b,n;
      double s1;
      init_state();
      //opt1(mint,maxt,deb,1,0,strat);
      for(r=0;r<N;r++){
        n=1LL<<(4*N);
        int g[n];
        combLB(r,N,g);
        for(w=1;w<N;w++){
          m=1LL<<(4*w);
          int f[(N-w+1)*m];
          combLB(r,w,f);
          s1=0;dmax=0;
          for(b=0;b<n;b++){
            v=0;
            for(i=0;i<=N-w;i++)v+=f[(i<<(4*w))+((b>>(4*i))&(m-1))];
            d=g[b]-v;assert(d>=0);
            //if(w==N-1&&d>0)printf("%d ",d);
            s1+=d;if(d>dmax)dmax=d;
            //printf("%10d %10d %10d\n",g[b],v,g[b]-v);
          }
          printf("%3d  %3d  %8d  %12g\n",r,w,dmax,s1/n);
        }
      }//r
    }
    break;
  case 12:
    {
      int i,n,o,t,x,y,nb[16],stats[N][strat][256];
      n=0;
      for(x=0;x<N;x++)for(y=0;y<strat;y++)for(i=0;i<256;i++)stats[x][y][i]=0;
      for(i=1,nb[0]=0;i<16;i++)nb[i]=nb[i>>1]+(i&1);
      while(1){
        for(x=0,t=0;x<N;x++){XB(x,strat,1)=randnib();t+=nb[XB(x,strat,1)];}
        if(t>2*N)for(x=0;x<N;x++)XB(x,strat,1)^=15;
        stripexhaust(1,0,strat,1);
        for(x=0;x<N;x++)for(y=0;y<strat;y++)stats[x][y][XB(x,y,0)<<4|XB(x,y,1)]++;
        n++;
        if(0){
          for(y=strat-1;y>=0;y--){
            for(x=0;x<N;x++){printf(" ");for(o=0;o<2;o++)printf("%X",XB(x,y,o));}
            printf("\n");
          }
          printf("\n");
        }
        if(n%10==0){
          for(y=strat-1;y>=0;y--){
            for(x=0;x<N;x++){
              //double p,s;
              //for(i=0,s=0;i<256;i++)if(stats[x][y][i]){p=stats[x][y][i]/(double)n;s-=p*log(p);}
              //printf(" %7.3f",s/log(2));
              for(i=0,t=0;i<256;i++)if(stats[x][y][i])t++;
              printf(" %3d",t);
            }
            printf("\n");
          }
          printf("\n");
        }
      }
    }
    break;
  case 13:// Gibbs tests
    {
      if(1){// Burn-in test
        int i,n,v,nn=20;
        double mu,va,beta,s0[nn],s1[nn],s2[nn];
        beta=3.0;
        for(i=0;i<nn;i++)s0[i]=s1[i]=s2[i]=0;
        for(n=0;n<10000;n++){
          init_state();
          for(i=0;i<nn;i++){
            tree1gibbs(randint(2),randint(2),randint(N),beta);
            v=val();assert(isfinite(v));
            s0[i]+=1;s1[i]+=v;s2[i]+=v*v;
          }
        }
        for(i=0;i<nn;i++){
          mu=s1[i]/s0[i];
          va=(s2[i]-s1[i]*s1[i]/s0[i])/(s0[i]-1);
          printf("%5d %12g %12g\n",i,mu,sqrt(va/s0[i]));
        }
      }
      if(0){// Autocorrelation test; assumes weightmode 0, statemap[0]=-1 (Ising form, uniform +/-1, no fields)
        int i,j,k,it,bp,nb=20,rep;
        int sbuf[nb][NBV],btab[16];
        double t,mu,va,beta,s0[nb],s1[nb],s2[nb];
        if(weightmode!=0||statemap[0]!=-1)fprintf(stderr,"Warning: expect weightmode=0, statemap[0]=-1\n");
        beta=.8;rep=10000;
        init_state();
        for(i=0;i<(beta+1)*20;i++)tree1gibbs(randint(2),randint(2),randint(N),beta);// burn-in guess
        for(i=0;i<nb;i++)s0[i]=s1[i]=s2[i]=0;
        bp=0;it=-nb;
        for(i=1,btab[0]=4;i<16;i++)btab[i]=btab[i>>1]-2*(i&1);// (# 0 bits) - (# 1 bits)
        while(1){
          for(i=0;i<rep;i++)tree1gibbs(randint(2),randint(2),randint(N),beta);
          memcpy(sbuf[bp],XBa,NBV*sizeof(int));
          if(it>=0){
            for(i=0;i<nb;i++){// Correlate current with "i" ago
              j=bp-i;if(j<0)j+=nb;
              for(k=0,t=0;k<NBV;k++)t+=btab[XBa[k]^sbuf[j][k]];
              s0[i]+=1;s1[i]+=t;s2[i]+=t*t;
            }
            if(it%100==0){
              printf("it=%d\n",it);
              for(i=0;i<nb;i++){
                mu=s1[i]/s0[i];
                va=(s2[i]-s1[i]*s1[i]/s0[i])/(s0[i]-1);
                printf("%6d   %12g %12g\n",i*rep,mu,sqrt(va/s0[i]));
              }
              printf("\n");
            }
          }
          it++;bp++;if(bp==nb)bp=0;
        }
      }
    }
    break;
  case 14:// Binder parameter estimate
    {
      int i,j,k,n,nd,nb=6,burnin;
      int sbuf[nb][NBV],btab[16];
      double q,x,t0,t1,beta,sp[9];
      for(i=1,btab[0]=4;i<16;i++)btab[i]=btab[i>>1]-2*(i&1);// (# 0 bits) - (# 1 bits)
      beta=0.1;
      burnin=0;//(beta+1)*50;// burn-in guess
      if(weightmode!=0||statemap[0]!=-1)fprintf(stderr,"Warning: expect weightmode=0, statemap[0]=-1\n");
      printf("beta = %g\n",beta);
      printf("burn-in = %d\n",burnin);
      nd=-1;//5964;
      for(i=0;i<=8;i++)sp[i]=0;// sp[i] = sum of i^th powers of q
      t0=cpu();
      for(n=0;n<nd||nd<0;){// Disorder samples
        initweights(weightmode);
        for(i=0;i<nb;i++){// State samples
          init_state();
          for(j=0;j<burnin;j++)tree1gibbs(randint(2),randint(2),randint(N),beta);
          memcpy(sbuf[i],XBa,NBV*sizeof(int));
        }
        for(i=0;i<nb-1;i++)for(j=i+1;j<nb;j++){
          for(k=0,q=0;k<NBV;k++)q+=btab[sbuf[i][k]^sbuf[j][k]];q/=NV;
          for(k=0,x=1;k<=8;k++){sp[k]+=x;x*=q;}
        }
        n++;
        t1=cpu();
        if(t1-t0>5||n==nd){
          t0=t1;
          printf("n=%d\n",n);
          printf("beta %g\n",beta);
          printf("burn-in %d\n",burnin);
          printf("nb %d\n",nb);
          for(j=1;j<=8;j++)printf("%3d %12g\n",j,sp[j]/sp[0]);
          printf("Binder %12g\n",.5*(3-sp[0]*sp[4]/(sp[2]*sp[2])));
          printf("\n");
          fflush(stdout);
        }
      }
    }
    break;
  case 15:// Exchange Monte-Carlo: try to work out optimal temperatures
    {
      int i,j,n,ech,pr=0;
      int nt=ngp>1?genp[1]:500;// Number of temperatures (fine grid for evaluation purposes)
      double tp,del,be0,be1,be[nt],s0[nt],s1[nt],s2[nt];
      int en[nt],ex[nt-1],sbuf[nt][NBV];
      //if((weightmode!=0&&weightmode!=2)||statemap[0]!=-1)fprintf(stderr,"Warning: expect weightmode=0 or 2, statemap[0]=-1\n");
      //initweights(weightmode);
      be0=ngp>2?genp[2]:0.5;be1=ngp>3?genp[3]:5;// low and high beta
      for(i=0;i<nt;i++)be[i]=be0*pow(be1/be0,i/(nt-1.));// Interpolate geometrically for first guess
      printf("nt=%d\n",nt);
      printf("be0=%g\n",be0);
      printf("be1=%g\n",be1);
      printf("%s mode\n",genp[0]?"Single bigvertex":"Tree");
      tp=ngp>4?genp[4]:0.25;
      printf("Going for transition probability %g\n",tp);
      ech=100*(genp[0]?16:1);// Equilibration chunksize
      for(i=0;i<nt;i++){init_state();memcpy(sbuf[i],XBa,NBV*sizeof(int));}
      for(i=0;i<nt-1;i++)ex[i]=0;
      for(i=0;i<nt;i++)s0[i]=s1[i]=s2[i]=0;
      n=-ech;
      while(1){
        for(i=0;i<nt;i++){
          memcpy(XBa,sbuf[i],NBV*sizeof(int));
          switch((int)(genp[0])){
            int d,r,c;
          case 0:
            tree1gibbs(randint(2),randint(2),randint(N),be[i]);
            break;
          case 1:
            for(d=0;d<2;d++)for(r=0;r<N;r++)for(c=0;c<N;c++)bigvertexgibbs(d,c,r,be[i]);
            break;
          }
          v=val();en[i]=v;
          memcpy(sbuf[i],XBa,NBV*sizeof(int));
          if(pr>=2)printf("%5d ",en[i]);
          if(n>=0){s0[i]+=1;s1[i]+=v;s2[i]+=v*v;}
        }
        if(pr>=2)printf("\n");
        for(i=0;i<nt-1;i++){
          if(pr>=3)printf("     ");
          del=(be[i+1]-be[i])*(en[i]-en[i+1]);
          if(del<0||randfloat()<exp(-del)){
            memcpy(XBa,sbuf[i],NBV*sizeof(int));
            memcpy(sbuf[i],sbuf[i+1],NBV*sizeof(int));
            memcpy(sbuf[i+1],XBa,NBV*sizeof(int));
            v=en[i];en[i]=en[i+1];en[i+1]=v;
            if(pr>=3)printf("X");
            if(n>=0)ex[i]++;
          } else if(pr>=3)printf(" ");
        }
        if(pr>=3)printf("\n");
        n++;
        if(n>0&&n%ech==0){
          int i0,nb;
          double p,err,minerr,mu1,sd1,mu[nt],sd[nt],ben[nt];
          for(i=0;i<nt;i++){mu[i]=s1[i]/s0[i];sd[i]=sqrt((s2[i]-s1[i]*s1[i]/s0[i])/(s0[i]-1));}
          printf("\n");
          if(pr>=1){
            printf("  ");for(i=0;i<nt-1;i++)printf(" %5.3f",ex[i]/(double)n);printf("\n");
            //for(i=0;i<nt;i++)printf("%5.3f ",s1[i]/s0[i]);printf("\n");
            //for(i=0;i<nt;i++)printf("%5.3f ",mu[i]);printf(" mu[]\n");
            //for(i=0;i<nt;i++)printf("%5.3f ",sd[i]);printf(" sd[]\n");
            printf("  ");
            for(i=0;i<nt-1;i++){
              mu1=-(be[i+1]-be[i])*(mu[i+1]-mu[i]);
              sd1=(be[i+1]-be[i])*sqrt(sd[i]*sd[i]+sd[i+1]*sd[i+1]);
              if(sd1>1e-6)p=Phi(-mu1/sd1)+exp(sd1*sd1/2-mu1)*Phi(mu1/sd1-sd1); else p=exp(-mu1);
              printf(" %5.3f",p);
            }
            printf("\n");
          }
          j=nt-1;nb=0;
          while(j>0){
            ben[nb++]=be[j];
            minerr=1e9;i0=-1;
            for(i=j-1;i>=0;i--){
              mu1=-(be[j]-be[i])*(mu[j]-mu[i]);
              sd1=(be[j]-be[i])*sqrt(sd[i]*sd[i]+sd[j]*sd[j]);
              // Stable version of Phi(-mu1/sd1)+exp(sd1*sd1/2-mu1)*Phi(mu1/sd1-sd1):
              if(sd1<1e-6)p=exp(-mu1); else p=Phi(-mu1/sd1)+phi(mu1/sd1)/Rphi(mu1/sd1-sd1);
              err=log(p/tp);
              if(fabs(err)<minerr){minerr=fabs(err);i0=i;}
              if(err<0)break;
            }
            j=i0;
          }
          printf("%d steps\n",n);
          printf("p=%.3f choice of be[]:",tp);
          for(i=nb-1;i>=0;i--)printf(" %5.3f",ben[i]);printf("\n");
        }
      }// while(1)
    }
    break;
  case 16:// Exchange Monte-Carlo, example calculating Binder ratio 
    {
      int h,i,j,k,m,n,r,eqb,nd,btab[16];
      double be0[]={0.108,0.137,0.166,0.196,0.226,0.258,0.291,0.326,0.364,0.405,0.451,0.500,0.557,0.624,0.704,0.808,0.944,1.131,1.438,2.000};
      // ^ N=8 -w0 -x-1 p=0.3
      //double be2[]={0.133,0.170,0.209,0.248,0.288,0.329,0.370,0.413,0.458,0.507,0.557,0.612,0.672,0.744,0.830,0.941,1.084,1.268,1.543,1.967,2.821,5.000};
      // ^ N=8 -w2 -x-1 p=0.3
      double be2[]={0.502,0.528,0.556,0.585,0.613,0.644,0.678,0.713,0.754,0.797,0.842,0.894,0.954,1.022,1.101,1.190,1.294,1.419,1.570,1.762,2.015,2.357,2.821,3.505,5.000};
      // ^ N=8 -w2 -x-1 p=0.6
      double be2_15[]={0.530,0.561,0.595,0.630,0.668,0.699,0.732,0.767,0.804,0.842,0.881,0.923,0.967,1.013,1.061,1.111,1.164,1.219,1.276,1.337,1.400,1.467,1.536,1.609,1.685,1.785,1.892,2.004,2.124,2.276,2.440,2.616,2.836,3.111,3.453,3.832,4.352,5.000};
      //double be2_15[]={2};//check
      // ^ N=15 -w7 p=0.4
      int nt;// Number of temperatures
      int nhist,maxhist=500;// Keep samples for the purposes of error-estimating
      double *be;
      if((weightmode!=0&&weightmode!=2)||statemap[0]!=-1)fprintf(stderr,"Warning: expect weightmode=0 or 2, statemap[0]=-1\n");
      if(weightmode==0){be=be0;nt=sizeof(be0)/sizeof(double);} else {be=be2;nt=sizeof(be2)/sizeof(double);}
      if(weightmode==7&&N==15){be=be2_15;nt=sizeof(be2_15)/sizeof(double);}
      double q,x,del,nex,maxerr,ex[nt-1];
      typedef struct {double n,qq[nt][2];} qest;
      qest lsp,sp,hist[maxhist];
      int en[nt],sbuf[2][nt][NBV];
      printf("nt=%d\n",nt);
      printf("beta_low=%g\n",be[0]);
      printf("beta_high=%g\n",be[nt-1]);
      printf("be[] =");for(i=0;i<nt;i++)printf(" %5.3f",be[i]);printf("\n");
      for(i=1,btab[0]=4;i<16;i++)btab[i]=btab[i>>1]-2*(i&1);// (# 0 bits) - (# 1 bits)
      sp.n=0;for(i=0;i<nt;i++)for(j=0;j<2;j++)sp.qq[i][j]=0;
      nhist=0;
      for(i=0,nex=0;i<nt-1;i++)ex[i]=0;// Count of exchanges
      nd=0;// Number of disorder samples
      eqb=(ngp>1?genp[1]:100);
      printf("Equilibration time %d\n",eqb);
      fflush(stdout);

      if(ngp>0&&genp[0]==-1){// optimise
        for(r=0;r<1;r++)for(i=0;i<nt;i++){init_state();memcpy(sbuf[r][i],XBa,NBV*sizeof(int));}
        int vmin=1000000000;
        while(1){// Thermal loop
          r=0;
          for(i=0;i<nt;i++){
            memcpy(XBa,sbuf[r][i],NBV*sizeof(int));
            for(j=0;j<1;j++)tree1gibbs(randint(2),randint(2),randint(N),be[i]);
            v=val();en[i]=v;
            if(v<vmin){vmin=v;printf("min = %d\n",vmin);}
            memcpy(sbuf[r][i],XBa,NBV*sizeof(int));
          }
          for(i=0;i<nt-1;i++){
            del=(be[i+1]-be[i])*(en[i]-en[i+1]);
            if(del<0||randfloat()<exp(-del)){
              memcpy(XBa,sbuf[r][i],NBV*sizeof(int));
              memcpy(sbuf[r][i],sbuf[r][i+1],NBV*sizeof(int));
              memcpy(sbuf[r][i+1],XBa,NBV*sizeof(int));
              v=en[i];en[i]=en[i+1];en[i+1]=v;
              ex[i]++;
            }
          }
          nex++;
          if((int)nex%100==0){
            for(i=0;i<nt;i++)printf("%6.3f ",be[i]);printf("  be[]\n");
            printf("  ");
            for(i=0;i<nt-1;i++)printf(" %6.3f",ex[i]/nex);printf("       exch[]\n");
            fflush(stdout);
          }
        }
      }

      while(1){// Loop over disorders
        initweights(weightmode);// Disorder (J_ij) sample
        for(r=0;r<2;r++)for(i=0;i<nt;i++){init_state();memcpy(sbuf[r][i],XBa,NBV*sizeof(int));}
        lsp.n=0;for(i=0;i<nt;i++)for(k=0;k<2;k++)lsp.qq[i][k]=0;
        n=-eqb;
        while(n<eqb){// Thermal loop
          for(r=0;r<2;r++){// Replica loop
            for(i=0;i<nt;i++){
              memcpy(XBa,sbuf[r][i],NBV*sizeof(int));
              for(j=0;j<1;j++)tree1gibbs(randint(2),randint(2),randint(N),be[i]);
              v=val();en[i]=v;
              memcpy(sbuf[r][i],XBa,NBV*sizeof(int));
            }
            for(i=0;i<nt-1;i++){
              del=(be[i+1]-be[i])*(en[i]-en[i+1]);
              if(del<0||randfloat()<exp(-del)){
                memcpy(XBa,sbuf[r][i],NBV*sizeof(int));
                memcpy(sbuf[r][i],sbuf[r][i+1],NBV*sizeof(int));
                memcpy(sbuf[r][i+1],XBa,NBV*sizeof(int));
                v=en[i];en[i]=en[i+1];en[i+1]=v;
                ex[i]++;
              }
            }
            nex++;
          }//r
          n++;
          if(n>=0){
            for(i=0;i<nt;i++){
              for(k=0,q=0;k<NBV;k++)q+=btab[sbuf[0][i][k]^sbuf[1][i][k]];q/=NV;
              x=q*q;lsp.qq[i][0]+=x;lsp.qq[i][1]+=x*x;
            }
            lsp.n+=1;// Keep this as a variable in case decide to vary the equilibrium point for different disorders
          }
        }
        nd++;
        
        if(nhist<maxhist)hist[nhist++]=lsp; else {
          nhist++;
          i=randint(nhist);
          if(i<maxhist)hist[i]=lsp;
        }
        int nsubsamp,nsamp=200;
        double p0=0.16;// Error percentile p0 to 1-p0, roughly corresponding to +/-1sd of a normal.
        double q0,q2,q4,samp[nt][nsamp];
        double est[nt],err[nt];
        sp.n+=lsp.n;
        for(i=0;i<nt;i++)for(j=0;j<2;j++)sp.qq[i][j]+=lsp.qq[i][j];// lsp.qq[i][j] = <q_i^(2(j+1))>   <.> = thermal sum
        for(i=0;i<nt;i++){
          q0=sp.n;
          q2=sp.qq[i][0];
          q4=sp.qq[i][1];
          est[i]=.5*(3-q0*q4/(q2*q2));
        }
        n=MIN(nhist,maxhist);
        nsubsamp=n;// say
        for(k=0;k<nsamp;k++){
          lsp.n=0;for(i=0;i<nt;i++)for(j=0;j<2;j++)lsp.qq[i][j]=0;
          for(m=0;m<nsubsamp;m++){
            h=randint(n);
            lsp.n+=hist[h].n;
            for(i=0;i<nt;i++)for(j=0;j<2;j++)lsp.qq[i][j]+=hist[h].qq[i][j];
          }
          for(i=0;i<nt;i++){
            q0=lsp.n;
            q2=lsp.qq[i][0];
            q4=lsp.qq[i][1];
            samp[i][k]=.5*(3-q0*q4/(q2*q2));
          }
        }
        for(i=0;i<nt;i++){
          double e0,e1;
          qsort(samp[i],nsamp,sizeof(double),cmpd);
          e0=samp[i][(int)floor(p0*nsamp)];
          e1=samp[i][(int)floor((1-p0)*nsamp)];
          err[i]=MAX(fabs(est[i]-e0),fabs(est[i]-e1))*sqrt(nsubsamp/(double)nhist);
        }
        printf("\n");
        printf("Number of disorders: %d\n",nd);
        for(i=0;i<nt;i++)printf("%6.3f ",be[i]);printf("  be[]\n");
        for(i=0;i<nt;i++)printf("%6.3f ",est[i]);printf("  est[]\n");
        for(i=0,maxerr=0;i<nt;i++){
          printf("%6.3f ",err[i]);
          if(err[i]>maxerr)maxerr=err[i];
        }
        printf("  err[]\n");
        printf("  ");
        for(i=0;i<nt-1;i++)printf(" %6.3f",ex[i]/nex);printf("       exch[]\n");
        fflush(stdout);
        if(nd>=2&&maxerr<1e-3)break;
      }
      
    }
    break;

  case 17:
    // Compare equilibration times of exchange Monte-Carlo by measuring <E>. Use chi^2 method on all temps to determine eqbn.
    // Currently configured to use only a particular disorder (specified by the input seed).
    {
      //if(weightmode!=2||statemap[0]!=-1)fprintf(stderr,"Warning: expect weightmode=2, statemap[0]=-1\n");
      double bew7[][50]={// Weightmode 7, be[]
        {0},
        {0},
        {0},
        {0},
        {0.202,0.325,0.508,0.690,0.887,1.131,1.409,1.782,2.382,3.666,10.000},// 4, 0.25
        //{0.203,0.285,0.411,0.541,0.679,0.820,0.975,1.149,1.344,1.597,1.898,2.327,3.037,4.460,10.000},// 4, 0.4
        {0},
        {0},
        {0},
        {0.267,0.352,0.438,0.520,0.604,0.690,0.782,0.880,0.982,1.096,1.214,1.355,1.524,1.741,2.020,2.382,2.920,3.783,5.511,10.000},// 8, 0.25
        //{0.203,0.265,0.328,0.389,0.452,0.516,0.581,0.643,0.707,0.776,0.846,0.915,0.990,1.071,1.158,1.252,1.355,1.477,1.610,1.768,1.958,2.202,2.516,2.920,3.525,4.495,6.150,10.000}// 8, 0.4
      };
      double be0[2]={genp[3]};
      double *be;
      //double be[]={0.507,0.548,0.590,0.639,0.691,0.750,0.819,0.898,0.999,1.121,1.282,1.506,1.854,2.502,5.000};
      // ^ N=8 -w2 -x-1 p=0.4
      //double be[]={0.108,0.137,0.167,0.199,0.237,0.278,0.325,0.388,0.482,0.635,0.886,1.366,2.369,5.000};
      // ^ N=4 -w11 p=0.4
      //double be[]={0.108,0.122,0.137,0.151,0.167,0.184,0.199,0.215,0.233,0.252,0.273,0.295,0.319,0.345,0.381,0.420,0.463,0.521,0.598,0.700,0.852,1.058,1.366,1.835,2.718,5.000};
      // ^ N=8 -w11 p=0.4
      int nt;// Number of temperatures
      int nd;// Number of disorders sampled
      int pr=2;
      if(genp[3]<=0){
        assert(weightmode==7);
        be=bew7[N]-(int)(genp[3]);
      }else be=be0;
      for(nt=0;be[nt]>0;nt++);
      int en[nt],sbuf[nt][NBV];
      double lem[2][nt];// Total energies for a given disorder
      double em[2][nt][3];// Energy moments over all disorders
      double een[2][nt],ven[2][nt];// Derived energy estimates and std errs
      double x,del,nex,ex[nt-1];
      int eqb,leqb;// Equilibration time
      double eps=ngp>2?genp[2]:0.1;// Target absolute error in energy
      double chi;
      int c,d,e,i,n,p,r;
      double nit,nsol;
      
      printf("Number of temperatures %d\n",nt);
      for(i=0;i<nt;i++)printf("%8.3f ",be[i]);printf("  be[]\n");
      printf("Monte Carlo mode %g (%g)\n",genp[0],genp[1]);
      for(i=0,nex=0;i<nt-1;i++)ex[i]=0;// Count of exchanges
      eqb=1;
      while(1){// Loop over equilibration times
        printf("\nEquilibration times %d and %d\n",eqb,eqb*2);
        for(e=0;e<2;e++)for(i=0;i<nt;i++)em[e][i][0]=em[e][i][1]=em[e][i][2]=0;
        nd=0;
        while(1){// Loop over disorders and runs
          //if(genp[1]==0)initweights(weightmode);// Disorder (J_ij) sample
          for(e=0;e<2;e++){// Loop over two equilibrations being compared
            leqb=eqb<<e;
            nit=nsol=0;d=p=r=0;
          lp0:
            for(i=0;i<nt;i++){init_state();memcpy(sbuf[i],XBa,NBV*sizeof(int));}
            for(i=0;i<nt;i++)lem[e][i]=0;
            for(n=0;n<2*leqb||genp[1]<0;n++){// Thermal loop
              for(i=0;i<nt;i++){
                memcpy(XBa,sbuf[i],NBV*sizeof(int));
                switch((int)(genp[0])){
                case 0:
                  tree1gibbs(randint(2),randint(2),randint(N),be[i]);
                  nit+=1;
                  break;
                case 1:
                  for(d=0;d<2;d++)for(r=0;r<N;r++)for(c=0;c<N;c++)bigvertexgibbs(d,c,r,be[i]);
                  nit+=1;
                  break;
                }
                v=val();en[i]=v;
                memcpy(sbuf[i],XBa,NBV*sizeof(int));
                if(genp[1]<0&&v==genp[1]){nsol+=1;printf("IT %g   SOL %g    IT/SOL %g\n",nit,nsol,nit/nsol);goto lp0;}
              }
              for(i=0;i<nt-1;i++){
                del=(be[i+1]-be[i])*(en[i]-en[i+1]);
                if(del<0||randfloat()<exp(-del)){
                  memcpy(XBa,sbuf[i],NBV*sizeof(int));
                  memcpy(sbuf[i],sbuf[i+1],NBV*sizeof(int));
                  memcpy(sbuf[i+1],XBa,NBV*sizeof(int));
                  v=en[i];en[i]=en[i+1];en[i+1]=v;
                  ex[i]++;
                }
              }
              nex++;
              if(n>=leqb)for(i=0;i<nt;i++)lem[e][i]+=en[i];// Add energies deemed to have been equilibrated
              if(pr>=3&&n>=leqb){for(i=0;i<nt;i++)printf("%8d ",en[i]);printf("  e=%d\n",e);}
            }// Thermal
            for(i=0;i<nt;i++)lem[e][i]/=leqb;
          }// e
          nd++;
          if(pr>=1){
            printf("\n");
            for(i=0;i<nt;i++)printf("%8.3f ",be[i]);printf("  be[]\n");
            printf("   ");
            for(i=0;i<nt-1;i++)printf(" %8.3f",ex[i]/nex);printf("        exch[]\n");
          }
          for(e=0;e<2;e++){
            for(i=0;i<nt;i++){
              x=lem[e][i];if(pr>=2)printf("%8.3f ",x);
              em[e][i][0]+=1;em[e][i][1]+=x;em[e][i][2]+=x*x;
              een[e][i]=em[e][i][1]/em[e][i][0];
              ven[e][i]=(em[e][i][2]-em[e][i][1]*em[e][i][1]/em[e][i][0])/(em[e][i][0]-1)/em[e][i][0];
            }
            if(pr>=2)printf("  sample_%d\n",eqb<<e);
          }
          if(pr>=1){
            for(e=0;e<2;e++){
              for(i=0;i<nt;i++)printf("%8.3f ",een[e][i]);printf("  een[%d][]\n",eqb<<e);
              for(i=0;i<nt;i++)printf("%8.4f ",sqrt(ven[e][i]));printf("  err[%d][]\n",eqb<<e);
            }
          }
          for(i=0;i<nt;i++){
            x=een[0][i]-een[1][i];
            x=x*x/(ven[0][i]+ven[1][i]);if(pr>=1)printf("%8.2f ",x);
          }
          if(pr>=1)printf("  chi^2 (raw)\n");
          chi=0;
          for(i=0;i<nt;i++){
            x=fabs(een[0][i]-een[1][i])-eps;x=MAX(x,0);
            x=x*x/(ven[0][i]+ven[1][i]);if(pr>=1)printf("%8.2f ",x);
            chi+=x;
          }
          if(pr>=1){
            printf("  chi^2 (reduced)\n");
            printf("Error %g cf chi^2_%d, N=%d, nd=%d, genp[]=",chi,nt,N,nd);
            for(i=0;i<ngp;i++)printf("%g%s",genp[i],i<ngp-1?",":"");
            printf(", CPU=%.2fs\n",cpu());
            fflush(stdout);
          }
          if(nd>=15&&chi>=nt+4*sqrt(nt))break;//alter
          if(nd>=(genp[5]==0?1000:genp[5]))goto ok0;
        }// Disorders
        if(genp[4]>0&&eqb>=genp[4]){printf("Giving up. Equilibration time %d deemed insufficient for target error %g at nd=%d, N=%d, method=%g.\n",eqb,eps,nd,N,genp[0]);return 1;}
        eqb*=2;
      }// Eqbn times
    ok0:;
      printf("Equilibration time %d deemed sufficient for target error %g at nd=%d, N=%d, method=%g\n",eqb,eps,nd,N,genp[0]);
    }
    break;

  case 18:
    // Compare equilibration times of exchange Monte-Carlo by measuring <E>. Determine eqbn by assuming top beta is enough to essentially force groundstate.
    // Currently configured to use only a particular disorder (specified by the input seed).
    {
      //if(weightmode!=2||statemap[0]!=-1)fprintf(stderr,"Warning: expect weightmode=2, statemap[0]=-1\n");
      double bew7[][50]={// Weightmode 7, be[]
        {0},
        {0},
        {0},
        {0},
        {0.202,0.325,0.508,0.690,0.887,1.131,1.409,1.782,2.382,3.666,10.000},// 4, 0.25
        //{0.203,0.285,0.411,0.541,0.679,0.820,0.975,1.149,1.344,1.597,1.898,2.327,3.037,4.460,10.000},// 4, 0.4
        {0},
        {0},
        {0},
        {0.267,0.352,0.438,0.520,0.604,0.690,0.782,0.880,0.982,1.096,1.214,1.355,1.524,1.741,2.020,2.382,2.920,3.783,5.511,10.000},// 8, 0.25
        //{0.203,0.265,0.328,0.389,0.452,0.516,0.581,0.643,0.707,0.776,0.846,0.915,0.990,1.071,1.158,1.252,1.355,1.477,1.610,1.768,1.958,2.202,2.516,2.920,3.525,4.495,6.150,10.000}// 8, 0.4
      };
      double be0[2]={genp[3]};
      double *be;
      int nt;// Number of temperatures
      int nd;// Number of disorders sampled
      int pr=genp[1];
      if(genp[3]<=0){
        assert(weightmode==7);
        be=bew7[N]-(int)(genp[3]);
      }else be=be0;
      for(nt=0;be[nt]>0;nt++);
      int vmin,en[nt],sbuf[nt][NBV];
      double em[nt][3];// Energy moments over all disorders
      double een[nt],ven[nt];// Derived energy estimates and std errs
      double x,y,del,nex,ex[nt-1];
      int eqb;// Current upper bound on equilibration time
      double eps=ngp>2?genp[2]:0.1;// Target absolute error in energy
      int c,d,e,i,n,p,r;
      double mu,va,se,nit,nsol;
      
      printf("Number of temperatures %d\n",nt);
      for(i=0;i<nt;i++)printf("%8.3f ",be[i]);printf("  be[]\n");
      printf("Monte Carlo mode %g\n",genp[0]);
      for(i=0,nex=0;i<nt-1;i++)ex[i]=0;// Count of exchanges
      eqb=1;vmin=1000000000;
      while(1){// Loop over equilibration lengths
        double ten[2*eqb],sten0[eqb+1],sten1[eqb+1],sten2[eqb+1];
        double lem[nt];
        printf("\nEquilibration length %d\n",eqb);
        for(i=0;i<eqb+1;i++)sten0[i]=sten1[i]=sten2[i]=0;
        for(i=0;i<nt;i++)em[i][0]=em[i][1]=em[i][2]=0;
        nd=0;
        while(1){// Loop over runs
          nit=nsol=0;d=p=r=0;
          for(i=0;i<nt;i++){init_state();memcpy(sbuf[i],XBa,NBV*sizeof(int));}
          for(i=0;i<nt;i++)lem[i]=0;
          for(n=0;n<2*eqb;n++){// Thermal loop
            for(i=0;i<nt;i++){
              memcpy(XBa,sbuf[i],NBV*sizeof(int));
              switch((int)(genp[0])){
              case 0:
                tree1gibbs(randint(2),randint(2),randint(N),be[i]);
                nit+=1;
                break;
              case 1:
                for(d=0;d<2;d++)for(r=0;r<N;r++)for(c=0;c<N;c++)bigvertexgibbs(d,c,r,be[i]);
                nit+=1;
                break;
              }
              v=val();en[i]=v;if(v<vmin)vmin=v;
              memcpy(sbuf[i],XBa,NBV*sizeof(int));
            }
            for(i=0;i<nt-1;i++){
              del=(be[i+1]-be[i])*(en[i]-en[i+1]);
              if(del<0||randfloat()<exp(-del)){
                memcpy(XBa,sbuf[i],NBV*sizeof(int));
                memcpy(sbuf[i],sbuf[i+1],NBV*sizeof(int));
                memcpy(sbuf[i+1],XBa,NBV*sizeof(int));
                v=en[i];en[i]=en[i+1];en[i+1]=v;
                ex[i]++;
              }
            }
            nex++;
            ten[n]=en[nt-1];// Store top energy at each sample step (for equilibration detection)
            if(n>=eqb)for(i=0;i<nt;i++)lem[i]+=en[i];// Store total energies at each temperature (for interest)
            if(pr>=3&&n>=eqb){for(i=0;i<nt;i++)printf("%8d ",en[i]);printf("\n");}
          }// Thermal
          for(i=0;i<nt;i++)lem[i]/=eqb;
          nd++;
          if(pr>=1){
            printf("\n");
            for(i=0;i<nt;i++)printf("%8.3f ",be[i]);printf("  be[]\n");
            printf("   ");
            for(i=0;i<nt-1;i++)printf(" %8.3f",ex[i]/nex);printf("        exch[]\n");
          }
          for(i=0;i<nt;i++){
            x=lem[i];if(pr>=2)printf("%8.3f ",x);
            em[i][0]+=1;em[i][1]+=x;em[i][2]+=x*x;
            een[i]=em[i][1]/em[i][0];
            ven[i]=(em[i][2]-em[i][1]*em[i][1]/em[i][0])/(em[i][0]-1)/em[i][0];
          }
          if(pr>=2)printf("  sample_%d\n",eqb);
          if(pr>=1){
            for(i=0;i<nt;i++)printf("%8.3f ",een[i]);printf("  een[%d][]\n",eqb);
            for(i=0;i<nt;i++)printf("%8.4f ",sqrt(ven[i]));printf("  err[%d][]\n",eqb);
          }
          for(n=1,x=0;n<=eqb;n++){
            x+=ten[2*n-2]+ten[2*n-1]-ten[n-1];
            // x = sum of ten[n],...,ten[2n-1] = the top-energy terms that would be used at eqb=n
            y=x/n;
            sten0[n]+=1;sten1[n]+=y;sten2[n]+=y*y;
          }
          mu=sten1[eqb]/sten0[eqb];
          va=(sten2[eqb]-sten1[eqb]*sten1[eqb]/sten0[eqb])/(sten0[eqb]-1);
          se=sqrt(va/sten0[eqb]);
          assert(mu>=vmin);
          if(pr>=1){
            printf("Error %.3g (std err %.3g), vmin=%d, N=%d, nd=%d, genp[]=",mu-vmin,se,vmin,N,nd);
            for(i=0;i<ngp;i++)printf("%g%s",genp[i],i<ngp-1?",":"");
            printf(", CPU=%.2fs\n",cpu());
            fflush(stdout);
          }
          // N(mu,se^2) is actually a very poor approximation to the posterior distribution of the energy of the top beta
          if(nd>=15&&mu-vmin>eps+2*se)break;
          if(nd>=50){//                  50, eps/4 are rough-and-ready parameters
            if(mu-vmin<eps&&se<eps/4){// 
              if(pr>=3)for(n=1;n<=eqb;n++)printf("%6d %12.6f %12g\n",n,sten1[n]/sten0[n],sten1[n]/sten0[n]-vmin);
              for(n=1,e=1;n<=eqb;n++)if(sten1[n]/sten0[n]-vmin>eps)e++;
              printf("Equilibration time %d deemed sufficient for target error %g at nd=%d, eqb=%d, N=%d, method=%g, workproduct=%d\n",
                     e,eps,nd,eqb,N,genp[0],e*nt);
              goto ok1;
            }else break;
          }
        }// Runs
        if(genp[4]>0&&eqb>=genp[4]){printf("Giving up. Equilibration time %d deemed insufficient for target error %g at nd=%d, N=%d, method=%g.\n",eqb,eps,nd,N,genp[0]);return 1;}
        eqb*=2;
      }// Eqbn times
    ok1:;
    }
    break;
  }// switch(mode)
  if(outstatefile)writestate(outstatefile);
  return 0;
}
