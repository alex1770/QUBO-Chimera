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

// QUBO solver
// Solves QUBO problem:
// Minimise sum_{i,j} Q_ij x_i x_j over choices of x_i
// i,j corresponds to an edge from i to j in the "Chimera" graph C_N.
// The sum is taken over both directions (i,j and j,i) and includes the diagonal terms
// (configurable using option -w).
// x_i can take the values statemap[0] and statemap[1] (default 0,1).
// This includes the case described in section 3.2 of http://www.cs.amherst.edu/ccm/cf14-mcgeoch.pdf
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

#define MAX(x,y) ((x)>(y)?(x):(y))
#define MAXVAL 10000 // Assume for convenience that all values (energies) are <= this in absolute value
                     // (Only used for stats)

// Isolate random number generator in case we need to replace it with something better
void initrand(int seed){srandom(seed);}
int randbit(void){return (random()>>16)&1;}
int randsign(void){return randbit()*2-1;}
int randnib(void){return (random()>>16)&15;}
int randint(int n){return random()%n;}

typedef long long int int64;// gcc's 64-bit type
double*work; // work[wid]=estimate of relative running time of stablestripexhaust at width wid
             // (in arbitrary work units)

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
    case 7:if((d<4&&deco(p)==0)||d==5){r=randsign();Q[p][i][d]=2*r;Q[p][i][6]-=r;Q[q][j][6]-=r;}
      break;
    case 8:if((d<4&&deco(p)==0)||d==5){r=randint(201)-100;Q[p][i][d]=2*r;Q[p][i][6]-=r;Q[q][j][6]-=r;}
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
        if(x1==x0&&abs(y1-y0)==1&&o0==1&&o1==1){e0=4+(y1-y0+1)/2;}else assert(0);
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

void initwork(){
  int wid;
  work=(double*)malloc((N+1)*sizeof(double));
  // work[wid] counts approx number of QBI references in a stable exhaust of width wid
  for(wid=1;wid<=N;wid++)work[wid]=N*(1LL<<4*(wid+1))*(wid+32/15.)*(N-wid+1)*3;
}

void shuf(int*a,int n){
  int i,j,t;
  for(i=0;i<n-1;i++){
    j=i+randint(n-i);t=a[i];a[i]=a[j];a[j]=t;
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
  unsigned char (*hc)[wid][M]=0,// Comb history: hc[r][x][b] = opt value of (c0+x,r,0) given (c0+y,r,I(y<=x))=b   (y=0,...,wid-1)
    (*hs)[wid][M]=0;           // Strut history: hs[r][x][b] = opt value of (c0+x,r,1) given (c0+y,r+I(y<=x),1)=b   (y=0,...,wid-1)
  vv=(short*)malloc(M*sizeof(short));
  if(upd){
    hc=(unsigned char (*)[wid][M])malloc(N*wid*M);
    hs=(unsigned char (*)[wid][M])malloc(N*wid*M);
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
  int v,s0,s1,v0,vmin,tv[16];
  for(s1=0;s1<16;s1++)tv[s1]=QB(x,y,1,1,s1,XB(x,y-1,1))+QB(x,y,1,2,s1,XB(x,y+1,1));
  vmin=1000000000;
  for(s0=0;s0<16;s0++){
    v0=QB(x,y,0,1,s0,XB(x-1,y,0))+QB(x,y,0,2,s0,XB(x+1,y,0));
    for(s1=0;s1<16;s1++){
      v=QB(x,y,0,0,s0,s1)+v0+tv[s1];
      if(v<vmin){vmin=v;XB(x,y,0)=s0;XB(x,y,1)=s1;}
    }
  }
  return vmin;
}

int stablek44exhaust(int cv){// Repeated k44 exhausts until no more improvement likely
  int i,r,v,x,y,ord[N*N];
  r=0;
  while(1){
    for(i=0;i<N*N;i++)ord[i]=i;
    shuf(ord,N*N);
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
      // (2nc isn't actually enough to ensure there is no more improvement possible)
    }
  }
}

void init_state(void){// Initialise state randomly
  int x,y,o;
  for(x=0;x<N;x++)for(y=0;y<N;y++)for(o=0;o<2;o++)XB(x,y,o)=randnib();
}

void pertstate(int sq){
  int o,x,y,x0,y0,it;
  for(it=0;it<1;it++){
    x0=randint(N-sq+1);y0=randint(N-sq+1);
    for(x=x0;x<x0+sq;x++)for(y=y0;y<y0+sq;y++)for(o=0;o<2;o++)XB(x,y,o)=randnib();
    if(1){
      if(x0>0)lineexhaust(0,x0-1,1);
      if(x0<N-sq)lineexhaust(0,x0+sq,1);
      if(y0>0)lineexhaust(1,y0-1,1);
      if(y0<N-sq)lineexhaust(1,y0+sq,1);
    }else{
      //for(x=x0;x<x0+sq;x++)lineexhaust(0,x,1);
      //for(y=y0;y<y0+sq;y++)lineexhaust(1,y,1);
    }
  }
}

int opt1(double mint,double maxt,int pr,int tns,double *tts,int strat,int gtr){
  //
  // Heuristic optimisation, writing back best value found. Can be used to find TTS, the
  // expected time to find an optimum solution, using the strategy labelled by 'strat'.
  // 'strat' is assumed to be a fixed strategy running forever which does not know what
  // the optimum value is. I.e., it is not allowed to make decisions as to how to search
  // based on outside knowledge of the optimum value. The aim is to get an accurate
  // estimate of the expected time for 'strat' to find its first optimum state.
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
  // There are simple strategies (strat=0,1) which are Markovian in nature, in that they
  // keep starting from a random configuration (of spins), and have no other state, so
  // they generate independent samples of TTS in their normal operation.
  //
  // Other strategies (which are the subject of the rest of this comment) may have "state"
  // or may not restart from a random configuration, for example they might use previously
  // found configurations to help make a new one.  In these cases, you need to stop the
  // strategy with it hits an optimum and then restart it cleanly, clearing all state.
  // Only that way can you be sure that you are averaging independent runs when taking the
  // average TTS.
  //
  // Of course, since the strategy doesn't actually know the optimum value (unless it has
  // been supplied it via gtr), and is not allowed to use it to change its behaviour, it
  // has to restart itself every time it hits a "presumed optimum", i.e., a (equal-)lowest
  // value found so far. This presumed optimum can be carried over from previous runs, so
  // long as it is not changing any decision that the strategy makes in any independent
  // run. I.e., you have to imagine the run carrying on forever, but the presumed optimum
  // just dictates when to take the time reading (at the point the run finds an equally
  // good value).
  //
  // If the presumed optimum is bettered, so making a new presumed optimum, then all
  // previous statistics have to be discarded, including the current run which found the
  // new presumed optimum. This is because the new presumed optimum was not found under
  // "infinity run" conditions: the early stopping at the old presumed optimum might have
  // biased it. Thus you actually need to find n+1 optima to generate n samples.
  //
  // S0: Randomise configuration; stablek44exhaust; repeat
  // S1: Randomise configuration; stablelineexhaust; repeat
  // S2: Randomise configuration; stablelineexhaust; if #lineexhausts since last reset
  //     point > (some constant) then do a stable-width2-exhaust from the best
  //     post-lineexhaust config since the last reset point. If just did a width2-exhaust,
  //     then do a reset, i.e., clear state. This reset is a choice: not necessary for
  //     unbiasedness, but may help not getting stuck.

  int v,bv,lbv,cv,ns,sr,nas,new,last,Xbest[NBV],Xlbest[NBV];
  int64 nn,stats[2*MAXVAL+1];
  double ff,t0,t1,t2,tt,w1,now;
  if(pr)printf("Min time to run: %gs\nMax time to run: %gs\nGroundtruth: %d\n",mint,maxt,gtr);
  bv=lbv=1000000000;nn=0;
  t0=cpu();// Initial time
  t1=0;// Elapsed time threshold for printing update
  // "presumed solution" means "minimum value found so far"
  ns=0;// Number of presumed solutions
  nas=0;// Number of actual solutions (of value gtr)
  t2=t0;// t2 = Time of last clean start
  memset(stats,0,sizeof(stats));
  w1=0;// Work done so far at width 1 since last width 2 exhaust or last presumed solution
  ff=N*N/40.;
  if(strat%10==2&&pr)printf("w1/w2 = %g\n",work[2]/(ff*work[1]));
  sr=(strat<2);// Simple-restarting strategy
  do{
    if(strat<10)init_state(); else pertstate(2);
    cv=val();
    switch(strat%10){
    case 0:
      cv=stablek44exhaust(cv);// Simple "local" strategy
      break;
    case 1:
      cv=stablestripexhaust(cv,1);
      break;
    case 2:
      cv=stablestripexhaust(cv,1);w1+=work[1];
      if(cv<=lbv){lbv=cv;memcpy(Xlbest,XBa,NBV*sizeof(int));}
      if(N>1&&ff*w1>=work[2]){
        if(pr>=2){printf("Width 2 exhaust");fflush(stdout);}
        memcpy(XBa,Xlbest,NBV*sizeof(int));
        cv=stablestripexhaust(lbv,2);
        if(pr>=2)printf("\n");
        w1=0;lbv=1000000000;// This has the effect of clearing the state, since with lbv=infinity, Xlbest will be overwritten before it is read
      }
      break;
    }
    if(abs(cv)<=MAXVAL)stats[MAXVAL+cv]++;
    if((pr>=3&&cv<=bv)||pr==4){printf("\n");prstate(stdout,1,Xbest);printf("cv %d    bv %d\n",cv,bv);}
    nn++;
    now=cpu();
    new=(cv<bv);
    if(new){bv=cv;ns=0;memcpy(Xbest,XBa,NBV*sizeof(int));}
    if(cv==bv){
      if(sr||!new)ns++; else t2=now;
      init_state();
      w1=0;lbv=1000000000;
    }
    if(cv==gtr){nas++;init_state();}
    if(gtr==1000000000)last=(ns>=tns); else last=(nas>=tns);
    tt=now-t0;
    last=(tt>=mint&&last)||tt>=maxt;
    if(new||tt>=t1||last){
      t1=MAX(tt*1.1,tt+5);
      if(pr>=1){
        printf("%12lld %10d %10d %8.2f %8.2f\n",nn,bv,ns,now-t2,tt);
        if(pr>=2&&bv>=-MAXVAL)for(v=0;v>=bv;v--)if(stats[MAXVAL+v])printf("%6d %12lld\n",v,stats[MAXVAL+v]);
        fflush(stdout);
      }
    }
  }while(!last);
  if(tts)*tts=(now-t2)/ns;
  memcpy(XBa,Xbest,NBV*sizeof(int));
  return bv;
}

int cmpint(const void*p,const void*q){return *(int*)p-*(int*)q;}

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
  unsigned char (*ok0)[16][16];
  ok0=(unsigned char(*)[16][16])malloc(65536*16*16);assert(ok0);
  for(v=0;v<NBV;v++)for(s=0;s<16;s++)ok[v][s]=1;
  tt0=1000000000;
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
  // Sort ok2[] to facilitate exhaust2()
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
  int a,c,r,v,x,A,np,ps0,s0,s0i,s1,s1i,s0i1,s1i1,mul0,mul1;
  int64 b,ns,maxs,size,sizer;
  double tns,ctns,cost,mincost;
  short*v0,*v1;
  int XBa0[NBV],QBa0[NBV][3][16][16],pre[65536][4],ok0[NBV][16],nok0[NBV],ok20[N*N][256],nok20[N*N];
  getrestrictedsets();
  memcpy(XBa0,XBa,sizeof(XBa0));memcpy(QBa0,QBa,sizeof(QBa0));
  memcpy(ok0,ok,sizeof(ok0));memcpy(nok0,nok,sizeof(nok0));
  memcpy(ok20,ok2,sizeof(ok20));memcpy(nok20,nok2,sizeof(nok20));
  mincost=1e100;A=-1;size=1LL<<60;
  if(deb>=1)printf("                  Memory/GiB   Time(a.u.)  Memory*Time\n");
  for(a=0;a<8;a++){// Loop over automorphisms of C_N to choose best representation to exhaust
    applyam(a,XBa0,QBa0,ok0,nok0,ok20,nok20);
    maxs=0;tns=0;
    for(r=0;r<N;r++){
      for(c=0;c<N;c++){
        ns=1;
        for(x=0;x<c;x++)ns*=nok[encp(x,r+1,1)];
        ns*=MAX(nok2[enc2(c,r)]*nok[encp(c+1,r,1)],nok[enc(c,r,1)]*nok2[enc2p(c+1,r)]);
        for(x=c+2;x<N;x++)ns*=nok[enc(x,r,1)];
        tns+=ns;
        if(ns>maxs)maxs=ns;
      }
    }
    cost=tns*maxs;// Using cost = time * memory
    if(deb>=1){double z=(double)maxs*2.*sizeof(short)/(1<<30);printf("Automorphism %d: %12g %12g %12g\n",a,z,tns,z*tns);}
    if(cost<mincost){mincost=cost;size=maxs;ctns=tns;A=a;}
  }
  applyam(A,XBa0,QBa0,ok0,nok0,ok20,nok20);
  if(deb>=1)printf("Choosing automorphism %d\n",A);
  printf("Size %.1fGiB\n",size*2.*sizeof(short)/(1<<30));
  printf("Time units %g\n",ctns);
  fflush(stdout);//exit(0);
  v0=(short*)malloc(size*sizeof(short));
  v1=(short*)malloc(size*sizeof(short));
  if(!(v0&&v1)){fprintf(stderr,"Couldn't allocate %gGiB in fullexhaust()\n",size*2.*sizeof(short)/(1<<30));return 1;}
  memset(v0,0,size*sizeof(short));
  for(r=0;r<N;r++){
    for(c=0;c<N;c++){
      // Add c,r,0 to interior
      // In multibase low-high order:
      // Old boundary (c,r)*   \OR               (1 full)  if c>0, OR
      //              (c,r,1)  /                 (1)       if c=0
      //              (x,r,1)    x=c+1,...,N-1,  (N-1-c)
      //              (x,r+1,1)  x=0,1,...,c-1   (c)
      // New boundary (c,r,1)                    (1)
      //              (c+1,r)*                   (1 full)  \ if c<N-1
      //              (x,r,1)     x=c+2,...,N-1, (N-2-c)   /
      //              (x,r+1,1)   x=0,...,c-1    (c)
      sizer=1;
      for(x=c+2;x<N;x++)sizer*=nok[enc(x,r,1)];
      for(x=0;x<c;x++)sizer*=nok[encp(x,r+1,1)];
      np=0;
      for(s1i=0;s1i<nok2[enc2p(c+1,r)];s1i++){
        s1=ok2[enc2p(c+1,r)][s1i];
        s1i1=okinv(c+1,r,1,s1>>4);
        ps0=1000;
        for(s0i=0;s0i<nok2[enc2(c,r)];s0i++){
          s0=ok2[enc2(c,r)][s0i];
          s0i1=okinv(c,r,1,s0>>4);
          if(c==0)pre[np][0]=s0i1+nok[enc(c,r,1)]*s1i1; else pre[np][0]=s0i+nok2[enc2(c,r)]*s1i1;
          pre[np][1]=QB(c,r,0,0,s0&15,s0>>4)+QB(c,r,0,2,s0&15,s1&15);
          pre[np][2]=0;
          if(s0i>0){assert(np>0);pre[np-1][2]=((s0>>4)>(ps0>>4));}
          pre[np][3]=s0i1+s1i*nok[enc(c,r,1)];
          ps0=s0;
          np++;
          //v=v0[s0i,s1i1,b]+QB(c,r,0,0,s0&15,s0>>4)+QB(c,r,0,2,s0&15,s1>>4);
          //if(v<vmin)vmin=v;
          //if(s0i0==<last one>){v1[s0i1,s1i,b]=vmin;vmin=32767;}
        }
        assert(np>0);pre[np-1][2]=1;
      }
      mul0=(c==0?nok[enc(c,r,1)]:nok2[enc2(c,r)])*nok[encp(c+1,r,1)];
      mul1=nok[enc(c,r,1)]*nok2[enc2p(c+1,r)];
      if(deb>=2)printf("%d %d 0 : %12lld -> %12lld\n",r,c,sizer*mul0,sizer*mul1);
      assert(sizer*MAX(mul0,mul1)<=size);
#pragma omp parallel for
      for(b=0;b<sizer;b++){// b=state of rest of new boundary (>=c+2,r,1), (<c,r+1,1)
        int p,v,vmin;
        vmin=32767;
        for(p=0;p<np;p++){
          v=v0[pre[p][0]+mul0*b]+pre[p][1];
          if(v<vmin)vmin=v;
          if(pre[p][2]){v1[pre[p][3]+mul1*b]=vmin;vmin=32767;}
        }
      }
      // Add c,r,1 to interior
      // In multibase low-high order:
      // Old boundary (c,r,1)                    (1)
      //              (c+1,r)*                   (1 full)   \ if c<N-1
      //              (x,r,1)     x=c+2,...,N-1, (N-2-c)    /
      //              (x,r+1,1)   x=0,...,c-1    (c)
      // New boundary (c+1,r)*                   (1 full)   \ if c<N-1
      //              (x,r,1)    x=c+2,...,N-1,  (N-2-c)    /
      //              (x,r+1,1)  x=0,1,...,c     (c+1)
      // Boundary loses (c,r,1) and gains (c,r+1,1)
      // New edge (c,r,1) to (c,r+1,1)
      sizer=nok2[enc2p(c+1,r)];
      for(x=c+2;x<N;x++)sizer*=nok[enc(x,r,1)];
      for(x=0;x<c;x++)sizer*=nok[encp(x,r+1,1)];
      mul0=nok[enc(c,r,1)];
      mul1=nok[encp(c,r+1,1)];
      if(deb>=2)printf("%d %d 1 : %12lld -> %12lld\n",r,c,sizer*mul0,sizer*mul1);
      assert(sizer*MAX(mul0,mul1)<=size);
      np=0;
      for(s1i1=0;s1i1<nok[encp(c,r+1,1)];s1i1++){
        for(s0i1=0;s0i1<nok[enc(c,r,1)];s0i1++){
          pre[np][0]=s0i1;
          pre[np][1]=QB(c,r,1,2,ok[enc(c,r,1)][s0i1],ok[encp(c,r+1,1)][s1i1]);
          pre[np][2]=(s0i1==nok[enc(c,r,1)]-1);
          pre[np][3]=s1i1;
          np++;
          //v=v1[s0i1+nok[enc(c,r,1)]*b]+QB(c,r,1,2,s01,s11);
          //if(v<vmin)vmin=v;
          //if(s0i1==nok[enc(c,r,1)]-1){v0[b+sizer*s1i1]=vmin;vmin=32767;}
        }
      }
#pragma omp parallel for
      for(b=0;b<sizer;b++){
        int p,v,vmin;
        vmin=32767;
        for(p=0;p<np;p++){
          v=v1[pre[p][0]+mul0*b]+pre[p][1];
          if(v<vmin)vmin=v;
          if(pre[p][2]){v0[b+sizer*pre[p][3]]=vmin;vmin=32767;}
        }
      }
    }//c
  }//r
  v=v0[0];
  free(v1);free(v0);
  applyam(0,XBa0,QBa0,ok0,nok0,ok20,nok20);
  return v;
}

int fullexhaust2(){
  // Uses restricted sets to cut down possibilities
  // and full automorphism group to choose best orientation
  int a,c,r,s,v,x,bc,bc0,A,s0,mul0,mul1,
    XBa0[NBV],QBa0[NBV][3][16][16],ok0[NBV][16],nok0[NBV],ok20[N*N][256],nok20[N*N],
    pre[4096][4],pre2[16][16];
  int64 b,br,bm,nc,ns,nc0,tnc,maxc,maxs,maxt,size0,size1,bp[N+1];
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
  if(!(v0&&v1)){fprintf(stderr,"Couldn't allocate %gGiB in fullexhaust()\n",(double)(size0+size1)*sizeof(short)/(1<<30));return 1;}
  t0+=cpu();

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
    // vc2[s2,b3] = min_{s3} Q(s3,b3)+Q(s3,s2)
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
#pragma omp parallel for
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
    assert(bp[N]<=size0);
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
#pragma omp parallel for
      for(br=0;br<bm;br++){// br = state of non-c columns
        int v,vmin,bc0,s0;
        for(bc0=0;bc0<mul1;bc0++){// bc = state of (c,r+1,1)
          vmin=1000000000;
          for(s0=0;s0<mul0;s0++){// s = state of (c,r,1)
            v=vold[s0+mul0*br]+pre2[bc0][s0];
            if(v<vmin)vmin=v;
          }
          vnew[br+bm*bc0]=vmin;
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
  return v;
}

int main(int ac,char**av){
  int opt,wn,mode,strat,weightmode,gtr;
  double mint,maxt;
  char *inprobfile,*outprobfile,*outstatefile;

  wn=-1;inprobfile=outprobfile=outstatefile=0;seed=time(0);mint=10;maxt=1e10;statemap[0]=0;statemap[1]=1;
  weightmode=5;mode=0;N=8;strat=2;deb=1;gtr=1000000000;
  while((opt=getopt(ac,av,"g:m:n:N:o:O:s:S:t:T:v:w:x:"))!=-1){
    switch(opt){
    case 'g': gtr=atoi(optarg);break;
    case 'm': mode=atoi(optarg);break;
    case 'n': wn=atoi(optarg);break;
    case 'N': N=atoi(optarg);break;
    case 'o': outprobfile=strdup(optarg);break;
    case 'O': outstatefile=strdup(optarg);break;
    case 's': seed=atoi(optarg);break;
    case 'S': strat=atoi(optarg);break;
    case 't': mint=atof(optarg);break;
    case 'T': maxt=atof(optarg);break;
    case 'v': deb=atoi(optarg);break;
    case 'w': weightmode=atoi(optarg);break;
    case 'x': statemap[0]=atoi(optarg);break;
    default:
      fprintf(stderr,"Usage: %s [OPTIONS] [inputproblemfile]\n",av[0]);
      fprintf(stderr,"       -g   ground truth value (for -m1 mode)\n");
      fprintf(stderr,"       -m   mode of operation:\n");
      fprintf(stderr,"            0   Try to find minimum value by heuristic search (default)\n");
      fprintf(stderr,"            1   Try to find rate of solution generation by repeated heuristic search\n");
      fprintf(stderr,"            2   Try to find expected minimum value by heuristic search\n");
      fprintf(stderr,"            3   Full exhaust (proving), basic method\n");
      fprintf(stderr,"            4   Consistency checks\n");
      fprintf(stderr,"            5   Full exhaust (proving), better method\n");
      fprintf(stderr,"       -n   num working nodes\n");
      fprintf(stderr,"       -N   size of Chimera graph\n");
      fprintf(stderr,"       -o   output problem (weight) file\n");
      fprintf(stderr,"       -O   output state file\n");
      fprintf(stderr,"       -s   seed\n");
      fprintf(stderr,"       -S   search strategy for heuristic search (0,1,2)\n");
      fprintf(stderr,"            0   Exhaust K44s repeatedly\n");
      fprintf(stderr,"            1   Exhaust lines repeatedly (default)\n");
      fprintf(stderr,"            2   Exhaust lines and line-pairs repeatedly\n");
      fprintf(stderr,"       -t   min run time for some modes\n");
      fprintf(stderr,"       -T   max run time for some modes\n");
      fprintf(stderr,"       -v   0,1,2,... verbosity level\n");
      fprintf(stderr,"       -w   weight creation convention\n");
      fprintf(stderr,"            0   All of Q_ij independently +/-1\n");
      fprintf(stderr,"            1   As 0, but diagonal not allowed\n");
      fprintf(stderr,"            2   Upper triangular\n");
      fprintf(stderr,"            3   All of Q_ij allowed, but constrained symmetric\n");
      fprintf(stderr,"            4   Constrained symmetric, diagonal not allowed\n");
      fprintf(stderr,"            5   Start with J_ij (i<j) and h_i IID {-1,1} and transform\n");
      fprintf(stderr,"                back to Q (ignoring constant term) (default)\n");
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
  initwork();
  initrand(seed);
  initgraph(wn);
  if(inprobfile){
    wn=readweights(inprobfile);// This overrides current setting of wn
  }else{
    initweights(weightmode);printf("Initialising random weight matrix with %d working node%s\n",wn,wn==1?"":"s");
  }
  printf("%d working node%s out of %d\n",wn,wn==1?"":"s",NV);
  printf("States are %d,%d\n",statemap[0],statemap[1]);
  printf("Weight-choosing mode: %d\n",weightmode);
  if(outprobfile){writeweights(outprobfile);printf("Wrote weight matrix to file \"%s\"\n",outprobfile);}
  double t0;
  switch(mode){
  case 0:// Find single minimum value
    opt1(mint,maxt,deb,1,0,strat,gtr);
    break;
  case 1:;// Find rate of solution generation
    int v;
    double tts;
    v=opt1(gtr==1000000000?0.5:mint,maxt,1,gtr==1000000000?500:10,&tts,strat,gtr);
    printf("Time to solution %gs, assuming true minimum is %d\n",tts,v);
    break;
  case 2:;// Find average minimum value
    double s0,s1,s2,va;
    s0=s1=s2=0;
    while(1){
      initweights(weightmode);
      v=opt1(0,maxt,0,500,0,strat,gtr);
      s0+=1;s1+=v;s2+=v*v;va=(s2-s1*s1/s0)/(s0-1);
      printf("%12g %12g %12g %12g\n",s0,s1/s0,sqrt(va),sqrt(va/s0));
    }
    break;
  case 3:// Prove using subset method
    printf("Restricted set exhaust\n");
    t0=cpu();
    v=fullexhaust();
    printf("Optimum %d found in %gs\n",v,cpu()-t0);
    break;
  case 4:;// Checks
    opt1(mint,maxt,1,1,0,strat,gtr);
    printf("Full exhaust %d\n",stripexhaust(0,0,N,0));
    int o,c0,c1;
    for(o=0;o<2;o++)for(c0=0;c0<N;c0++)for(c1=c0+1;c1<=N;c1++){
      v=stripexhaust(o,c0,c1,0)+stripval(o,0,c0)+stripval(o,c1,N);
      printf("Strip %2d %2d %2d   %6d\n",o,c0,c1,v);
    }
    break;
  case 5:// Prove using subset method v2
    printf("Restricted set exhaust\n");
    t0=cpu();
    v=fullexhaust2();
    printf("Optimum %d found in %gs\n",v,cpu()-t0);
    break;
  case 6:
    readstate("state");printf("state = %d\n",val());break;
  case 8:
    {
      int n,o,wid,v0,upd;
      double t0;
      opt1(mint,maxt,1,1,0,strat,gtr);
      printf("val=%d\n",val());
      upd=1;
      wid=2;
      for(o=0;o<2;o++)for(c0=0;c0<N-wid+1;c0++){
        c1=c0+wid;
        for(n=0,t0=cpu();n<(2000000>>(wid*4))+1;n++)v0=stripexhaust(o,c0,c1,upd);
        v0+=stripval(o,0,c0)+stripval(o,c1,N);
        if(upd)assert(v0==val());
        printf("Strip %d %2d %2d   %6d   %gs\n",o,c0,c1,v0,(cpu()-t0)/n);
        fflush(stdout);
      }
    }
    break;
  }
  if(outstatefile)writestate(outstatefile);
  return 0;
}
