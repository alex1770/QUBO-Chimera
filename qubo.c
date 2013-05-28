#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <getopt.h>
#include <assert.h>

// QUBO solver
// Solves random instances of QUBO problem as described in section 3.2 of
// http://www.cs.amherst.edu/ccm/cf14-mcgeoch.pdf
// 
// Chimera graph, C_N:
// Vertices are (x,y,o,i)  0<=x,y<N, 0<=o<2, 0<=i<4
// Edge from (x,y,o,i) to (y',x',o',i') if
// (x,y)=(x',y'), o!=o', OR
// |x-x'|=1, y=y', o=o'=0, i=i', OR
// |y-y'|=1, x=x', o=o'=1, i=i'
// 
// x,y are the horizontal,vertical co-ords of the K4,4
// o=0..1 is the "orientation" (0=horizontally connected, 1=vertically connected)
// i=0..3 is the index within the "semi-K4,4"="bigvertex"
// There is an involution given by {x<->y o<->1-o}

#define N 8
#define NV (8*N*N)        // Num vertices
#define NE (8*N*(3*N-1))  // Num edges (not used)
#define NBV (2*N*N)       // Num "big" vertices (semi-K4,4s)
#define NBE (N*(3*N-2))   // Num "big" edges (not used)
#define enc(x,y,o) ((o)+((N*(x)+(y))<<1))
#define decx(p) (((p)>>1)/N)
#define decy(p) (((p)>>1)%N)
#define deco(p) ((p)&1)
// encI is the same as enc but incorporates the involution x<->y, o<->1-o
#define encI(inv,x,y,o) (((inv)^(o))+((N*(x)+(y)+(inv)*(N-1)*((y)-(x)))<<1))
//#define encI(inv,x,y,o) ((inv)?enc(y,x,1-(o)):enc(x,y,o))
//#define encI(inv,x,y,o) (enc(x,y,o)+(inv)*(enc(y,x,1-(o))-enc(x,y,o)))

int Q[NBV][4][7]; // Weights: Q[r][i][d] = weight of i^th vertex of r^th big vertex in direction d
                  // Directions 0-3 corresponds to intra-K_4,4 neighbours, and
                  // 4 = Left or Down, 5 = Right or Up, 6 = self
int adj[NBV][4][7][2]; // Complete adjacency list, including both directions along an edge and self-loops
                       // adj[p][i][d]={q,j} <-> d^th neighbour of encoded vertex p, index i, is 
                       //                        encoded vertex q, index j                    
                       // d as above
int okv[NBV][4];
int XBplus[(N+2)*N*2];
int *XBa=XBplus+N*2;// XBa[enc(x,y,o)] = State (0..15) of big vert
                    // Allow extra space to avoid having to check for out-of-bounds accesses
                    // (Doesn't matter that they wrap horizontally, since the weights will be 0 for these edges.)
int QBa[NBV][3][16][16]; // Weights for big verts (derived from Q[])
                         // QBa[enc(x,y,o)][d][s0][s1] = total weight from big vert (x,y,o) in state s0
                         //                              to the big vert in direction d in state s1
                         // d=0 is intra-K_4,4, d=1 is Left/Down, d=2 is Right/Up
#define QB(x,y,o,d,s0,s1) (QBa[enc(x,y,o)][d][s0][s1])
#define QBI(inv,x,y,o,d,s0,s1) (QBa[encI(inv,x,y,o)][d][s0][s1])
#define XB(x,y,o) (XBa[enc(x,y,o)])
#define XBI(inv,x,y,o) (XBa[encI(inv,x,y,o)])

// Globals corresponding to command line options
int wn,seed,mode,weightmode,statemap[2];
double ttr;
char *inprobfile,*outprobfile,*outstatefile;

#define MAX(x,y) ((x)>(y)?(x):(y))
// Isolate random number generator in case we need to replace it with something better
void initrand(int seed){srandom(seed);}
int randbit(void){return (random()>>16)&1;}
int randnib(void){return (random()>>16)&15;}
int randint(int n){return random()%n;}

double cpu(){return clock()/(double)CLOCKS_PER_SEC;}

void initgraph(void){
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

void getbigweights(void){// Get derived weights on "big graph" QB[] from Q[]
  // Intended so that the energy is calculated by summing over each big-edge exactly once,
  // not forwards and backwards.  See val() below.
  // That means that the off-diagonal bit of Q[][] has to be replaced by Q+Q^T, not
  // (1/2)(Q+Q^T) as would happen if you later intended to sum over both big-edge directions.
  // The self-loops are incorporated (once) into the intra-K_4,4 terms, QB(*,*,*,0,*,*).
  int d,i,j,k,o,p,q,x,y,po,s0,s1,x0,x1;
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

void initweights(void){// Initialise a symmetric weight matrix with random +/-1s
  int d,i,j,p,q;
  for(p=0;p<NBV;p++)for(i=0;i<4;i++)for(d=0;d<7;d++){
    q=adj[p][i][d][0];j=adj[p][i][d][1];
    Q[p][i][d]=0;
    if(!(q>=0&&okv[p][i]&&okv[q][j]))continue;
    // weightmode
    // 0           All of Q_ij independently +/-1
    // 1           Diagonal not allowed
    // 2           Upper triangular
    // 3           All of Q_ij allowed, but constrained symmetric
    switch(weightmode){
    case 0:Q[p][i][d]=randbit()*2-1;break;
    case 1:if(d<6)Q[p][i][d]=randbit()*2-1;break;
    case 2:if((d<4&&deco(p)==0)||d==5)Q[p][i][d]=randbit()*2-1;break;
    case 3:if((d<4&&deco(p)==0)||d==5)Q[p][i][d]=2*(randbit()*2-1); else if(d==6)Q[p][i][d]=(randbit()*2-1);
    }
  }
  getbigweights();
}

void init_state(void){// Initialise state randomly
  int x,y,o;
  for(x=0;x<N;x++)for(y=0;y<N;y++)for(o=0;o<2;o++)XB(x,y,o)=randnib();
}
void initstatebiased(int count[N][N][256]){
  int k,r,s,x,y;
  for(x=0;x<N;x++)for(y=0;y<N;y++){
    for(k=0,s=0;k<256;k++)s+=count[x][y][k];
    r=randint(s);
    for(k=0;k<256;k++){r-=count[x][y][k];if(r<0)break;}
    assert(k<256);
    XB(x,y,0)=k&15;XB(x,y,1)=k>>4;
  }
}

void writeweights(char *f){
  int d,i,j,p,q;
  FILE *fp;
  fp=fopen(f,"w");assert(fp);
  fprintf(fp,"%d %d\n",N,N);
  for(p=0;p<NBV;p++)for(i=0;i<4;i++)for(d=0;d<7;d++){
    q=adj[p][i][d][0];j=adj[p][i][d][1];
    if(q>=0)fprintf(fp,"%d %d %d %d   %d %d %d %d   %8d\n",
                    decx(p),decy(p),deco(p),i,
                    decx(q),decy(q),deco(q),j,
                    Q[p][i][d]);
  }
  fclose(fp);
}

int readweights(char *f){
  int d,i,n,p,w,v0,v1,x0,y0,o0,i0,e0,x1,y1,o1,i1,nx,ny,gtr;
  char l[1000];
  FILE *fp;
  printf("Reading weight matrix from file \"%s\"\n",f);
  fp=fopen(f,"r");assert(fp);
  assert(fgets(l,1000,fp));
  n=sscanf(l,"%d %d %d",&nx,&ny,&gtr);assert(n>=2);if(n==2)gtr=1000000;// gtr=ground truth (if stored)
  assert(nx==N&&ny==N);
  // Ensure weights=0 for edges that go out of bounds
  for(p=0;p<NBV;p++)for(i=0;i<4;i++){okv[p][i]=0;for(d=0;d<7;d++)Q[p][i][d]=0;}
  while(fgets(l,1000,fp)){
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
  return gtr;
}

void prstate(FILE*fp,int style){
  // style = 0: hex grid
  // style = 1: hex grid xored with previous, "gauge-fixed" in the +/-1 case
  // style = 2: list of vertices
  static int X0[NBV]={0};
  int nb[16];
  int i,j,o,p,t,x,xor;
  x=xor=0;
  if(style==1){
    xor=-1;
    if(statemap[0]==-1){
      for(i=1,nb[0]=0;i<16;i++)nb[i]=nb[i>>1]+(i&1);
      for(i=0,t=0;i<N;i++)for(j=0;j<N;j++)for(o=0;o<2;o++)t+=nb[XB(i,j,o)^X0[enc(i,j,o)]];
      x=(t>=NV/2?15:0);
    }
  }
  if(style<2){
    for(j=N-1;j>=0;j--){
      for(i=0;i<N;i++){fprintf(fp," ");for(o=0;o<2;o++)fprintf(fp,"%X",XB(i,j,o)^(X0[enc(i,j,o)]&xor)^x);}
      fprintf(fp,"\n");
    }
  }else{
    for(p=0;p<NBV;p++)for(i=0;i<4;i++)fprintf(fp,"%d %d %d %d  %4d\n",decx(p),decy(p),deco(p),i,statemap[(XBa[p]>>i)&1]);
  }
  memcpy(X0,XBa,sizeof(X0));
}

int opt0(double ttr,int pr){// Simple K_4,4-wise optimisation
  int r,v,x,y,s0,s1,bv,cv,vmin;
  long long int nn;
  double t0,t1,tt;
  bv=1000000000;nn=0;t0=cpu();t1=0;
  do{
    init_state();
    cv=val();
    do{
      for(x=0;x<N;x++)for(y=0;y<N;y++){
        vmin=1000000000;
        for(s0=0;s0<16;s0++)for(s1=0;s1<16;s1++){
          v=QB(x,y,0,0,s0,s1);
          v+=QB(x,y,0,1,s0,XB(x-1,y,0));
          v+=QB(x,y,0,2,s0,XB(x+1,y,0));
          v+=QB(x,y,1,1,s1,XB(x,y-1,1));
          v+=QB(x,y,1,2,s1,XB(x,y+1,1));
          if(v<vmin){vmin=v;XB(x,y,0)=s0;XB(x,y,1)=s1;}
        }
      }
      v=val();r=(v<cv);if(r)cv=v;
    }while(r);
    nn++;
    tt=cpu()-t0;
    if(cv<bv||tt>=t1||tt>=ttr){
      if(cv<bv)bv=cv;
      if(pr){printf("%12lld %10d %8.2f\n",nn,bv,tt);fflush(stdout);}
      t1=MAX(tt*1.2,tt+5);
    }
  }while(tt<ttr);
  return bv;
}

int lineexhaust(int c,int d){
  // If d=0 exhaust column c, if d=1 exhaust row c
  // Comments and variable names are as if in the column case (d=0)
  
  int b,o,r,s,v,smin0,smin1,vmin0,vmin1;
  int v0[16],v1[16];// Map from boundary state to value
  int h0[16][N][2],h1[16][N][2];// History

  for(r=0;r<N;r++){
    for(b=0;b<16;b++){// b = state of (c,r,1)
      if(r>0){
        vmin0=1000000000;smin0=-1;
        for(s=0;s<16;s++){// s = state of (c,r-1,1)
          v=v0[s]+QBI(d,c,r-1,1,2,s,b);
          if(v<vmin0){vmin0=v;smin0=s;}
        }
        memcpy(h1[b],h0[smin0],(2*r-1)*sizeof(int));
        h1[b][r-1][1]=smin0;
      }else vmin0=0;
      vmin1=1000000000;smin1=-1;
      for(s=0;s<16;s++){// s = state of (c,r,0)
        v=QBI(d,c,r,0,0,s,b)+
          QBI(d,c,r,0,1,s,XBI(d,c-1,r,0))+
          QBI(d,c,r,0,2,s,XBI(d,c+1,r,0));
        if(v<vmin1){vmin1=v;smin1=s;}
      }
      v1[b]=vmin0+vmin1;
      h1[b][r][0]=smin1;
    }//b
    memcpy(v0,v1,sizeof(v0));
    memcpy(h0,h1,sizeof(h0));
  }//r

  vmin0=1000000000;smin0=-1;
  for(s=0;s<16;s++){// s = state of (c,N-1,1)
    v=v0[s];
    if(v<vmin0){vmin0=v;smin0=s;}
  }
  for(r=0;r<N;r++)for(o=0;o<2;o++)XBI(d,c,r,o)=h0[smin0][r][o];
  XBI(d,c,N-1,1)=smin0;
  return vmin0;
}

int planeexhaust(int o){// not currently used
  // Exhaust (*,*,o) "plane" (x=*, y=*, o fixed)
  // Comments and variable names are as if in the case o=0 (horizontally connected nodes)
  int b,c,r,s,v,smin,vmin;
  int v0[16],v1[16];// Map from boundary state to value
  int h0[16][N],h1[16][N];// History
  for(r=0;r<N;r++){
    for(b=0;b<16;b++)v0[b]=0;
    for(c=0;c<N;c++){
      // Following is convenient but inefficient: should optimise bitwise
      for(b=0;b<16;b++){// b = state of (c+1,r,0)
        vmin=1000000000;smin=-1;
        for(s=0;s<16;s++){// s = state of (c,r,0)
          v=v0[s]+QBI(o,c,r,0,2,s,b)+QBI(o,c,r,0,0,s,XBI(o,c,r,1));
          if(v<vmin){vmin=v;smin=s;}
        }
        memcpy(h1[b],h0[smin],c*sizeof(int));
        h1[b][c]=smin;
        v1[b]=vmin;
      }//b
      memcpy(v0,v1,sizeof(v0));
      memcpy(h0,h1,sizeof(h0));
    }//c
    for(c=0;c<N;c++)XBI(o,c,r,0)=h0[0][c];
  }
  return 0;
}

void shuf(int*a,int n){
  int i,j,t;
  for(i=0;i<n-1;i++){
    j=i+randint(n-i);t=a[i];a[i]=a[j];a[j]=t;
  }
}

int opt1(double ttr,int pr,int tns,double *tts){// Optimisation using line (column/row) exhausts
  int o,r,v,x,bv,cv,ns,new,last;//,k,y,count[N][N][256];
  long long int nn;
  double t0,t1,tt;
  bv=1000000000;nn=0;t0=cpu();t1=0;ns=0;
  //for(x=0;x<N;x++)for(y=0;y<N;y++)for(k=0;k<256;k++)count[x][y][k]=1;
  do{
    //initstatebiased(count);
    init_state();
    cv=val();
    r=0;
    while(1){
      int i,ord[2*N];
      for(i=0;i<2*N;i++)ord[i]=i;
      //shuf(ord,2*N);
      shuf(ord,N);shuf(ord+N,N);
      for(i=0;i<2*N;i++){
        x=ord[i]%N;o=ord[i]/N;
        lineexhaust(x,o);
        v=val();assert(v<=cv);
        if(v<cv){cv=v;r=0;}else{r+=1;if(r==2*N)goto el0;}
        // (It's actually possible that 2N isn't enough to ensure there is no more improvement
        // possible, because lineexhaust() can change the state to something of equal value.)
      }
    }
  el0:
    //for(x=0;x<N;x++)for(y=0;y<N;y++)count[x][y][XB(x,y,0)+(XB(x,y,1)<<4)]+=1;
    if(pr==2&&cv<=bv){printf("\n");prstate(stdout,1);printf("cv %d\n",cv);}
    nn++;
    tt=cpu()-t0;
    if((new=(cv<bv))){bv=cv;ns=0;}
    if(cv==bv)ns++;
    last=(tt>=ttr&&ns>=tns);
    if(new||tt>=t1||last){
      t1=MAX(tt*1.2,tt+5);
      if(pr==1){printf("%12lld %10d %10d %8.2f\n",nn,bv,ns,tt);fflush(stdout);}
    }
  }while(!last);
  if(tts)*tts=tt/ns;
  return bv;
}

void initoptions(int ac,char**av){
  int opt;
  wn=NV;inprobfile=outprobfile=outstatefile=0;seed=time(0);ttr=1e10;statemap[0]=0;statemap[1]=1;
  weightmode=0;mode=0;
  while((opt=getopt(ac,av,"m:n:o:s:t:w:x:O:"))!=-1){
    switch(opt){
    case 'm': mode=atoi(optarg);break;
    case 'n': wn=atoi(optarg);assert(wn<=NV);break;
    case 'o': outprobfile=strdup(optarg);break;
    case 'O': outstatefile=strdup(optarg);break;
    case 's': seed=atoi(optarg);break;
    case 't': ttr=atof(optarg);break;
    case 'w': weightmode=atoi(optarg);break;
    case 'x': statemap[0]=atoi(optarg);break;
    default:
      fprintf(stderr,"Usage: %s [-m mode] [-n workingnodes] [-o outputprobfile] [-s seed] "
              "[-t targettimetorun] [-w weightmode] [-x lowerstatevalue] [-O outputstatefile] "
              "[inputprobfile]\n",av[0]);
      exit(1);
    }
  }
  if(optind<ac)inprobfile=strdup(av[optind]);
  printf("N=%d\n",N);
  printf("Mode: %d\n",mode);
  printf("Seed: %d\n",seed);
  printf("Target time to run: %gs\n",ttr);
}

int main(int ac,char**av){
  int gtr;
  initoptions(ac,av);
  memset(XBplus,0,sizeof(XBplus));
  initrand(seed);
  initgraph();
  if(inprobfile){
    gtr=readweights(inprobfile);
  }else{
    initweights();printf("Initialising random weight matrix with %d working node%s\n",wn,wn==1?"":"s");
  }
  printf("%d working node%s out of %d\n",wn,wn==1?"":"s",NV);
  printf("States are %d,%d\n",statemap[0],statemap[1]);
  if(outprobfile){writeweights(outprobfile);printf("Wrote weight matrix to file \"%s\"\n",outprobfile);}
  switch(mode){
  case 0:// Find single minimum value
    opt1(ttr,1,1,0);
    if(outstatefile){FILE*fp=fopen(outstatefile,"w");prstate(fp,2);fclose(fp);}
    break;
  case 1:;// Find average minimum value
    double v,s0,s1,s2;
    s0=s1=s2=0;
    while(1){
      initweights();
      v=opt1(ttr,0,1,0);
      s0+=1;s1+=v;s2+=v*v;
      printf("%12g %12g %12g\n",s0,s1/s0,sqrt((s2-s1*s1/s0)/(s0-1)));
    }
    break;
  case 2:;// Find rate of solution generation
    double tts;
    gtr=opt1(0.5,1,1000,&tts);
    printf("Time to solution %gs, assuming true minimum is %d\n",tts,gtr);
    break;
  }
  return 0;
}
