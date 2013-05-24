#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

// QUBO solver (heuristic - no proof)
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
// o is the "orientation" (0=horizontally connected, 1=vertically connected)
// i is the index within the "semi-K4,4"

#define L 3
#define N 8
// Require 2^L >= N, but N doesn't have to be a power of 2

#define HORIZ 3
#define VERT (1<<(L+3))
#define M (1<<(2*L+3))
#define NV (N*N*8)
#define NE (8*N*(3*N-1))
#define enc(x,y,o,i) ((i)+((o)<<2)+((x)<<3)+((y)<<(L+3)))
#define decx(p) (((p)>>3)&((1<<L)-1))
#define decy(p) (((p)>>(L+3))&((1<<L)-1))
#define deco(p) (((p)>>2)&1)
#define deci(p) ((p)&3)
// Bit encoding of vertex:
// 876543210
// \|/\|/|\|
//  |  | | Bits 0,1 = i
//  |  | Bit 2 = o
//  |  Bits 3...(L+2) = x
//  Bit (L+3)...(2L+2) = y

// Convention is that smaller numbers always correspond to horizontal

int adj[M][6];// Neighbours as encoded vertices (-1 for non-existent)
              // 0-3 corresponds to intra-K_4,4 neighbours
              // 4 = Left or Down, 5 = Right or Up
int Q[M][6];// Weights
int X[M];// State (0 or 1)
int vlist[NV]; // encoded vertex list
int elist[NE][4]; // elist[e][0] = encoded start vertex of edge e
                  // elist[e][1] = edge number (0...5) from vertex elist[e][0] to elist[e][2]
                  // elist[e][2] = encoded end vertex of edge e
                  // elist[e][3] = edge number (0...5) from vertex elist[e][2] to elist[e][0]

void initrand(){srandom(time(0));}

int randbit(void){return (random()>>16)&1;}

void initgraph(void){
  int i,j,o,p,q,x,y,nv,ne;
  nv=ne=0;
  for(x=0;x<N;x++)for(y=0;y<N;y++)for(o=0;o<2;o++)for(i=0;i<4;i++){
    p=enc(x,y,o,i);vlist[nv++]=p;
    for(j=0;j<4;j++)adj[p][j]=enc(x,y,1-o,j);
    if(o==0){
      adj[p][4]=x>0?enc(x-1,y,o,i):-1;
      adj[p][5]=x<N-1?enc(x+1,y,o,i):-1;
    }else{
      adj[p][4]=y>0?enc(x,y-1,o,i):-1;
      adj[p][5]=y<N-1?enc(x,y+1,o,i):-1;
    }
    for(j=0;j<6;j++){
      q=adj[p][j];
      if(p<q){
        elist[ne][0]=p;elist[ne][1]=j;
        elist[ne][2]=q;elist[ne][3]=(j<4?i:9-j);
        ne++;
      }
    }
  }
  assert(nv==NV&&ne==NE);
}

void initweights(void){// Initialise a symmetric weight matrix with random +/-1s
  int i,j;
  for(i=0;i<NV;i++)for(j=0;j<6;j++)Q[i][j]=0; // To ensure weight=0 for non-existent edges
  for(i=0;i<NE;i++)
    Q[elist[i][0]][elist[i][1]]=Q[elist[i][2]][elist[i][3]]=randbit()*2-1;
}

void initspins(void){// Initialise state with random 0,1s
  int i;
  for(i=0;i<NV;i++)X[vlist[i]]=randbit();
}

void writeweights(char *f){
  int i,e0,e1,v0,v1;
  FILE *fp;
  fp=fopen(f,"w");assert(fp);
  fprintf(fp,"%d %d\n",N,N);
  for(i=0;i<NE;i++){
    v0=elist[i][0];v1=elist[i][2];
    e0=elist[i][1];e1=elist[i][3];
    fprintf(fp,"%d %d %d %d  %d        %d %d %d %d  %d        %6d\n",
            decx(v0),decy(v0),deco(v0),deci(v0),e0,
            decx(v1),decy(v1),deco(v1),deci(v1),e1,
            Q[v0][e0]);
  }
  fclose(fp);
}

void readweights(char *f){
  int i,w,x0,y0,o0,i0,e0,x1,y1,o1,i1,e1,nx,ny;
  FILE *fp;
  fp=fopen(f,"r");assert(fp);
  assert(fscanf(fp,"%d %d",&nx,&ny)==2);
  assert(nx==N&&ny==N);
  for(i=0;i<NE;i++){
    assert(fscanf(fp,"%d %d %d %d %d %d %d %d %d %d %d",
                  &x0,&y0,&o0,&i0,&e0,
                  &x1,&y1,&o1,&i1,&e1,
                  &w)==11);
    Q[enc(x0,y0,o0,i0)][e0]=Q[enc(x1,y1,o1,i1)][e1]=w;
  }
  fclose(fp);
}

int val(void){
  int i,v;
  v=0;
  for(i=0;i<NE;i++)v+=Q[elist[i][0]][elist[i][1]]*X[elist[i][0]]*X[elist[i][2]];
  return v;
}

int dval(int p){
  int j,v;
  v=0;
  for(j=0;j<6;j++)if(adj[p][j]>=0)v+=Q[p][j]*X[adj[p][j]];
  return v;
}

int main(int ac,char**av){
  printf("N=%d\n",N);
  initrand();
  initgraph();
  if(ac<2){initweights();printf("Initialising random weight matrix\n");}else
    {readweights(av[1]);printf("Reading weight matrix from file \"%s\"\n",av[1]);}
  writeweights("tempproblem");
  int i,p,r,bv,cv,dv;
  long long int nn;
  bv=1000000000;nn=0;
  while(1){
    initspins();
    cv=val();
    r=0;
    while(r<NV){
      for(i=0;i<NV;i++){
        p=vlist[i];
        dv=dval(p)*(1-2*X[p]);
        if(dv<0){cv+=dv;X[p]=1-X[p];r=0;} else {r++;if(r==NV)break;}
      }
    }
    nn++;
    if(cv<bv||((nn&(nn-1))==0)){
      if(cv<bv)bv=cv;
      printf("%12lld %10d %8.2f\n",nn,bv,clock()/(double)CLOCKS_PER_SEC);
      fflush(stdout);
    }
  }
  /*
  int b,i,p,s,v;
  int h0[16][N][2],h1[16][N][2];// history
  for(b=0;b<16;b++){
    vmin=1000000000;
    p=enc(c,r-1,1,0);
    for(s=0;s<16;s++){
      v=v0[s];
      for(i=0;i<4;i++)v+=Q[p+i][4]*((s>>i)&1)*((b>>i)&1);
      if(v<vmin){vmin=v;smin=s;}
    }
    memcpy(h1[b],h0[smin],(2*r-1)*sizeof(int));
    h1[b][r-1][1]=smin;
    h1[b][r][0]=0;
    p=enc(c,r,0,0);
    for(i=0;i<4;i++){
      v=0;
      for(j=0;j<4;j++)v+=Q[p+i][j]*((b>>j)&1);
      if(Q[p+i][4]>=0)v+=Q[p+i][4]*x[p-HORIZ];
      if(Q[p+i][5]>=0)v+=Q[p+i][5]*x[p+HORIZ];
      if(v<0){h1[b][r][0]|=1<<i;vmin+=v;}
    }
    v1[b]=vmin;
  }
  */
  return 0;
}
