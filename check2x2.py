# Finds the average minimum value of QUBO on C_2 by exhaustively solving random instances
#
# Note that the C_2 graph is effectively an octagon where each octagon vertex
# contains 4 C_2 vertices, and octagon edges are alternately K_4,4s and 4 parallel lines.
# 
#     KL
#    L  K
#    K  L
#     LK
#
#   K=K_4,4
#   L=4 lines

def tr(x): return 1-2*x

from random import randrange

def k44():# pick random k44 weights
  l=[range(4) for i in range(4)]
  for i in range(4):
    for j in range(4):
      l[i][j]=randrange(2)*2-1
  return l

def lines():# pick random line weights
  return [randrange(2)*2-1 for i in range(4)]

def valk44(g,b0,b1):
  v=0
  for i in range(4):
    for j in range(4):
      v+=g[i][j]*tr((b0>>i)&1)*tr((b1>>j)&1)
  return v

def vallines(g,b0,b1):
  v=0
  for i in range(4):
    v+=g[i]*tr((b0>>i)&1)*tr((b1>>i)&1)
  return v


s0=0;s1=s2=0.  
while 1:
  vv=[[1e10]*16 for i in range(16)]
  for i in range(16): vv[i][i]=0
  for s in range(4):# steps around the loop
    
    g=k44()
    vv1=[range(16) for i in range(16)]
    for i in range(16):
      for k in range(16):
        vv1[i][k]=min([valk44(g,i,j)+vv[j][k] for j in range(16)])
    vv=vv1
    
    g=lines()
    vv1=[range(16) for i in range(16)]
    for i in range(16):
      for k in range(16):
        vv1[i][k]=min([vallines(g,i,j)+vv[j][k] for j in range(16)])
    vv=vv1
  
  v=min([vv[i][i] for i in range(16)])
  s0+=1;s1+=v;s2+=v*v
  if s0>1: print "%8d %12g %12g"%(s0,s1/s0,((s2-s1*s1/s0)/(s0-1))**.5)
