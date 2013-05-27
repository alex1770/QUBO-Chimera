# Finds the average minimum value of QUBO on C_2 by exhaustively solving random instances
#
# The C_2 graph is effectively an octagon where each octagon vertex contains four C_2
# vertices, and octagon edges are alternately K_4,4s and a set of four parallel lines.
# 
#     KL
#    L  K
#    K  L
#     LK
#
#   K=K_4,4
#   L=4 lines

#def tr(x): return 2*x-1
def tr(x): return x
full=2
# full=0 <-> sum_{i<j} Q_ij x_i x_j
# full=1 <-> sum_{i!=j} Q_ij x_i x_j
# full=2 <-> sum_{i,j} Q_ij x_i x_j
# Q_ij in {-1,1}

from random import randrange
def randweight(): return randrange(2)*2-1

def k44():# pick random k44 weights and self-weights
  if full: return ([[randweight()+randweight() for i in range(4)] for j in range(4)],
                   [randweight() for i in range(4)])
  else: return [[[randweight() for i in range(4)] for j in range(4)]]

def lines():# pick random line weights and self-weights
  if full: return ([randweight()+randweight() for i in range(4)],
                   [randweight() for i in range(4)])
  else: return [[randweight() for i in range(4)]]

def valk44(g,b0,b1):
  v=sum([sum([g[0][i][j]*tr((b0>>i)&1)*tr((b1>>j)&1) for j in range(4)]) for i in range(4)])
  if full==2: v+=sum([g[1][i]*tr((b1>>i)&1)**2 for i in range(4)])
  return v

def vallines(g,b0,b1):
  v=sum([g[0][i]*tr((b0>>i)&1)*tr((b1>>i)&1) for i in range(4)])
  if full==2: v+=sum([g[1][i]*tr((b1>>i)&1)**2 for i in range(4)])
  return v


s0=0;s1=s2=0.  
while 1:
  vv=[[1e10]*16 for i in range(16)]# map from boundary values to optimum inside
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
