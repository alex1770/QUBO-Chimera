# Converts IM format (variables in {-1,1}) to QUBO format (variables in {0,1}), 
# also calculating the constant offset for the objective function.
# The vertices use the Chimera notation described in qubo.c: (x,y,o,i)
# Input consists of lines like
# 3 4 0 1   -1
# which means that h_{(3,4,0,1)}=-1
# and lines like
# 3 4 0 1   4 4 0 1    2
# which means J_{(3,4,0,1),(4,4,0,1)}=2

import sys
h={};J={};V={};con=0
for x in sys.stdin:
  y=x.strip()
  z=y.split()
  if y[0]=='#' or len(z)==2: print y;continue
  v0=tuple(z[:4]);V[v0]=1
  if len(z)==5: h[v0]=float(z[4]);continue
  if len(z)==9:
    v1=tuple(z[4:8]);w=float(z[8])
    if v0==v1:
      con+=w
    else:
      V[v1]=1
      J[(v0,v1)]=J.get((v0,v1),0)+w
    continue
  print >>sys.stderr,"Unrecognised line:",x;assert 0

d={}
for v0 in V:
  d[v0]=h.get(v0,0)
for (v0,v1) in J:
  d[v0]-=J[(v0,v1)]
  d[v1]-=J[(v0,v1)]
s=tr=0
for (v0,v1) in J: s+=J[(v0,v1)]
for v0 in d: tr+=d[v0]
print "# (QUBO value) - (IM value) = %g"%(-con+s+tr)
for v0 in d:
  if d[v0]!=0: print "%s %s %s %s"%v0,"  %s %s %s %s"%v0,"%10g"%(2*d[v0])
for (v0,v1) in J:
  if J[(v0,v1)]!=0: print "%s %s %s %s"%v0,"  %s %s %s %s"%v1,"%10g"%(4*J[(v0,v1)])
