# Converts my file format for QUBO problems (Chimera structure of vertices shown) to
# standard format (vertices are just numbers, matrix is upper triangular).

import sys
e={}
for x in sys.stdin:
  y=x.strip()
  if y[0]=='#': print y;continue
  z=y.split()
  if len(z)!=2:
    v0=tuple(z[:4])
    v1=tuple(z[4:8])
    w=int(z[8])
    if v0>v1: (v0,v1)=(v1,v0)
    e[(v0,v1)]=e.get((v0,v1),0)+w

d={};n=0
l=list(e);l.sort()
for x in l:
  if e[x]!=0:
    for v in x:
      if v not in d: n+=1;d[v]=n

l=[]
for x in e:
  if e[x]!=0: l.append((d[x[0]],d[x[1]],e[x]))

l.sort()
for x in l:
  print "%d %d %d"%x
