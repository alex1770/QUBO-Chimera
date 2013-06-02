# Converts my file format for QUBO problems (Chimera structure of vertices shown) to
# standard format (vertices are just numbers).

import sys
d={};n=0
for x in sys.stdin:
  y=x.strip()
  if y[0]=='#': print y;continue
  z=y.split()
  if len(z)!=2:
    v0=tuple(z[:4])
    v1=tuple(z[4:8])
    w=float(z[8])
    if v0 not in d: n+=1;d[v0]=n
    if v1 not in d: n+=1;d[v1]=n
    print "%d %d %g"%(d[v0],d[v1],w)
