# Converts instance of Ising format to QUBO format
# Expecting lines of the form
# x0 y0 o0 i0   x1 y1 o1 i1   w
# representing J((x0,y0,o0,i0),(x1,y1,o1,i1))=-w, or
# x0 y0 o0 i0   w
# representing h(x0,y0,o0,i0)=-w (external field)
# This format is not used by any current program, but it is what
# my program would used if it handled Ising case (-x-1) with fields, and
# it is what Boixo format would be if it used my vertex encoding. (At least
# it is a superset of Boixo format, since it also allows v>v'. See below.)
# The purpose of this conversion is to allow generation of naturally Ising
# problems for my program without having to include the converter in the
# instance generator.
#
# Ising (input) hamiltonian is
# - sum_{v!=v'}J(v,v')s(v)s(v') - sum_v h(v)s(v)
# for s(v)=+/-1
#
# QUBO (output) hamiltonian is
# sum_{v,v'}Q(v,v')x(v)x(v')
# for x(v)=0,1
#
# Note that the input sum includes J(v,v') as well as J(v',v), but conventionally there is
# an order on vertices and J() is chosen such that J(v,v')=0 for v>v'.  The classic {-1,1}
# spin-glass has J(v,v')=(+/-)1 for v<v' and J(v,v')=0 for v>=v'.  (Boixo format only has
# entries for J(v,v') when v<v'.)
#
# Looks like this program is a near-duplicate of IM2QUBO.py

import sys

N=0
h={};J={};V={}
for l in sys.stdin:
  m=l.strip()
  if m[0]=='#': print m;continue
  m=[int(x) for x in m.split()]
  if len(m)==2: n=max(m[0],m[1])
  elif len(m)==5:
    w=m[4]
    if w==0: continue
    v0=tuple(m[0:4]);V[v0]=1
    h[v0]=h.get(v0,0)-w
    N=max(N,v0[0]+1,v0[1]+1)
  elif len(m)==9:
    w=m[8]
    if w==0: continue
    v0=tuple(m[0:4]);V[v0]=1
    v1=tuple(m[4:8]);V[v1]=1
    J[(v0,v1)]=J.get((v0,v1),0)-w
    N=max(N,v0[0]+1,v0[1]+1,v1[0]+1,v1[1]+1)
  else: print >>sys.stderr,"Unrecognised line",l.rstrip();sys.exit(1)

print >>sys.stderr,"Read n =",n
print >>sys.stderr,"Inferring N =",N
print >>sys.stderr,"Use ./qubo -N%d"%N
assert N<=n and len(V)<=8*N*N

d={}
for v0 in V:
  d[v0]=-h.get(v0,0)
for (v0,v1) in J:
  d[v0]+=J[(v0,v1)]
  d[v1]+=J[(v0,v1)]
s=tr=0
for (v0,v1) in J: s-=J[(v0,v1)]
for v0 in d: tr+=d[v0]
for f in [sys.stdout,sys.stderr]: print >>f,"# (QUBO value) - (IM value) = %g"%(s+tr)
for (v0,v1) in J: J[(v0,v1)]*=4
for v0 in d: J[(v0,v0)]=-2*d[v0]

print N,N
for (v0,v1) in sorted(list(J)):
  print "%d %d %d %d"%v0,"  %d %d %d %d"%v1,"%10d"%(-J[(v0,v1)])
