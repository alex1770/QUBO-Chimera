# Converts instance of Boixo format (chimeraFigGo.pdf) instances to mine.
# My (x,y,o,i) = his 8x+64y+4(1-o)+i
# If the command line argument 'q' is supplied then it will convert to QUBO format,
# otherwise the output will be IM form and '-x-1' must be supplied to the qubo command.
# Must convert to QUBO format if fields are used.
# Converts rectangles to squares (e.g., 6x7 becomes 7x7).

def dec(b): return ((b>>3)&7,(b>>6)&7,((b>>2)&1)^1,b&3)

import sys
qubo=(len(sys.argv)>=2 and sys.argv[1].lower()=='q')# Convert to 0,1 (QUBO) mode
print >>sys.stderr,"Converting from Chimera(Boixo) to Chimera(Selby,%s)"%(['IM','QUBO'][qubo])
N=0
h={};J={};V={};field=0
for l in sys.stdin:
  m=l.strip()
  if m[0]=='#': print m;continue
  m=m.split()
  if len(m)==1: n=int(m[0]);continue
  assert len(m)==3
  w=int(m[2])
  if w==0: continue
  v0=dec(int(m[0]));V[v0]=1
  v1=dec(int(m[1]));V[v1]=1
  if v0==v1:
    h[v0]=h.get(v0,0)-w;field=1
  else:
    J[(v0,v1)]=J.get((v0,v1),0)-w
  N=max(N,v0[0]+1,v0[1]+1,v1[0]+1,v1[1]+1)

if field and not qubo: print >>sys.stderr,"Must convert to QUBO form if using fields";assert 0
print >>sys.stderr,"Read n =",n
print >>sys.stderr,"Inferring N =",N
print >>sys.stderr,"Use ./qubo%s -N%d"%([" -x-1",""][qubo],N)
assert len(V)<=n and n<=8*N*N

if qubo:
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
