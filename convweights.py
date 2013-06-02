# Converts instance of D-Wave One from their file format to mine.
# Instance files are from the ancillary list to http://arxiv.org/abs/1305.5837
# Working qubit map from http://arxiv.org/abs/1304.4595

wqm="""
XX X. X. ..
XX .X XX X.
XX XX XX X.
XX .X XX XX

.X XX XX XX
XX XX XX XX 
XX XX XX XX
XX XX XX .X

XX XX XX XX
XX XX XX .X
X. XX XX .X
XX XX XX .X

XX XX XX XX
XX XX XX .X
XX XX X. XX
.X X. XX ..
"""

l=[]
for x in wqm.strip().split('\n'): 
  y=x.strip()
  if y!='': l.append(list(y.replace(' ','')))
n=0;m=[None]*129
for y in range(3,-1,-1):
  for x in range(4):
    for o in [1,0]:
      for i in range(4):
        r=(3-y)*4+i
        c=x*2+1-o
        if l[r][c]=='X': n+=1;m[n]="%d %d %d %d"%(x,y,o,i)
assert n==108

import sys
print 4,4
for x in sys.stdin:
  y=x.strip()
  if y[0]=='#': print y;continue
  z=y.split();assert len(z)==3
  print m[int(z[0])]," ",m[int(z[1])],"%10s"%z[2]
