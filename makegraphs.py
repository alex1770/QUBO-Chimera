# We're comparing order statistics from 100 things (V5 data) with order statistics of 1000 things (V6 and Prog-QUBO data).
# Put on the same graph by downsampling the 1000 samples to 100.
# What does the 87^th lowest value out of 100 (counting from 1) correspond to in 1000 ordered things?
#
# It's approximately the 865^th lowest value, but better is to imagine taking a random
# subset of size 100 from your size 1000 sample and working out the 87^th lowest value of
# that. The rank of that item in your size 1000 sample is a random variable with negative
# hypergeometric distribution:
# P(rank_{1000} = s) = (s-1 choose 86)*(1000-s choose 100-87)/(1000 choose 100)
#
# (Ranks start at 1 here.)

import scipy
import scipy.special
from math import log,exp
def fact(x): return scipy.special.gamma(x+1)
def logfact(x): return scipy.special.gammaln(x+1)
def logbin(n,r): return logfact(n)-logfact(r)-logfact(n-r)
def bin(n,r): return exp(logbin(n,r))

from subprocess import Popen,PIPE

N=1000;n=100

for wn in [439,502]:
  scaleres=[]
  absres=[]
  absresnosetup=[]
  
  if wn==439:
    f=open('Fig6.439.sorted','r')
    V5=[]
    for x in f.read().strip().split('\n'):
      if x[0]!='#': V5.append(int(x))
    f.close()
    assert len(V5)==n
    V5t=[None]+[1000./V5[n-1-i]*.29e-3 for i in range(n)]
    V5tm=(V5t[n/2]+V5t[(n+1)/2])/2
    V5r=[None]+[x for x in V5t[1:]]
    absres.append([x+0.201 if x else None for x in V5r])
    absresnosetup.append(V5r)
    scaleres.append([x/V5tm if x else None for x in V5r])
  
  f=open('Fig8.%d.sorted'%wn,'r')
  V6=[]
  for x in f.read().strip().split('\n'):
    if x[0]!='#': V6.append(int(x))
  f.close()
  assert len(V6)==N
  V6t=[None]+[10000./V6[N-1-i]*.126e-3 for i in range(N)]
  V6m=(V6t[N/2]+V6t[(N+1)/2])/2
  V6r=[None]
  for r in range(1,n+1):
    t=0
    for s in range(1,N+1): t+=bin(s-1,r-1)*bin(N-s,n-r)/bin(N,n)*V6t[s]
    V6r.append(t)
  absres.append([x+0.036 if x else None for x in V6r])
  absresnosetup.append(V6r)
  scaleres.append([x/V6m if x else None for x in V6r])
  
  f=open('output/weightmode5/%d.strat0%s/summary.sorted-by-optimality-and-TTF'%(wn,'.m0' if wn==439 else ''),'r')
  l=f.read().strip().split('\n');f.close()
  S0t=[None]
  for x in l:
    if x[0]=='#': continue
    y=x.split()
    if y[2]=='0': S0t.append(float(y[3]))
    else: S0t.append(None)
  assert len(S0t)==n+1
  S0m=(S0t[n/2]+S0t[(n+1)/2])/2
  S0r=[None]+[x if x else None for x in S0t[1:]]
  scaleres.append([x/S0m if x else None for x in S0r])

  f=open('output/weightmode5/%d.strat1/summary.sorted-by-TTS'%wn,'r')
  l=f.read().strip().split('\n')[1:];f.close()
  assert len(l)==N
  S1t=[None]+[float(x.split()[2]) for x in l]
  S1m=(S1t[N/2]+S1t[(N+1)/2])/2
  S1r=[None]
  for r in range(1,n+1):
    t=0
    for s in range(1,N+1): t+=bin(s-1,r-1)*bin(N-s,n-r)/bin(N,n)*S1t[s]
    S1r.append(t)
  absres.append(S1r)
  absresnosetup.append(S1r)
  scaleres.append([x/S1m if x else None for x in S1r])

  for (name,res) in [('timecomp',absres),('timecomp-nosetup',absresnosetup),('scaling',scaleres)]:
    fn='%s%d'%(name,wn)
    f=open(fn,'w')
    for r in range(1,101):
      print >>f,"%4d"%r,
      for x in res:
        if x[r]: print >>f,"%12g"%(log(x[r])/log(10)),
        else: print >>f,"%12s"%"-",
      print >>f
    f.close()
    print "Written scaling data to",fn

  fn='scaling%d'%wn
  p=Popen("gnuplot",shell=True,stdin=PIPE).stdin
  print >>p,'set terminal postscript color solid "Helvetica" 9'
  print >>p,'set output "%s.ps"'%fn
  print >>p,'set zeroaxis'
  print >>p,'set xrange [0:101]'
  print >>p,'set key left'
  print >>p,'set title "log_10 running time relative to median, n=%d"'%wn
  print >>p,'set xlabel "Hardness rank, p, from 1 to 100"'
  print >>p,'set ylabel "log_10(TTS(p)/TTS(50.5))"'
  print >>p,'set y2tics mirror'
  if wn==439:
    print >>p,'plot "scaling439" using ($1):($2) title "D-Wave V5, McGeoch n=439 set", "scaling439" using ($1):($3) title "D-Wave V6, McGeoch large n=439 set", "scaling439" using ($1):($4) title "Prog-QUBO-S0, Set 1 (n=439)", "scaling439" using ($1):($5) title "Prog-QUBO-S1, Set 1 (n=439)"'
  else:
    print >>p,'plot "scaling502" using ($1):($2) title "D-Wave V6, McGeoch n=502 set", "scaling502" using ($1):($3) title "Prog-QUBO-S0, Set 2 (n=502)", "scaling502" using ($1):($4) title "Prog-QUBO-S1, Set 2 (n=502)"'
  p.close()
  print "Written graph to %s.ps"%fn

  fn='timecomp%d'%wn
  p=Popen("gnuplot",shell=True,stdin=PIPE).stdin
  print >>p,'set terminal postscript color solid "Helvetica" 9'
  print >>p,'set output "%s.ps"'%fn
  print >>p,'set zeroaxis'
  print >>p,'set xrange [0:101]'
  print >>p,'set key left'
  print >>p,'set title "log_10 running time in seconds, n=%d"'%wn
  print >>p,'set xlabel "Hardness rank, p, from 1 to 100"'
  print >>p,'set ylabel "log_10(TTS(p)/1 second)"'
  print >>p,'set y2tics mirror'
  if wn==439:
    print >>p,'plot "timecomp439" using ($1):($2) title "D-Wave V5, McGeoch n=439 set", "timecomp439" using ($1):($3) title "D-Wave V6, McGeoch large n=439 set", "timecomp439" using ($1):($4) title "Prog-QUBO-S1, Set 1 (n=439)"'
  else:
    print >>p,'plot "timecomp502" using ($1):($2) title "D-Wave V6, McGeoch n=502 set", "timecomp502" using ($1):($3) title "Prog-QUBO-S1, Set 2 (n=502)"'
  p.close()
  print "Written graph to %s.ps"%fn

  fn='timecomp-nosetup%d'%wn
  p=Popen("gnuplot",shell=True,stdin=PIPE).stdin
  print >>p,'set terminal postscript color solid "Helvetica" 9'
  print >>p,'set output "%s.ps"'%fn
  print >>p,'set zeroaxis'
  print >>p,'set xrange [0:101]'
  print >>p,'set key left'
  print >>p,'set title "log_10 running time in seconds, n=%d. D-Wave times exlude setup."'%wn
  print >>p,'set xlabel "Hardness rank, p, from 1 to 100"'
  print >>p,'set ylabel "log_10(TTS(p)/1 second)"'
  print >>p,'set y2tics mirror'
  if wn==439:
    print >>p,'plot "timecomp-nosetup439" using ($1):($2) title "D-Wave V5, McGeoch n=439 set", "timecomp-nosetup439" using ($1):($3) title "D-Wave V6, McGeoch large n=439 set", "timecomp-nosetup439" using ($1):($4) title "Prog-QUBO-S1, Set 1 (n=439)"'
  else:
    print >>p,'plot "timecomp-nosetup502" using ($1):($2) title "D-Wave V6, McGeoch n=502 set", "timecomp-nosetup502" using ($1):($3) title "Prog-QUBO-S1, Set 2 (n=502)"'
  p.close()
  print "Written graph to %s.ps"%fn
