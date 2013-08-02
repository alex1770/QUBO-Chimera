#!/usr/bin/python

# Make graphs of the timing results from V5, V6, Prog-QUBO

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
# (Ranks start at 1 in the above example.)

import scipy
import scipy.special
from math import log,exp
def fact(x): return scipy.special.gamma(x+1)
def logfact(x): return scipy.special.gammaln(x+1)
def logbin(n,r): return logfact(n)-logfact(r)-logfact(n-r)
def bin(n,r):
  if r>=0 and r<=n: return exp(logbin(n,r))
  else: return 0.

from subprocess import Popen,PIPE

output='png'# 'png' or 'ps'
n=100# Resample down to this

# Assumed setup and read times in seconds
V5setup=.201;V5read=.29e-3
V6setup=.036;V6read=.126e-3

def add(label,times,setup):
  N=len(times)
  tr=[]
  for r in range(n):
    t=0
    for s in range(r,N-(n-1-r)): t+=bin(s,r)*bin(N-1-s,n-1-r)/bin(N,n)*times[s]# Yes it's stupidly inefficient
    tr.append(t)
  absres.append([x+setup for x in tr])
  absresnosetup.append(tr)
  med=(times[(N-1)/2]+times[N/2])/2# median of no setup case
  scaleres.append([x/med for x in tr])# scaled no setup case
  labels.append(label)

for wn in [439,502]:
  scaleres=[]
  absres=[]
  absresnosetup=[]
  labels=[]
  
  if wn==439:
    f=open('Fig6.439.sorted','r')
    V5=[]
    for x in f.read().strip().split('\n'):
      if x[0]!='#': V5.append(int(x))
    f.close()
    N=len(V5)
    add("D-Wave V5, McGeoch n=439 set",[1000./V5[N-1-i]*V5read for i in range(n)],V5setup)
  
  f=open('Fig8.%d.sorted'%wn,'r')
  V6=[]
  for x in f.read().strip().split('\n'):
    if x[0]!='#': V6.append(int(x))
  f.close()
  N=len(V6)
  add("D-Wave V6, McGeoch large n=%d set"%wn,[10000./V6[N-1-i]*V6read for i in range(N)],V6setup)
  
  for s in [1,3]:#[0,1,3,10]:
    f=open('output/weightmode5/%d.strat%d/summary.sorted-by-TTS'%(wn,s),'r')
    l=[]
    for x in f.read().strip().split('\n'):
      if x[0]!='#': l.append(float(x.split()[2]))
    f.close()
    add("Prog-QUBO-S%d, Set %d (n=%d)"%(s,1+(wn==502),wn),l,0)
  
  for (name,res) in [('timecomp',absres),('timecomp-nosetup',absresnosetup),('scaling',scaleres)]:
    fn='%s%d'%(name,wn)
    f=open(fn,'w')
    for i in range(len(labels)):
      print >>f,"# Column %d = %s"%(i+1,labels[i])
    for r in range(n):
      print >>f,"%4d"%(r+1),
      for x in res:
        assert x[r]
        print >>f,"%12g"%(log(x[r])/log(10)),
      print >>f
    f.close()
    print "Written scaling data to",fn

  for (fn,title,ylabel) in [
    ('scaling%d','"log_10 running time relative to median, n=%d"','"log_10(TTS(p)/TTS(50.5))"'),
    ('timecomp%d','"log_10 running time in seconds, n=%d"','"log_10(TTS(p)/1 second)"'),
    ('timecomp-nosetup%d','"log_10 running time in seconds, n=%d. D-Wave times exlude setup."','"log_10(TTS(p)/1 second)"')]:
    fn=fn%wn;title=title%wn
    p=Popen("gnuplot",shell=True,stdin=PIPE).stdin
    if output=='ps':
      print >>p,'set terminal postscript color solid "Helvetica" 9'
    elif output=='png':
      print >>p,'set terminal pngcairo size 1400,960'
      print >>p,'set bmargin 5;set lmargin 15;set rmargin 15;set tmargin 5'
    else: assert 0
    print >>p,'set output "%s.%s"'%(fn,output)
    print >>p,'set zeroaxis'
    print >>p,'set xrange [0:101]'
    print >>p,'set key left'
    print >>p,'set title',title
    print >>p,'set xlabel "Hardness rank, p, from 1 to 100"'
    print >>p,'set ylabel',ylabel
    print >>p,'set y2tics mirror'
    s='plot '
    for i in range(len(labels)):
      if i>0: s+=', '
      s+='"%s" using ($1):($%d) title "%s"'%(fn,i+2,labels[i])
    print >>p,s
    p.close()
    print "Written graph to %s.%s"%(fn,output)
