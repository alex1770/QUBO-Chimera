#!/usr/bin/python
# Scheduler: distributes a job queue amongst 'n' processes
# Replacement for stupid multiprocessing.Pool.stuff
# Usage: schedule.py [options] RUNSCRIPT JOBLIST
# Each line in the file JOBLIST gets passed as an argument to the script RUNSCRIPT
# If -p <params> is used then RUNSCRIPT also gets passed the constant argument <params>
# If -s <startscript> is used then startscript runs first (getting the argument <params> if present)
# There is no asynchronous mode at the moment: you have to wait for a process to complete
# before seeing its output (though you'd normally redirect the output from the jobs anyway).
# Todo: make it kill its child processes if killed (and stop if stopped, cont if cont'd)

import optparse,multiprocessing,subprocess,datetime,sys,Queue,thread
from optparse import OptionParser

def waiter(id,p):
  (out,err)=p.communicate()
  results.put((id,out,err,p.returncode))

parser = OptionParser(usage="usage: %prog [options] RUNSCRIPT JOBLIST")
parser.add_option("-?","--wtf",action="help",help="Show this help message and exit")
parser.add_option("-s","--start",dest="start",help="Start script (default: none)")
parser.add_option("-n","--ncpus",dest="ncpus",help="Number of cpus to schedule to (default: all)")
parser.add_option("-p","--params",dest="params",help="Parameter(s) to pass to runscript (and start script, if it exists)")

results=Queue.Queue()
(options,args)=parser.parse_args()
start=options.start
ncpus=options.ncpus
if ncpus==None: ncpus=multiprocessing.cpu_count()
else: ncpus=int(ncpus)
params=options.params
if params==None: params=""
else: params=" "+params
if len(args)!=2: parser.error("Expected two arguments")
runscript=args[0]
joblistfn=args[1]

if __name__ == '__main__':
  if start: subprocess.call("%s%s"%(start,params),shell=True)
  f=open(joblistfn,'r');jobs=f.read().rstrip('\n').split('\n');f.close()
  n=len(jobs)
  a=0# #active
  s=0# #started
  # #completed = s-a
  while s-a<n:
    while a<ncpus and s<n: 
      print datetime.datetime.now(),"STARTING",jobs[s];sys.stdout.flush()
      p=subprocess.Popen("%s %s%s"%(runscript,jobs[s],params),stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)
      thread.start_new_thread(waiter,(s,p))
      a+=1;s+=1
    (i,out,err,rc)=results.get()
    sys.stdout.write(out)
    sys.stderr.write(err)
    print datetime.datetime.now(),"ENDING",jobs[i],"with return code",rc
    sys.stdout.flush();sys.stderr.flush()
    a-=1
