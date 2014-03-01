#!/usr/bin/python
# Scheduler: distributes a job queue amongst 'n' processes
# Replacement for stupid multiprocessing.Pool.stuff
# Usage: schedule.py [options] RUNSCRIPT JOBLIST
# Each line in the file JOBLIST gets passed as an argument to the script RUNSCRIPT
# If -p <params> is used then RUNSCRIPT also gets passed the constant argument <params>
# If -s <startscript> is used then startscript runs first (getting the argument <params> if present)
#
# There is no asynchronous mode: you have to wait for a process to complete before seeing
# its output (though you'd normally redirect the output from the jobs anyway).
#
# Jobids (lines in JOBLIST) are regarded as unique, so if a job occurs twice in the queue,
# it will only get run once.
#
# If you update the JOBLIST file during a run, then it will reload the job queue. This is
# a convenience feature for test code and does not use locks, so has a (very) small chance
# of failure due to the race condition.
# 
# Todo: possibly make it kill its child processes if killed (and stop if stopped, cont if cont'd)

import os,sys,optparse,multiprocessing,subprocess,datetime,Queue,thread
from optparse import OptionParser

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
jobmtime=-1
jobsstarted={}

def loadjobs():# Load new jobs file if more recently modified than when previously loaded
  global jobmtime,jobs
  m=os.stat(joblistfn).st_mtime
  if m>jobmtime:
    f=open(joblistfn,'r');jobs=[x for x in f.read().split('\n')[:-1] if x not in jobsstarted];f.close()
    n=len(jobs)
    print datetime.datetime.now(),["RE",""][jobmtime<0]+"LOADING jobqueue with",n,"job"+["s",""][n==1]
    jobmtime=m
    return 1
  return 0

def waiter(id,p):
  (out,err)=p.communicate()
  results.put((id,out,err,p.returncode))

if __name__ == '__main__':
  if start: subprocess.call("%s%s"%(start,params),shell=True)
  loadjobs()
  n=len(jobs)# n = number of jobs in current jobqueue
  a=0# number of active jobs
  s=0# number of processed (attempted-started) jobs from current jobqueue
  while s<n or a>0:
    while s<n and a<ncpus:
      job=jobs[s]
      if job in jobsstarted:
        print datetime.datetime.now(),"IGNORING duplicate job",job;sys.stdout.flush()
      else:
        print datetime.datetime.now(),"STARTING",job;sys.stdout.flush()
        p=subprocess.Popen("%s %s%s"%(runscript,job,params),stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)
        thread.start_new_thread(waiter,(job,p))
        jobsstarted[job]=1
        a+=1
      s+=1
    (job,out,err,rc)=results.get()
    sys.stdout.write(out)
    sys.stderr.write(err)
    print datetime.datetime.now(),"ENDING  ",job,"with return code",rc
    if loadjobs(): n=len(jobs);s=0
    sys.stdout.flush();sys.stderr.flush()
    a-=1
