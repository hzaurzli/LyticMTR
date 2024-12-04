import os,sys,re,time
import random
import subprocess as sub
from subprocess import *
import subprocess as sub
import glob
import shutil

class tools:
    def __init__(self):
        self.hmmsearch = 'hmmsearch'

    def run(self, cmd, wkdir=None):
        sys.stderr.write("Running %s ...\n" % cmd)
        p = Popen(cmd, shell=True, cwd=wkdir)
        p.wait()
        return p.returncode
        
    def run_hmmsearch_2(self,out, e_val, hmm, inputfile):
        cmd = '%s --domtblout %s -E %s --cpu 2 %s %s' % (self.hmmsearch, out, e_val, hmm, inputfile)
        return cmd



def fasta2dict_2(fasta_name):
    with open(fasta_name) as fa:
        fa_dict = {}
        for line in fa:
            line = line.replace('\n', '')
            if line.startswith('>'):
              seq_name = line[1::].strip()
              fa_dict[seq_name] = ''
            else:
              fa_dict[seq_name] += line.replace('\n', '')
    return fa_dict
    
    
dic_fa = {}
with open('/home/user/Desktop/laber/NCBI_redup.fa') as f:
    lines = f.readlines()  # 读取所有行
    first_line = lines[0]
    dic_fa = fasta2dict_2('/home/user/Desktop/laber/NCBI_redup.fa')
f.close()

f2 = open('/home/user/Desktop/laber/all_protein.txt')
Domain_Info_lis = []
with open('/home/user/Desktop/laber/Domain_Info.txt', 'w') as w2:
  for line in f2:
    if line[0] != "#" and len(line.split())!=0:
      arr = line.strip().split(" ")
      arr = list(filter(None, arr))
      name = arr[0]

      li = arr[0] + '\t' + arr[3] + '(Length:' + arr[5] + ')' + '\t' + arr[4].split('.')[0] + '(Length:' + arr[5] + ')' + '\t' + arr[21] + '\t' + arr[19] + '-' + arr[20] + '\n'
      Domain_Info_lis.append(li)
 
  Domain_Info_lis_new = list(set(Domain_Info_lis))
  for line in Domain_Info_lis_new:
    print(line)
    w2.write(line)
w2.close()
