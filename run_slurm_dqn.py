import os
import subprocess
import time
import random
import numpy as np

class hashabledict(dict):
  def __hash__(self):
    return hash(tuple(sorted(self.items())))

def pending():
  output = subprocess.check_output('squeue -u nimamo'.split())
  output = output.decode('utf8').split('\n')
  cc = 0
  for ll in output:
    if 'PD' in ll:
      cc += 1
  return cc


#rl_type = 'PG' # DQN, PG
regressor = 'SNN'
conv = 3
geomS = ['1-1-1', '2-2-2', '3-3-1', '3-3-3']
pmaxS = [.05, .1]
#train_params = [
                 ##{'tbs': 35, 'ti': 90, 'rti': 30},
                 ##{'tbs': 70, 'ti': 90, 'rti': 30},
                 #{'tbs': 100, 'ti': 50, 'rti': 25},
#]

tp = {'tbs': 100, 'ti': 50, 'rti': 25}

confDicts = []

lr = .005
hidden = 10

for rl_type in ['PG', 'DQN']:
  for pgepochs in [1, 5]:
    for encoder in ['ISI', 'Poisson']:
      for lsm in [0, 1]:
        if not lsm:
          confDicts.append(hashabledict({
            'rltype': rl_type,
            'regressor': regressor,
            'conv': conv,
            'encoder': encoder,
            'lr': lr,
            'lsm': lsm,
            'tbs': tp['tbs'],
            'ti': tp['ti'],
            'rti': tp['rti'],
            'hidden': hidden,
            'pgepochs': pgepochs,
          })) 
        else:
          for pmax in pmaxS:
            for macrocol in geomS:
              macSize = np.prod(list(map(int, macrocol.split('-'))))
              for minicol in geomS:
                minSize = np.prod(list(map(int, minicol.split('-'))))
                reservoir_size = minSize * macSize
                for rinp_ratio in [.25, .5, .75, 1]:
                  for rout_ratio in [.25, .5, .75, 1]:
                    rinp_size = max(1, int(reservoir_size * rinp_ratio))
                    rout_size = max(1, int(rinp_size * rout_ratio))
                    for specrad in [0]:
                      for alpha in [.01, .1]:
                        confDicts.append(hashabledict({
                          'rltype': rl_type,
                          'regressor': regressor,
                          'conv': conv,
                          'encoder': encoder,
                          'lr': lr,
                          'lsm': lsm,
                          'pmax': pmax,
                          'macrocol': macrocol,
                          'minicol': minicol,
                          'specrad': specrad,
                          'alpha': alpha,
                          'tbs': tp['tbs'],
                          'ti': tp['ti'],
                          'rti': tp['rti'],
                          'hidden': hidden,
                          'rinp': rinp_size,
                          'rout': rout_size,
                          'lsm_size': reservoir_size,
                          'pgepochs': pgepochs,
                        }))   
                      
  


command = "sbatch --export={} ./dqn_job_dt.sh"

confDicts = list(set(confDicts))
random.shuffle(confDicts)
print('Total confs:', len(confDicts))
for ii, cc in enumerate(confDicts):
  explist = []
  explist.append("RLTYPE='{}'".format(cc['rl_type']))
  explist.append("REGRESSOR='{}'".format(cc['regressor']))
  explist.append("ENCODER='{}'".format(cc['encoder']))
  explist.append("CONV={}".format(cc['conv']))
  explist.append("LR='{}'".format(cc['lr']))
  explist.append("tbs='{}'".format(cc['tbs']))
  explist.append("ti='{}'".format(cc['ti']))
  explist.append("rti='{}'".format(cc['rti']))
  explist.append("LSM='{}'".format(cc['lsm']))
  explist.append("hidden='{}'".format(cc['hidden']))
  explist.append("pgepochs='{}'".format(cc['pgepochs']))
  if cc['lsm']:
    explist.append("readoutinp='{}'".format(cc['rinp']))
    explist.append("readoutout='{}'".format(cc['rout']))
    explist.append("minicol='{}'".format(cc['minicol']))
    explist.append("macrocol='{}'".format(cc['macrocol']))
    explist.append("SpecRAD='{}'".format(cc['specrad']))
    explist.append("PMAX='{}'".format(cc['pmax']))
    explist.append("ALPHA='{}'".format(cc['alpha']))
  exportline = ','.join(explist)
  print(ii, command.format(exportline))
  print('ppending', pending())
  while True:
    if pending() < 20:
      break
    time.sleep(2)
  time.sleep(3)
  _ = subprocess.Popen(command.format(exportline).split())