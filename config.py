import torch
import os, sys

n_channel = 4
n_su = 6

device = os.getenv('device', 'gpu')

if device == 'gpu':
   if torch.cuda.is_available():
      device = torch.device("cuda")     
   else:
      device = torch.device("cpu")
else:
   device = torch.device("cpu")

print('Running on', device.type)

def get_slurm_env(type_, name, default):
   res = os.getenv(name, str(default)).replace("'","")
   if type_ == 'int':
      res = int(res)
   elif type_ == 'float':
      res = float(res)
   elif type_ == 'bool':
      res = bool(res)
   return res

REGRESSOR = get_slurm_env('str', 'REGRESSOR', 'SurrGrad') #SNN, SNN_scaled, LSM, SurrGrad
ENCODER = get_slurm_env('str', 'ENCODER', 'ISI')

CONV_TYPE = get_slurm_env('int', 'CONV', 3)

learning_rate = get_slurm_env('float', 'LR', '0.05')

tbs = get_slurm_env('int', 'tbs', 35)
ti = get_slurm_env('int', 'ti', 90)
rti = get_slurm_env('int', 'rti', 30)

hidden = get_slurm_env('int', 'hidden', 10)

USE_LSM = get_slurm_env('bool', 'LSM', 1)

minicol = get_slurm_env('str', 'minicol', '2-2-2')
minicol = list(map(int, minicol.split('-')))

macrocol = get_slurm_env('str', 'macrocol', '2-2-2')
macrocol = list(map(int, macrocol.split('-')))

SpecRAD = get_slurm_env('bool', 'SpecRAD', 0)
PMAX = get_slurm_env('float', 'PMAX', '0.1')
ALPHA = get_slurm_env('float', 'ALPHA', '0.01')

readout_inp = get_slurm_env('int', 'readoutinp', '32')
readout_out = get_slurm_env('int', 'readoutout', '16')

print('REGRESSOR', REGRESSOR, 'ENCODER', ENCODER, 'CONV_TYPE', CONV_TYPE, 'learning_rate',
      learning_rate, 'tbs', tbs, 'ti', ti, 'rti', rti, 'USE_LSM',
      USE_LSM, 'minicol', minicol, 'macrocol', macrocol, 'SpecRAD', SpecRAD,
      'PMAX', PMAX, 'ALPHA', ALPHA, 'READ_INP', readout_inp,
      'READ_OUT', readout_out, 'HIDDEN', hidden)

fname_list = []
fname_list.append(REGRESSOR)
fname_list.append(ENCODER)
fname_list.append('CONV{}'.format(CONV_TYPE))
fname_list.append('LR{:.4f}'.format(learning_rate))
fname_list.append('tbs{}'.format(tbs))
fname_list.append('ti{}'.format(ti))
fname_list.append('rti{}'.format(rti))
if USE_LSM:
   ll = []
   ll.append('LSM')
   ll.append('SR{:d}'.format(SpecRAD))
   ll.append('PMAX{:.3f}'.format(PMAX))
   ll.append('Mini{}'.format('_'.join([str(_) for _ in minicol])))
   ll.append('Macro{}'.format('_'.join([str(_) for _ in macrocol])))
   ll.append('Alp{:.4f}'.format(ALPHA))
   ll.append('lsminp{}'.format(readout_inp))
   ll.append('lsmout{}'.format(readout_out))
   ll.append('hidden{}'.format(hidden))
   fname_list.append('--'.join(ll))
else:
   fname_list.append('NoLSM')

FNAME = '_-_'.join(fname_list)
print(FNAME)

RESULT_PATH = '/content/drive/MyDrive/TNNLS/LAST/'
if not os.path.exists(RESULT_PATH):
   RESULT_PATH = './results/'
   
if os.path.exists(os.path.join(RESULT_PATH, '{}_PU{}_SU{}.hkl'.format(FNAME, n_channel, n_su))):
   print('Already had executed this configuration. Exiting...')
   sys.exit(1)
   
with open(os.path.join(RESULT_PATH, '{}_PU{}_SU{}.hkl'.format(FNAME, n_channel, n_su)), 'wb') as fo:
   fo.write('X'.encode("ascii"))
