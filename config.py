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

REGRESSOR = os.getenv('REGRESSOR', 'SurrGrad').replace("'","") #SNN, SNN_scaled, LSM, SurrGrad
ENCODER = os.getenv('ENCODER', 'ISI').replace("'","")

CONV_TYPE = int(os.getenv('CONV', '3').replace("'",""))

learning_rate = float(os.getenv('LR', '0.05').replace("'",""))

tbs = int(os.getenv('tbs', '35').replace("'",""))
ti = int(os.getenv('ti', '90').replace("'",""))
rti = int(os.getenv('rti', '30').replace("'",""))

USE_LSM = bool(int(os.getenv('LSM', '1').replace("'","")))
minicol = os.getenv('minicol', '2-2-2').replace("'","")
minicol = list(map(int, minicol.split('-')))

macrocol = os.getenv('macrocol', '2-2-2').replace("'","") 
macrocol = list(map(int, macrocol.split('-')))

SpecRAD = bool(int(os.getenv('SpecRAD', '0').replace("'","")))
PMAX = float(os.getenv('PMAX', '0.1').replace("'",""))
ALPHA = float(os.getenv('ALPHA', '0.01').replace("'",""))

print('REGRESSOR', REGRESSOR, 'ENCODER', ENCODER, 'CONV_TYPE', CONV_TYPE, 'learning_rate',
      learning_rate, 'tbs', tbs, 'ti', ti, 'rti', rti, 'USE_LSM',
      USE_LSM, 'minicol', minicol, 'macrocol', macrocol, 'SpecRAD', SpecRAD,
      'PMAX', PMAX, 'ALPHA', ALPHA)

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
