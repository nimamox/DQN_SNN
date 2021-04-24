import torch
import os

device = 'cpu'

if device == 'gpu':
   if torch.cuda.is_available():
      device = torch.device("cuda")     
   else:
      device = torch.device("cpu")
else:
   device = torch.device("cpu")

print('Running on', device.type)

REGRESSOR = 'SurrGrad' #SNN, SNN_scaled, LSM, SurrGrad
ENCODER = 'ISI'

CONV_TYPE = 3

learning_rate = 0.05

tbs = 35
ti = 90
rti = 30

USE_LSM = True
minicol = [2, 2, 2]
macrocol = [2, 2, 2]
SpecRAD = False
PMAX = .1
ALPHA = .01

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
   RESULT_PATH = '.'