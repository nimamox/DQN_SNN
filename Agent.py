from config import *
import torch
import torch.optim as optim

from torch import nn
import torch.nn.functional as F


import NeuralEncBenchmark
from NeuralEncBenchmark.ttfs import TTFS_encoder
from NeuralEncBenchmark.isi import ISI_encoding
from NeuralEncBenchmark.multiplexing_ttfs import multiplexing_encoding_TTFS_phase
from NeuralEncBenchmark.multiplexing_isi import multiplexing_encoding_ISI_phase
from NeuralEncBenchmark.datasets import *

from NeuralEncBenchmark.torch_device import dtype
from NeuralEncBenchmark.sparse_data_generator import sparse_generator
from NeuralEncBenchmark.surrogate_encoder import encode_data

from NeuralEncBenchmark.surrogate_model import run_snn
from NeuralEncBenchmark.surrogate_train import init_model, compute_classification_accuracy, train

import sys
sys.path.append('PCRITICAL')

from modules.pcritical import PCritical
from modules.utils import OneToNLayer
from modules.topologies import SmallWorldTopology

ISI_external_cache = {}

class LIFNeuron(nn.Module):
   def __init__(self, dim_in, Rd, Cm, Rs, Vth, V_reset, dt):
      super().__init__()
      # self.batch_size = batch_size
      self.dim_in = dim_in
      self.rd = Rd
      self.cm = Cm
      self.rs = Rs
      self.vth = Vth
      self.v_reset = V_reset
      # self.v = torch.full([self.batch_size, self.dim_in], self.v_reset).to(device)
      self.dt = dt

      self.tau_in = 1/(self.cm*self.rs)
      self.tau_lk = 1/(self.cm)*(1/self.rd + 1/self.rs) 

   @staticmethod
   def soft_spike(x):
      a = 2.0
      return torch.sigmoid_(a*x)

   def spiking(self):
      if self.training == True:
         spike_hard = torch.gt(self.v, self.vth).float()
         spike_soft = self.soft_spike(self.v - self.vth)
         v_hard = self.v_reset*spike_hard + self.v*(1 - spike_hard)
         v_soft = self.v_reset*spike_soft + self.v*(1 - spike_soft)
         self.v = v_soft + (v_hard - v_soft).detach_()
         return spike_soft + (spike_hard - spike_soft).detach_()
      else:
         spike_hard = torch.gt(self.v, self.vth).float()
         self.v = self.v_reset*spike_hard + self.v*(1 - spike_hard)
         return spike_hard


   def forward(self, v_inject):
      'Upgrade membrane potention every time step by differantial equation.'
      # print(v_inject.shape)
      self.v += (self.tau_in*v_inject - self.tau_lk*self.v) * self.dt
      return self.spiking(), self.v

   def reset(self, batch_size):
      'Reset the membrane potential to reset voltage.'
      self.v = torch.full([batch_size, self.dim_in], self.v_reset).to(device)

class MAC_Crossbar(nn.Module):
   def __init__(self, dim_in, dim_out, W_std):
      super().__init__()
      self.dim_in = dim_in
      self.dim_out = dim_out
      self.weight = nn.Parameter(torch.zeros(dim_in, dim_out).to(device))
      torch.nn.init.normal_(self.weight, mean=0.0, std=W_std)

   def forward(self, input_vector):
      output = input_vector.mm(self.weight)
      return output

class Three_Layer_SNN(nn.Module):
   def __init__(self, param):
      super().__init__()
      self.linear1 = MAC_Crossbar(param['dim_in'], param['dim_h'], param['W_std1'])
      self.neuron1 = LIFNeuron(param['dim_h'], param['Rd'], param['Cm'],
                               param['Rs'], param['Vth'], param['V_reset'], param['dt'])
      self.linear2 = MAC_Crossbar(param['dim_h'], param['dim_out'], param['W_std2'])
      self.neuron2 = LIFNeuron(param['dim_out'], param['Rd'], param['Cm'], 
                               param['Rs'], param['Vth']*20, param['V_reset'], param['dt'])

   def forward(self, input_vector):
      out_vector = self.linear1(input_vector)
      # debug print, very useful to see what happend in every layer
      #print('0', out_vector.max())
      # out_vector = self.BatchNorm1(out_vector)
      #print('1', out_vector.max())
      out_vector, _ = self.neuron1(out_vector)
      #print('2', out_vector.sum(1).max())
      out_vector = self.linear2(out_vector)
      #print('3', out_vector.max())
      # out_vector = self.BatchNorm2(out_vector)
      #print('4', out_vector.max())
      out_vector, out_v = self.neuron2(out_vector)
      #print('5', out_vector.sum(1).max())
      return out_vector, out_v

   def reset_(self, batch_size):
      '''
      Reset all neurons after one forward pass,
      to ensure the independency of every input image.
      '''
      for item in self.modules():
         if hasattr(item, 'reset'):
            item.reset(batch_size)

class LinReg(nn.Module):
   def __init__(self, inputSize, outputSize):
      nn.Module.__init__(self)
      self.linear = nn.Linear(inputSize, outputSize)
   def forward(self, x):
      out = self.linear(x)
      return out


class MLP(nn.Module):
   def __init__(self, inputSize, outputSize, h=10):
      nn.Module.__init__(self)
      self.linear1 = nn.Linear(inputSize, h)
      self.linear2 = nn.Linear(h, outputSize)
   def forward(self, x):
      out = self.linear1(x)
      out = F.relu(out)
      out = self.linear2(out)
      return out

def Poisson_encoding(x):
   out_spike = torch.rand_like(x).le(x).float()
   return out_spike

def Poisson_encoder(x, T_sim):
   out_spike = torch.zeros([x.shape[0], x.shape[1], T_sim])
   for t in range(T_sim):
      out_spike[:,:,t] = Poisson_encoding(x)
   return out_spike.to(device)


class InputLayer(nn.Module):
   def __init__(self, N, dim_input, dim_output, weight=100.0):
      super().__init__()

      pre = np.arange(dim_input * N) % dim_input
      post = (
         np.random.permutation(max(dim_input, dim_output) * N)[: dim_input * N]
           % dim_output
      )
      i = torch.LongTensor(np.vstack((pre, post)))
      v = torch.ones(dim_input * N) * weight

      # Save the transpose version of W for sparse-dense multiplication
      self.W_t = torch.sparse.FloatTensor(
         i, v, torch.Size((dim_input, dim_output))
           ).t()

   def forward(self, x):
      return self.W_t.mm(x.t()).t()

   def _apply(self, fn):
      super()._apply(fn)
      self.W_t = fn(self.W_t)
      return self

class ReadoutLayer(nn.Module):
   def __init__(self, reservoir_size, dim_input, dim_output):
      super().__init__()
      self.pre = np.random.permutation(np.arange(reservoir_size))[:dim_input]
      self.post = np.arange(dim_input) % dim_output
      i = torch.LongTensor(np.vstack((self.pre, self.post)))
      v = torch.ones(i.shape[1])
      self.W_t = torch.sparse.FloatTensor(
         i, v, torch.Size((reservoir_size, dim_output))
           ).t()
   def forward(self, x):
      res = self.W_t.mm(x.t()).t()
      res[res > .5] = 1
      return res

   def _apply(self, fn):
      super()._apply(fn)
      self.W_t = fn(self.W_t)
      return self


class DQN_SANDBOX():
   def __init__(self, 
                regressor,
                 agent_id,
               n_actions,
                 n_features,
                 memory_size,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 lr=0.01
                 ):
      self.regressor = regressor
      self.agent_id = agent_id
      self.n_actions = n_actions
      self.n_features = n_features
      #self.n_layers = n_layers
      self.gamma = reward_decay
      self.memory_size = memory_size
      self.batch_size = memory_size
      self.epsilon = e_greedy 
      self.scale_max = None

      # total learning step
      self.learn_step_counter = 0

      # initialize learning rate
      self.lr = lr

      # initialize zero memory [s, a, r, s_]
      if regressor not in ['LinReg', 'MLP']:
         self.memory = np.zeros((self.memory_size, 2 + 2))
      else:
         self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
      self.whole_memory = []

      # build net
      self.criterion = nn.MSELoss() 
      self.optimizer = None
      self._build_net()

      self.cost_his = []


   def _build_net(self):
      if self.agent_id == 0:
         print('Regressor:', self.regressor)
         print('CONV_TYPE:', CONV_TYPE)
         print('USE_LSM:', USE_LSM)
         print('Learning Rate:', self.lr)

      if CONV_TYPE == 1:
         self.convert_state_scaled = self.convert_state_scaled_1
      elif CONV_TYPE == 2:
         self.convert_state_scaled = self.convert_state_scaled_2
      elif CONV_TYPE == 3:
         self.convert_state_scaled = self.convert_state_scaled_3
      else:
         raise Exception('Invalid conversion')

      if USE_LSM:
         topology = SmallWorldTopology(
            SmallWorldTopology.Configuration(
                  minicolumn_shape=minicol,
                  macrocolumn_shape=macrocol,
                  p_max=PMAX,
                  # minicolumn_spacing=1460,
                  # intracolumnar_sparseness=635.0,
                  # neuron_spacing=40.0,
                  spectral_radius_norm=SpecRAD,
                  inhibitory_init_weight_range=(0.1, 0.3),
                  excitatory_init_weight_range=(0.2, 0.5),
               )
         )
         lsm_N = topology.number_of_nodes()
         N_inputs = 5
         if CONV_TYPE == 3:
            N_inputs = 6
         self.reservoir = PCritical(1, topology, alpha=ALPHA).to(device)
         #self.lsm = torch.nn.Sequential(OneToNLayer(1, N_inputs, lsm_N), self.reservoir).to(device)
         self.lsm = torch.nn.Sequential(InputLayer(1, N_inputs, lsm_N),
                                        self.reservoir,
                                        ReadoutLayer(lsm_N, readout_inp, readout_out)
                                        ).to(device)


      if self.regressor == 'LinReg':
         self.eval_net =  LinReg(self.n_features, self.n_actions)
         self.optimizer = optim.Adam(self.eval_net.parameters(), lr=self.lr)
         self.target_net = LinReg(self.n_features, self.n_actions)
         self.target_net.load_state_dict(self.eval_net.state_dict())
         self.target_net.eval()
      elif self.regressor == 'MLP':
         hid = 20
         self.eval_net =  MLP(self.n_features, self.n_actions, hid)
         self.optimizer = optim.Adam(self.eval_net.parameters(), lr=self.lr)
         self.target_net = MLP(self.n_features, self.n_actions, hid)
         self.target_net.load_state_dict(self.eval_net.state_dict())
         self.target_net.eval()
      elif self.regressor == 'SurrGrad':
         self.snn_params = {}
         if USE_LSM:
            self.snn_params['dim_in'] = readout_out
         else:
            self.snn_params['dim_in'] = 5
            if CONV_TYPE == 3:
               self.snn_params['dim_in'] = 6

         self.snn_params['T_sim'] = 10
         self.eval_net, self.surr_alpha, self.surr_beta = init_model(self.snn_params['dim_in'], hidden, 8, .05)
         self.target_net = []
         for vv in self.eval_net:
            self.target_net.append(vv.clone())
         self.optimizer = optim.Adam(self.eval_net, lr=self.lr, betas=(0.9, 0.999)) #TODO: learning rate
         self.all_obs_spikes = []

      elif self.regressor.startswith('SNN'):
         self.snn_params = {
            'seed': 1337,

               'Rd': 5.0e3,    # this device resistance is mannually set for smaller leaky current?
          'Cm': 3.0e-6,   # real capacitance is absolutely larger than this value
        'Rs': 1.0,      # this series resistance value is mannually set for larger inject current?

        'Vth': 0.8,     # this is the real device threshould voltage
        'V_reset': 0.0, 

        'dt': 1.0e-6,   # every time step is dt, in the one-order differential equation of neuron
        'T_sim': 10,   # could control total spike number collected
        'dim_in': 5,
        'dim_h': hidden,
        'dim_out': 8,
        'epoch': 10,

        'W_std1': 1.0,
        'W_std2': 1.0,
         }

         if USE_LSM:
            self.snn_params['dim_in'] = readout_out
         else:
            self.snn_params['dim_in'] = 5
            if CONV_TYPE == 3:
               self.snn_params['dim_in'] = 6

         self.eval_net =  Three_Layer_SNN(self.snn_params)
         self.optimizer = optim.Adam(self.eval_net.parameters(), lr=self.lr)
         self.target_net =  Three_Layer_SNN(self.snn_params)
         self.target_net.load_state_dict(self.eval_net.state_dict())
         self.target_net.eval()
         self.all_obs_spikes = []
      else:
         raise Exception('Invalid regressor')

   def store_transition(self, s, a, r, s_):
      if not hasattr(self, 'memory_counter'):
         self.memory_counter = 0

      transition = np.hstack((s, [a, r], s_))

      # replace the old memory with new memory
      index = self.memory_counter % self.memory_size
      self.memory[index, :] = transition

      self.memory_counter += 1    

   def spike_encoder(self, observation, step=None):
      if USE_LSM:
         observation = observation[np.newaxis, :]
         observation = self.convert_state_scaled(observation)
         obs_pois = self.SEncoding(observation)
         obs_spikes = []
         for t in range(self.snn_params['T_sim']):
            S = self.lsm(obs_pois[:,:,t])
            obs_spikes.append(S)
         self.obs_spikes = torch.stack(obs_spikes, dim=2)
         obs_spikes_reshaped = self.obs_spikes.detach().reshape(self.obs_spikes.shape[1], 
                                                                self.obs_spikes.shape[2])
         if self.regressor == 'SurrGrad':
            self.obs_spikes = torch.einsum('ijk->ikj', self.obs_spikes)
            self.all_obs_spikes.append(torch.einsum('ij->ji', obs_spikes_reshaped))
         else:
            self.all_obs_spikes.append(obs_spikes_reshaped)
      else:
         observation = observation[np.newaxis, :]
         observation = self.convert_state_scaled(observation)
         self.obs_spikes = self.SEncoding(observation)
         obs_spikes_reshaped = self.obs_spikes.detach().reshape(self.obs_spikes.shape[1], 
                                                                self.obs_spikes.shape[2])
         if self.regressor == 'SurrGrad':
            self.obs_spikes = torch.einsum('ijk->ikj', self.obs_spikes)
            self.all_obs_spikes.append(torch.einsum('ij->ji', obs_spikes_reshaped))
         else:
            self.all_obs_spikes.append(obs_spikes_reshaped)


   def choose_action(self, observation):
      # to have batch dimension when feed into tf placeholder
      observation = observation[np.newaxis, :]

      if np.random.uniform() < self.epsilon:
         # forward feed the observation and get q value for every actions
         if self.regressor in ['LinReg', 'MLP']:
            actions_value = self.eval_net(torch.Tensor(observation))
            action = actions_value.max(1)[1][0].detach()
         elif self.regressor == 'SurrGrad':
            with torch.no_grad():
               result = self.run_surr_grad_snn(self.eval_net, self.obs_spikes)
               action = result.max(1)[1][0].detach()
         elif self.regressor.startswith('SNN'):
            with torch.no_grad():
               self.eval_net.eval()
               result = self.run_ncomm_snn(self.eval_net, self.obs_spikes)
               action = result.max(1)[1][0].detach()
         else:
            observation = self.convert_state_scaled(observation)
            raise Exception('Invalid regressor')
         #actions_value = self.eval_net.predict(observation)
         #actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
         #action = np.argmax(actions_value)
      else:
         action = np.random.randint(0, self.n_actions)
      return action

   def convert_state_scaled_1(self, observation):
      normalizer_max = 2.06
      new_obs = observation.copy()
      new_obs[:,0] = new_obs[:,0] / normalizer_max
      new_obs[:,1:] = new_obs[:,1:] 
      return new_obs

   def convert_state_scaled_2(self, observation):
      normalizer_max = 2.06
      ch = np.argmax(observation[:,1:], 1)
      new_obs = np.zeros([observation.shape[0], observation.shape[1]])
      # new_obs[np.arange(new_obs.shape[0]), ch+1] = observation[:,0] / self.scale_max[ch]
      new_obs[np.arange(new_obs.shape[0]), ch+1] = observation[:,0] / (self.scale_max[ch])
      new_obs[:,0] = observation[:,0] / normalizer_max
      return new_obs

   def convert_state_scaled_3(self, observation):
      normalizer_max = 2.06
      ch = np.argmax(observation[:,1:], 1)
      new_obs = np.zeros([observation.shape[0], observation.shape[1]+1])
      new_obs[:,0] = observation[:,0] / normalizer_max
      new_obs[:,1] = observation[:,0] / (self.scale_max[ch])
      new_obs[:,2:] = observation[:,1:] 
      return new_obs

   def SEncoding(self, X):
      if ENCODER == 'Poisson':
         return Poisson_encoder(torch.Tensor(X), self.snn_params['T_sim'])
      elif ENCODER == 'ISI':
         ed = encode_data(X, X, nb_units=X.shape[1], encoder_type="ISI_inverse", batch_size=X.shape[0], nb_steps=10, TMAX=10, external_ISI_cache=ISI_external_cache)
         ft = next(sparse_generator(ed, shuffle=False))[0]
         return torch.einsum('ijk->ikj', ft.to_dense())
      else:
         raise Exception('Invalid Encoding')

class DQN_SANDBOX(DQN_SANDBOX):
   def run_ncomm_snn(self, network, inp_spike):
      network.reset_(inp_spike.shape[0])
      out_vs = []
      # inp_spike = Poisson_encoder(torch.Tensor(X), self.snn_params['T_sim'])
      for t in range(self.snn_params['T_sim']):
         out_spike, out_v = network(inp_spike[:,:,t])
         out_vs.append(out_v)
      out_vs = torch.stack(out_vs, dim=2)
      return out_vs.sum(dim=2)

   def run_surr_grad_snn(self, network, inp_spike):
      surr_out, _ = run_snn(inp_spike, inp_spike.shape[0], 
                            self.snn_params['T_sim'], network, self.surr_alpha, self.surr_beta)
      return surr_out.sum(1)

   def learn_snn(self, episode_size, step, 
                 training_batch_size = 50, training_iteration = 100, replace_target_iter = 25,
                  debug=False, seed=1337, opt_cb=None):
      method = 'double'
      sequential = False
      nForget = 50

      spike_inp = torch.stack(self.all_obs_spikes, dim=0)
      episode_size = spike_inp.shape[0]-1

      if debug:
         print('DEBUG tmp!')
         set_seed(seed)
         if self.regressor == 'SurrGrad':
            self.eval_net[0] = self.eval_net[0].clone().detach().requires_grad_()
            self.eval_net[1] = self.eval_net[1].clone().detach().requires_grad_()
            self.target_net[0] = self.target_net[0].clone().detach()
            self.target_net[1] = self.target_net[1].clone().detach()

            eval_net_copy = [None, None]
            eval_net_copy[0] = self.eval_net[0].clone().detach().requires_grad_()
            eval_net_copy[1] = self.eval_net[1].clone().detach().requires_grad_()
            target_net_copy = [None, None]
            target_net_copy[0] = self.target_net[0].clone().detach()
            target_net_copy[1] = self.target_net[1].clone().detach()
         else:
            eval_net_copy = self.eval_net.state_dict()
            target_net_copy = self.target_net.state_dict()
         if opt_cb:
            opt_cb(self)

      if (step == episode_size - 1):
         #Drop first nForget episodes
         index_train = np.arange(nForget, episode_size)
      else:
         index_train = np.arange(0, episode_size)

      losses = []
      # print('shape', episode_size, self.memory.shape, training_batch_size, spike_inp.shape)
      # import pdb; pdb.set_trace()
      for i in range(training_iteration+1):
         if i == training_iteration:
            minibatch = self.memory
         else:
            np.random.shuffle(index_train)
            minibatch = self.memory[index_train[:training_batch_size]]
         # print('i', i, minibatch.shape)

         with torch.no_grad():
            if self.regressor == 'SurrGrad':
               q_eval = self.run_surr_grad_snn(self.eval_net, spike_inp[minibatch[:, 0],:,:])
               q_next = self.run_surr_grad_snn(self.target_net, spike_inp[minibatch[:, -1],:,:]) #TODO: 0
            elif self.regressor == 'SNN':
               # raise Exception('ABABAB')
               q_eval = self.run_ncomm_snn(self.eval_net, spike_inp[minibatch[:, 0],:,:])
               q_next = self.run_ncomm_snn(self.target_net, spike_inp[minibatch[:, -1],:,:]) #TODO: 0
            else: 
               raise Exception('Invalid regressor')

            if (method == 'double'):
               if self.regressor == 'SurrGrad':
                  q_next_action = self.run_surr_grad_snn(self.eval_net, spike_inp[minibatch[:, -1],:,:])
               elif self.regressor == 'SNN':
                  q_next_action = self.run_ncomm_snn(self.eval_net, spike_inp[minibatch[:, -1],:,:])
               next_action = q_next_action.max(1)[1].detach()

         # change q_target w.r.t q_eval's action
         q_target = q_eval.detach()

         eval_act_index = minibatch[:, 1].astype(int)
         reward = minibatch[:, 2]

         if (method == 'normal'):
            next_q_value = self.gamma * q_next.max(1)[0].detach()
            for index in range(len(eval_act_index)):
               q_target[index, eval_act_index[index]] = reward[index] + next_q_value[index]

         elif (method == 'double'):
            for index in range(len(eval_act_index)):
               q_target[index, eval_act_index[index]] = reward[index] + \
                  self.gamma * q_next[index, next_action[index]]
         self.optimizer.zero_grad()

         if i == training_iteration:
            torch.set_grad_enabled(False)
         if self.regressor == 'SurrGrad':
            outputs = self.run_surr_grad_snn(self.eval_net, spike_inp[minibatch[:, 0],:,:])
         elif self.regressor == 'SNN':
            outputs = self.run_ncomm_snn(self.eval_net, spike_inp[minibatch[:, 0],:,:])
         loss = self.criterion(outputs, q_target)
         losses.append(loss.detach().item())
         if i == training_iteration:
            torch.set_grad_enabled(True)
            last_loss = loss.detach().item()
         else:
            loss.backward()
         if debug == 2:
            import pdb; pdb.set_trace()
         self.optimizer.step()

         if ((i+1) % replace_target_iter == 0):
            if debug:
               print('replace target', i)
            if self.regressor.startswith('SurrGrad'):
               self.target_net[0] = self.eval_net[0].detach()
               self.target_net[1] = self.eval_net[1].detach()
            else:
               self.target_net.load_state_dict(self.eval_net.state_dict())

      if not debug:
         self.all_obs_spikes = [spike_inp[-1,:,:]]
         self.cost_his.append(last_loss)

      if debug:
         if self.regressor == 'SurrGrad':
            self.eval_net[0] = eval_net_copy[0]
            self.eval_net[1] = eval_net_copy[1]
            self.target_net[0] = target_net_copy[0]
            self.target_net[1] = target_net_copy[1]
         else:
            self.eval_net.load_state_dict(eval_net_copy)
            self.target_net.load_state_dict(target_net_copy)
      return last_loss

   def update_lr(self, lr):
      if self.agent_id == 0:
         print('* New LR:', lr)
      if self.regressor.startswith('SurrGrad'):
         self.optimizer = optim.Adam(self.eval_net, lr=lr, betas=(0.9, 0.999))
      else:
         self.optimizer = optim.Adam(self.eval_net.parameters(), lr=lr)
      #self.eval_net.lr = lr
      #self.target_net.lr = lr    