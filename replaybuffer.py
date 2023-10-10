import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, args):
        self.s = np.zeros((args.batch_size, args.state_dim))
        self.a = np.zeros((args.batch_size, 1))
        self.a_logprob = np.zeros((args.batch_size, 1))
        self.r = np.zeros((args.batch_size, 1))
        self.s_ = np.zeros((args.batch_size, args.state_dim))
        self.dw = np.zeros((args.batch_size, 1))
        self.done = np.zeros((args.batch_size, 1))
        self.count = 0
        self.device = torch.device(args.device)

    def store(self, s, a, a_logprob, r, s_, dw, done):
        self.s[self.count] = s.cpu().detach().numpy()
        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r
        self.s_[self.count] = s_.cpu().detach().numpy()
        self.dw[self.count] = dw
        self.done[self.count] = done
        self.count += 1

    def numpy_to_tensor(self):
        s = torch.tensor(self.s, dtype=torch.float, device=self.device)
        a = torch.tensor(self.a, dtype=torch.long, device=self.device)  # In discrete action space, 'a' needs to be torch.long
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float, device=self.device)
        r = torch.tensor(self.r, dtype=torch.float, device=self.device)
        s_ = torch.tensor(self.s_, dtype=torch.float, device=self.device)
        dw = torch.tensor(self.dw, dtype=torch.float, device=self.device)
        done = torch.tensor(self.done, dtype=torch.float, device=self.device)

        return s, a, a_logprob, r, s_, dw, done
