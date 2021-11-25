HID_SIZE = 128 # size of the hidden layer

class modelA2C(nn.Module):
    def __init__(self, obs_size, act_size):
        super(ModelA2C, self).__init__()
        # The network have 3 heads instead of one. Two are for the mean and sigma of gaussian distributions for the value of the action space. The other is the critic and represents the value of the state itself.
        self.base = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.ReLu(),
        )
        self.mu = nn.Sequential(
            nn.Linear(HID_SIZE, act_size),
            nn.Tanh(),
        )
        self.var = nn.Sequential(
            nn.Linear(HID_SIZE, act_size),
            nn.Softplus(),
        )
        self.value = nn.Linear(HID_SIZE, 1)