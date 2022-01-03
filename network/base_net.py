import torch
import torch.nn as nn
import torch.nn.functional as f


class RNN(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, args):
        super(RNN, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, obs, hidden_state):
        x = f.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h


# Critic of Central-V
class Critic(nn.Module):
    def __init__(self, input_shape, args):
        super(Critic, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.critic_dim)
        self.fc2 = nn.Linear(args.critic_dim, args.critic_dim)
        self.fc3 = nn.Linear(args.critic_dim, 1)

    def forward(self, inputs):
        x = f.relu(self.fc1(inputs))
        x = f.relu(self.fc2(x))
        q = self.fc3(x)
        return q


class IBFComm(nn.Module):
    def __init__(self, input_shape, args):
        super(IBFComm, self).__init__()
        self.args = args
        self.n_agents = args.n_agents

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, args.comm_embed_dim * self.n_agents)

        self.inference_model = nn.Sequential(
            nn.Linear(input_shape + args.comm_embed_dim * self.n_agents, 4 * args.comm_embed_dim * self.n_agents),
            nn.ReLU(True),
            nn.Linear(4 * args.comm_embed_dim * self.n_agents, 4 * args.comm_embed_dim * self.n_agents),
            nn.ReLU(True),
            nn.Linear(4 * args.comm_embed_dim * self.n_agents, args.n_actions)
        )

    def forward(self, inputs):
        x = f.relu(self.fc1(inputs))
        x = f.relu(self.fc2(x))
        gaussian_params = self.fc3(x)

        mu = gaussian_params
        #sigma = F.softplus(gaussian_params[:, self.args.comm_embed_dim * self.n_agents:])
        sigma = torch.ones(mu.shape).cuda()

        return mu, sigma




class Pagent(nn.Module):
    def __init__(self, input_shape, args):
        super(Pagent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        a = self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
        # 为每个episode中的每个agent都初始化一个eval_hidden、target_hidden
        # self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        return a

    def forward(self, obs, hidden_state):
        x = f.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        # q = torch.clamp(q,-20,2)
        q = torch.clamp(q, -5, 2)
        # q = torch.sigmoid(q)  # qmix_sac

        # q = 4* f.tanh(q)  # TODO
        # q = q.clone()
        return q, h

