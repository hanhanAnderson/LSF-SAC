import torch
import os
from network.base_net import RNN, Pagent, IBFComm
from network.qmix_net import QMixNet


class QMIX:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.comm_embed_dim = args.comm_embed_dim
        input_shape = self.obs_shape
        # 根据参数决定RNN的输入维度
        if args.last_action:
            input_shape += self.n_actions
        if args.reuse_network:
            input_shape += self.n_agents
        self.log_alpha = torch.zeros(1, dtype=torch.float32)  # , requires_grad=True)
        # add comm info
        self.s_mu = torch.zeros(1)
        self.s_sigma = torch.ones(1)

        ms_shape = self.comm_embed_dim * self.n_agents
        # 神经网络
        self.eval_rnn = RNN(input_shape + ms_shape, args)  # 每个agent选动作的网络
        self.target_rnn = RNN(input_shape + ms_shape, args)

        self.comm = IBFComm(input_shape, args)

        self.eval_qmix_net = QMixNet(args)  # 把agentsQ值加起来的网络
        self.target_qmix_net = QMixNet(args)

        self.eval_pagent = Pagent(input_shape, args)
        # self.target_pagent = Pagent(input_shape, args)

        self.args = args
        if self.args.cuda:
            self.comm.cuda()
            self.eval_rnn.cuda()
            self.target_rnn.cuda()
            self.eval_qmix_net.cuda()
            self.target_qmix_net.cuda()
            self.eval_pagent.cuda()
            # self.target_pagent.cuda()
            self.log_alpha = self.log_alpha.cuda()

        self.log_alpha.requires_grad=True
        self.alpha = self.args.alpha
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)

        self.model_dir = args.model_dir + '/' + args.alg + '/' + args.map
        # 如果存在模型则加载模型
        if self.args.load_model:
            if os.path.exists(self.model_dir + '/rnn_net_params.pkl'):
                path_rnn = self.model_dir + '/rnn_net_params.pkl'
                path_qmix = self.model_dir + '/qmix_net_params.pkl'
                path_pagent = self.model_dir + '/pagent_net_params.pkl'
                path_comm = self.model_dir + '/comm_net_params.pkl'
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.eval_rnn.load_state_dict(torch.load(path_rnn, map_location=map_location))
                self.eval_qmix_net.load_state_dict(torch.load(path_qmix, map_location=map_location))
                self.eval_pagent.load_state_dict(torch.load(path_pagent, map_location=map_location))
                self.comm.load_state_dict(torch.load(path_comm, map_location=map_location))
                # self.log_alpha = torch.load(self.model_dir + '/' + '_log_alpha.pkl',map_location=map_location)
                # self.alpha = self.log_alpha.exp()
                print('Successfully load the model: {} and {}'.format(path_rnn, path_qmix))
            else:
                raise Exception("No model!")

        # 让target_net和eval_net的网络参数相同
        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())
        # self.target_pagent.load_state_dict(self.eval_pagent.state_dict())

        self.eval_parameters =  list(self.eval_rnn.parameters()) + list(self.comm.parameters()) + list(self.eval_qmix_net.parameters())
        if args.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)

        self.peval_parameters = self.eval_pagent.parameters()
        self.policy_optimiser = torch.optim.RMSprop(self.peval_parameters, lr=args.lr)

        # 执行过程中，要为每个agent都维护一个eval_hidden
        # 学习过程中，要为每个episode的每个agent都维护一个eval_hidden、target_hidden
        self.eval_hidden = None
        self.target_hidden = None
        self.peval_hidden = None
        # self.ptarget_hidden = None
        print('Init alg QMIX')

    def learn(self, batch, max_episode_len, train_step, epsilon=None):  # train_step表示是第几次学习，用来控制更新target_net网络的参数
        '''
        在learn的时候，抽取到的数据是四维的，四个维度分别为 1——第几个episode 2——episode中第几个transition
        3——第几个agent的数据 4——具体obs维度。因为在选动作时不仅需要输入当前的inputs，还要给神经网络输入hidden_state，
        hidden_state和之前的经验相关，因此就不能随机抽取经验进行学习。所以这里一次抽取多个episode，然后一次给神经网络
        传入每个episode的同一个位置的transition
        '''
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():  # 把batch里的数据转化成tensor
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        s, s_next, u, r, avail_u, avail_u_next, terminated = batch['s'], batch['s_next'], batch['u'], \
                                                             batch['r'], batch['avail_u'], batch['avail_u_next'], \
                                                             batch['terminated']
        mask = 1 - batch["padded"].float()  # 用来把那些填充的经验的TD-error置0，从而不让它们影响到学习
        r = 10. * (r - r.mean()) / (r.std() + 1e-6)  # might not need norm
        if self.args.cuda:
            s = s.cuda()
            u = u.cuda()
            r = r.cuda()
            s_next = s_next.cuda()
            terminated = terminated.cuda()
            mask = mask.cuda()
            self.s_mu = self.s_mu.cuda()
            self.s_sigma = self.s_sigma.cuda()

        actions_prob = []
        actions_logprobs = []
        episode_num = batch['o'].shape[0]
        q_evals, q_targets = [], []
        p_evals = []

        mac_out = []
        mu_out = []
        sigma_out = []
        logits_out = []
        m_sample_out = []
        g_out = []
        # 得到每个agent对应的Q值，维度为(episode个数, max_episode_len， n_agents， n_actions)
        episode_num = batch['o'].shape[0]
        # FUNC BEGIN
        q_evals, q_targets = [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_inputs(batch, transition_idx)  # 给obs加last_action、agent_id
            if self.args.cuda:
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
                self.target_hidden = self.target_hidden.cuda()
                # self.ptarget_hidden = self.ptarget_hidden.cuda()
                self.peval_hidden = self.peval_hidden.cuda()

            ###################COMM INFO #########################EVAL############
            mu, sigma, logits, m_sample = None, None, None, None
            # (mu, sigma), messages, m_sample = self._test_communicate(ep_batch.batch_size, agent_inputs, thres=thres, prob=prob)
            mu, sigma = self.comm(inputs) #inputs :(3,42) n_agents, input_shape, out: (3,9)
            normal_distribution = torch.distributions.Normal(mu, sigma)
            m_sample = normal_distribution.rsample()  #shape: (3,9), n_agents, comm_dimen*n_agents
            bs = episode_num
            message = m_sample.clone().view(bs, self.n_agents, self.n_agents, -1) #shape (1,3,3,3)
            message = message.permute(0, 2, 1, 3).contiguous().view(bs * self.n_agents, -1) #shape (3,9)
            agent_inputs = torch.cat([inputs, message], dim=1)
            logits = self._logits(episode_num, agent_inputs)

            """
            m_sample
            tensor([[-0.1998,  0.2178,  0.0718,  1.4496, -0.3582, -0.6019, -1.8840,  1.4017,
                     -0.1017],
                    [-0.1682, -0.2860,  0.0284,  1.0920, -0.4180,  0.8369,  1.5309, -0.6954,
                      0.2244],
                    [-0.3476, -0.3097,  0.6931,  0.5904,  1.0163, -1.0165, -0.1685,  0.8008,
                      0.2508]], device='cuda:0', grad_fn=<AddBackward0>)
            
            message_before_permute
            tensor([[[[-0.1998,  0.2178,  0.0718],
                      [ 1.4496, -0.3582, -0.6019],
                      [-1.8840,  1.4017, -0.1017]],
            
                     [[-0.1682, -0.2860,  0.0284],
                      [ 1.0920, -0.4180,  0.8369],
                      [ 1.5309, -0.6954,  0.2244]],
            
                     [[-0.3476, -0.3097,  0.6931],
                      [ 0.5904,  1.0163, -1.0165],
                      [-0.1685,  0.8008,  0.2508]]]], device='cuda:0',
                   grad_fn=<ViewBackward>)
            
            AfterPermute:
            tensor([[-0.1998,  0.2178,  0.0718, -0.1682, -0.2860,  0.0284, -0.3476, -0.3097,
                      0.6931],
                    [ 1.4496, -0.3582, -0.6019,  1.0920, -0.4180,  0.8369,  0.5904,  1.0163,
                     -1.0165],
                    [-1.8840,  1.4017, -0.1017,  1.5309, -0.6954,  0.2244, -0.1685,  0.8008,
                      0.2508]], device='cuda:0', grad_fn=<ViewBackward>)
            """

            q_eval, self.eval_hidden = self.eval_rnn(agent_inputs, self.eval_hidden)
            # q_eval

            # q_eval, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)  # inputs维度为(40,96)，得到的q_eval维度为(40,n_actions)

            # 把q_eval维度重新变回(8, 5,n_actions)
            q_eval = q_eval.view(episode_num, self.n_agents, -1)

            mu = mu.view(episode_num, self.n_agents, -1)
            sigma = sigma.view(episode_num, self.n_agents, -1)
            logits = logits.view(episode_num, self.n_agents, -1)
            m_sample = m_sample.view(episode_num, self.n_agents, -1)

            q_evals.append(q_eval)


            mu_out.append(mu)
            sigma_out.append(sigma)
            logits_out.append(logits)
            m_sample_out.append(m_sample)

            ###################COMM INFO #########################TARGET##############
            mu_n, sigma_n, logits_n, m_sample_n = None, None, None, None
            # (mu_n, sigma), messages, m_sample = self._test_communicate(ep_batch.batch_size, agent_inputs, thres=thres, prob=prob)
            mu_n, sigma_n = self.comm(inputs_next)
            normal_distribution_n = torch.distributions.Normal(mu_n, sigma_n)
            ms_n = normal_distribution_n.rsample()
            bs = episode_num

            message_n = ms_n.clone().view(bs, self.n_agents, self.n_agents, -1)
            message_n = message_n.permute(0, 2, 1, 3).contiguous().view(bs * self.n_agents, -1)
            agent_inputs_n = torch.cat([inputs_next, message_n], dim=1)
            q_target, self.target_hidden = self.target_rnn(agent_inputs_n, self.target_hidden)

            q_target = q_target.view(episode_num, self.n_agents, -1)
            #
            # mu_n = mu_n.view(episode_num, self.n_agents, -1)
            # sigma_n = sigma_n.view(episode_num, self.n_agents, -1)
            # logits_n = logits_n.view(episode_num, self.n_agents, -1)
            # m_sample_n = m_sample_n.view(episode_num, self.n_agents, -1)

            q_targets.append(q_target)
            #########################################################################################################
            ###
            agent_outs, self.peval_hidden = self.eval_pagent(inputs, self.peval_hidden)
            avail_actions_ = avail_u[:, transition_idx]
            reshaped_avail_actions = avail_actions_.reshape(episode_num * self.n_agents, -1)
            agent_outs[reshaped_avail_actions == 0] = -1e11
            agent_outs = torch.softmax(agent_outs / 1, dim=1)
            agent_outs = agent_outs.view(episode_num, self.n_agents, -1)
            actions_prob.append(agent_outs)
            z = agent_outs == 0.0
            z = z.float() * 1e-8
            actions_logprobs.append(torch.log(agent_outs + z))
            ####
        # 得的q_eval和q_target是一个列表，列表里装着max_episode_len个数组，数组的的维度是(episode个数, n_agents，n_actions)
        # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组
        mu_out = torch.stack(mu_out, dim=1)  # Concat over time
        sigma_out = torch.stack(sigma_out, dim=1)  # Concat over time
        logits_out = torch.stack(logits_out, dim=1)
        m_sample_out = torch.stack(m_sample_out, dim=1)

        label_target_max_out = torch.stack(q_evals, dim=1)

        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        # FUNC END
        actions_prob = torch.stack(actions_prob, dim=1)
        log_prob_pi = torch.stack(actions_logprobs, dim=1)

        q_i_mean_negi_mean = torch.sum(actions_prob * (self.alpha * log_prob_pi - q_evals), dim=-1)  # (1,60,3) # TODO
        Q_i_mean_negi_mean = self.eval_qmix_net(q_i_mean_negi_mean, s)  # TODO # (1,60,1)
        Q_i_mean_negi_mean = Q_i_mean_negi_mean.repeat(repeats=(1, 1, self.n_agents))  # (1,60,3)
        policy_loss = (Q_i_mean_negi_mean * mask).sum() / mask.sum()
        self.policy_optimiser.zero_grad()
        policy_loss.backward(retain_graph=True)  # policy gradient
        # grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_params, self.args.grad_norm_clip)
        self.policy_optimiser.step()

        target_entropy = -1. * self.n_actions
        # alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean() #target_entropy=-2
        alpha_loss = (torch.sum(actions_prob.detach() * (-self.log_alpha * (log_prob_pi + target_entropy).detach()),
                                dim=-1) * mask).sum() / mask.sum()
        # print('alpha loss: ',alpha_loss)
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()


        # 取每个agent动作对应的Q值，并且把最后不需要的一维去掉，因为最后一维只有一个值了
        q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)


        # 得到target_q
        q_targets[avail_u_next == 0.0] = - 9999999

        label_target_max_out[avail_u_next == 0] = -9999999
        label_target_actions = q_targets.max(dim=3, keepdim=True)[1]


        q_targets = q_targets.max(dim=3)[0]

        #######################


        q_total_eval = self.eval_qmix_net(q_evals, s)
        q_total_target = self.target_qmix_net(q_targets, s_next)

        targets = r + self.args.gamma * q_total_target * (1 - terminated)

        td_error = (q_total_eval - targets.detach())
        masked_td_error = mask * td_error  # 抹掉填充的经验的td_error

        # 不能直接用mean，因为还有许多经验是没用的，所以要求和再比真实的经验数，才是真正的均值
        loss = (masked_td_error ** 2).sum() / mask.sum()

#######################
        expressiveness_loss = 0.0
        label_prob = torch.gather(logits_out, 3, label_target_actions).squeeze(3)
        expressiveness_loss += (-torch.log(label_prob + 1e-6)).sum() / mask.sum()

        # Compute KL divergence
        compactness_loss = torch.distributions.kl_divergence(torch.distributions.Normal(mu_out, sigma_out), torch.distributions.Normal(self.s_mu, self.s_sigma)).sum() /mask.sum()

        # Entropy loss
        entropy_loss = -torch.distributions.Normal(self.s_mu, self.s_sigma).log_prob(m_sample_out).sum() / mask.sum()

        # Gate loss
        gate_loss = 0

        # Total loss
        # comm_beta = self.get_comm_beta(t_env)
        # comm_entropy_beta = self.get_comm_entropy_beta(t_env)
        comm_beta = 0.001
        comm_entropy_beta= 1e-6
        comm_loss = expressiveness_loss + comm_beta * compactness_loss + comm_entropy_beta * entropy_loss

        loss += comm_loss
        # comm_beta = torch.Tensor([comm_beta])
        # comm_entropy_beta = torch.Tensor([comm_entropy_beta])
###################

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()

        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())
            # self.target_pagent.load_state_dict(self.eval_pagent.state_dict())
################################################
    def get_comm_beta(self, t_env):
        comm_beta = self.args.comm_beta
        if self.args.is_comm_beta_decay and t_env > self.args.comm_beta_start_decay:
            comm_beta += 1. * (self.args.comm_beta_target - self.args.comm_beta) / \
                         (self.args.comm_beta_end_decay - self.args.comm_beta_start_decay) * \
                         (t_env - self.args.comm_beta_start_decay)
        return comm_beta

    def get_comm_entropy_beta(self, t_env):
        comm_entropy_beta = self.args.comm_entropy_beta
        if self.args.is_comm_entropy_beta_decay and t_env > self.args.comm_entropy_beta_start_decay:
            comm_entropy_beta += 1. * (self.args.comm_entropy_beta_target - self.args.comm_entropy_beta) / \
                                 (self.args.comm_entropy_beta_end_decay - self.args.comm_entropy_beta_start_decay) * \
                                 (t_env - self.args.comm_entropy_beta_start_decay)
        return comm_entropy_beta

    def _logits(self, bs, inputs):
        # shape = (bs * self.n_agents, -1)
        t_logits = self.comm.inference_model(inputs)
        logits = torch.nn.functional.softmax(t_logits, dim=1)
        return logits

##################################################


    def _get_inputs(self, batch, transition_idx):
        # 取出所有episode上该transition_idx的经验，u_onehot要取出所有，因为要用到上一条
        obs, obs_next, u_onehot = batch['o'][:, transition_idx], \
                                  batch['o_next'][:, transition_idx], batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)
        # 给obs添加上一个动作、agent编号

        if self.args.last_action:
            if transition_idx == 0:  # 如果是第一条经验，就让前一个动作为0向量
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
            inputs_next.append(u_onehot[:, transition_idx])
        if self.args.reuse_network:
            # 因为当前的obs三维的数据，每一维分别代表(episode编号，agent编号，obs维度)，直接在dim_1上添加对应的向量
            # 即可，比如给agent_0后面加(1, 0, 0, 0, 0)，表示5个agent中的0号。而agent_0的数据正好在第0行，那么需要加的
            # agent编号恰好就是一个单位矩阵，即对角线为1，其余为0
            inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
            inputs_next.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        # 要把obs中的三个拼起来，并且要把episode_num个episode、self.args.n_agents个agent的数据拼成40条(40,96)的数据，
        # 因为这里所有agent共享一个神经网络，每条数据中带上了自己的编号，所以还是自己的数据
        inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs_next], dim=1)
        return inputs, inputs_next

    def get_q_values(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        q_evals, q_targets = [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_inputs(batch, transition_idx)  # 给obs加last_action、agent_id
            if self.args.cuda:
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
                self.target_hidden = self.target_hidden.cuda()
            q_eval, self.eval_hidden = self.eval_rnn(inputs,
                                                     self.eval_hidden)  # inputs维度为(40,96)，得到的q_eval维度为(40,n_actions)
            q_target, self.target_hidden = self.target_rnn(inputs_next, self.target_hidden)

            # 把q_eval维度重新变回(8, 5,n_actions)
            q_eval = q_eval.view(episode_num, self.n_agents, -1)
            q_target = q_target.view(episode_num, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)
        # 得的q_eval和q_target是一个列表，列表里装着max_episode_len个数组，数组的的维度是(episode个数, n_agents，n_actions)
        # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        return q_evals, q_targets

    def init_hidden(self, episode_num):
        # 为每个episode中的每个agent都初始化一个eval_hidden、target_hidden
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.target_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.peval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.eval_pagent.init_hidden()
        # self.ptarget_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_qmix_net.state_dict(), self.model_dir + '/' + num + '_qmix_net_params.pkl')
        torch.save(self.eval_rnn.state_dict(), self.model_dir + '/' + num + '_rnn_net_params.pkl')
        torch.save(self.eval_pagent.state_dict(), self.model_dir + '/' + num + '_pagent_net_params.pkl')
        # torch.save(self.log_alpha,self.model_dir + '/' + '_log_alpha.pkl')
