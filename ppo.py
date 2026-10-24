"""PPO math and GPU-resident storage, independent of the simulation backend."""
import torch
from torch import nn


@torch.no_grad()
def compute_gae(rewards, values, next_values, terminated, truncated, gamma=.99, lam=.95):
    """All tensors have shape (T,N). next_values are evaluated BEFORE reset.

    Bootstrap timeouts but never true terminals. Stop recursion at either boundary.
    Return tensors have independent storage and include an unfinished rollout tail.
    """
    advantages = torch.zeros_like(rewards)
    tail = torch.zeros_like(rewards[0])
    for t in reversed(range(rewards.shape[0])):
        bootstrap = torch.where(terminated[t], 0., next_values[t])
        delta = rewards[t] + gamma * bootstrap - values[t]
        tail = delta + gamma * lam * (~(terminated[t] | truncated[t])).float() * tail
        advantages[t] = tail
    return advantages, advantages + values


class EpisodeReturns:
    """Online R=sum(r_t), J=sum(gamma**t*r_t); preserve unfinished episodes."""
    def __init__(self, n_envs, device, gamma):
        self.R = torch.zeros(n_envs, device=device, dtype=torch.float64)
        self.J = torch.zeros_like(self.R)
        self.discount = torch.ones_like(self.R)
        self.length = torch.zeros(n_envs, device=device, dtype=torch.long)
        self.gamma = gamma

    @torch.no_grad()
    def update(self, rewards, done):
        self.R += rewards
        self.J += self.discount * rewards
        self.discount *= self.gamma
        self.length += 1
        ids = torch.where(done)[0]
        result = {'ids': ids, 'R': self.R[ids].clone(), 'J': self.J[ids].clone(),
                  'length': self.length[ids].clone()}
        self.R[ids] = 0
        self.J[ids] = 0
        self.discount[ids] = 1
        self.length[ids] = 0
        return result


class Rollout:
    def __init__(self, steps, n_envs, obs_dim, action_dim, device):
        self.obs = torch.zeros((steps, n_envs, obs_dim), device=device)
        self.actions = torch.zeros((steps, n_envs, action_dim), device=device)
        self.mu = torch.zeros_like(self.actions)
        self.sigma = torch.zeros_like(self.actions)
        for name in ('logprob', 'values', 'next_values', 'rewards'):
            setattr(self, name, torch.zeros((steps, n_envs), device=device))
        self.terminated = torch.zeros((steps, n_envs), device=device, dtype=torch.bool)
        self.truncated = torch.zeros_like(self.terminated)

    def finish(self, gamma, lam):
        self.advantages, self.returns = compute_gae(
            self.rewards, self.values, self.next_values, self.terminated, self.truncated, gamma, lam)


@torch.no_grad()
def gaussian_kl(old_mu, old_sigma, new_mu, new_sigma):
    """Exact KL(old || new), summed across independent action dimensions."""
    return (torch.log(new_sigma / old_sigma)
            + (old_sigma.square() + (old_mu - new_mu).square()) / (2 * new_sigma.square())
            - .5).sum(-1)


@torch.no_grad()
def adapt_lr(optimizer, kl, desired_kl):
    lr = optimizer.param_groups[0]['lr']
    if kl > 2. * desired_kl:
        lr = max(1e-5, lr / 1.5)
    elif 0. < kl < desired_kl / 2.:
        lr = min(1e-2, lr * 1.5)
    for group in optimizer.param_groups:
        group['lr'] = lr
    return lr


def ppo_update(agent, optimizer, rollout, args):
    obs = rollout.obs.flatten(0, 1)
    actions = rollout.actions.flatten(0, 1)
    old_mu, old_sigma = rollout.mu.flatten(0, 1), rollout.sigma.flatten(0, 1)
    old_logprob, old_values = rollout.logprob.flatten(), rollout.values.flatten()
    returns = rollout.returns.flatten()
    advantages = rollout.advantages.flatten()
    # RSL-RL default: normalize over the full rollout once, not each minibatch.
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    totals = torch.zeros(6, device=obs.device)
    batches = 0
    agent.train()
    for _ in range(args.n_epochs_policy):
        # A new permutation every epoch; tensor_split keeps every sample.
        indices = torch.randperm(len(obs), device=obs.device)

        for ids in torch.tensor_split(indices, args.n_minibatches):
            _, logprob, entropy, value, mu, sigma = agent.action_value(obs[ids], actions[ids])

            with torch.no_grad():
                kl = gaussian_kl(old_mu[ids], old_sigma[ids], mu, sigma).mean().clamp_min(0.)
                if args.schedule == 'adaptive':
                    adapt_lr(optimizer, kl.item(), args.desired_kl)

            ratio = (logprob - old_logprob[ids]).exp()
            unclipped = -advantages[ids] * ratio
            clipped = -advantages[ids] * ratio.clamp(1. - args.eps_ppo, 1. + args.eps_ppo)
            policy_loss = torch.maximum(unclipped, clipped).mean()
            squared = (value - returns[ids]).square()

            if args.clip_vloss:
                clipped_value = old_values[ids] + (value - old_values[ids]).clamp(-args.eps_ppo, args.eps_ppo)
                squared = torch.maximum(squared, (clipped_value - returns[ids]).square())

            # RSL-RL value loss convention: MSE, without CleanRL's extra 0.5.

            value_loss = squared.mean()

            loss = policy_loss + args.vf_coef * value_loss - args.ent_coeff * entropy.mean()

            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite PPO loss; inspect rollout and policy statistics')

            optimizer.zero_grad(set_to_none=True)

            loss.backward()

            actor_params = list(agent.actor.parameters()) + [agent.log_std]


            actor_norm = nn.utils.clip_grad_norm_(actor_params, args.max_grad_norm, error_if_nonfinite=True)

            critic_norm = nn.utils.clip_grad_norm_(agent.critic.parameters(), args.max_grad_norm, error_if_nonfinite=True)

            optimizer.step()

            with torch.no_grad():
                agent.log_std.clamp_(args.log_std_min, args.log_std_max)
                totals += torch.stack((policy_loss.detach(), value_loss.detach(), kl,
                                       ((ratio - 1.).abs() > args.eps_ppo).float().mean(),
                                       actor_norm.detach(), critic_norm.detach()))
            batches += 1
    means = (totals / batches).tolist()
    result = dict(zip(('policy_loss', 'value_loss', 'kl', 'clip_fraction',
                       'actor_grad_norm', 'critic_grad_norm'), means))
    result['entropy'] = agent.entropy().item()
    result['learning_rate'] = optimizer.param_groups[0]['lr']
    var = torch.var(returns, unbiased=False)
    result['explained_variance'] = (1 - torch.var(returns - old_values, unbiased=False) / var).item() if var > 0 else 0.
    result['std_mean'] = agent.log_std.exp().mean().item()
    return result
