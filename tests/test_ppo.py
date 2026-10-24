import tempfile
from pathlib import Path
import unittest
import torch
from torch.distributions import Normal, kl_divergence
from src.networks import ActorCritic, PolicyNetwork, ValueNetwork
from src.algorithms.ppo import compute_gae, EpisodeReturns, gaussian_kl, adapt_lr
from src.algorithms.trainer import parse_args, train


class PPOTests(unittest.TestCase):
    def test_timeout_bootstrap_and_episode_boundaries(self):
        rewards = torch.tensor([[1., 1.], [2., 2.], [3., 3.]])
        values = torch.zeros_like(rewards)
        next_values = torch.tensor([[0., 0.], [10., 10.], [5., 5.]])
        term = torch.tensor([[False, False], [True, False], [False, False]])
        trunc = torch.tensor([[False, False], [False, True], [False, False]])
        adv, returns = compute_gae(rewards, values, next_values, term, trunc, .9, 1.)
        torch.testing.assert_close(returns, torch.tensor([[2.8, 10.9], [2., 11.], [7.5, 7.5]]))
        torch.testing.assert_close(adv, returns)
        # True termination dominates simultaneous timeout, even with nonfinite V(next).
        a, _ = compute_gae(torch.ones(1, 1), torch.zeros(1, 1), torch.full((1, 1), float('nan')),
                           torch.ones(1, 1, dtype=torch.bool), torch.ones(1, 1, dtype=torch.bool))
        self.assertEqual(a.item(), 1.)

    def test_episode_returns_preserve_unfinished_episodes(self):
        m = EpisodeReturns(2, 'cpu', .5)
        first = m.update(torch.tensor([1., 10.]), torch.tensor([False, True]))
        self.assertEqual(first['R'].tolist(), [10.])
        second = m.update(torch.tensor([2., 20.]), torch.tensor([True, False]))
        self.assertEqual(second['R'].tolist(), [3.])
        self.assertEqual(second['J'].tolist(), [2.])
        self.assertEqual(second['length'].tolist(), [2])
        third = m.update(torch.tensor([3., 30.]), torch.tensor([False, True]))
        self.assertEqual(third['R'].tolist(), [50.])
        self.assertEqual(third['J'].tolist(), [35.])

    def test_analytic_kl_and_learning_rate(self):
        old_mu, mu = torch.randn(4, 12), torch.randn(4, 12)
        old_std, std = torch.rand(4, 12) + .1, torch.rand(4, 12) + .1
        expected = kl_divergence(Normal(old_mu, old_std), Normal(mu, std)).sum(-1)
        torch.testing.assert_close(gaussian_kl(old_mu, old_std, mu, std), expected)
        p = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.Adam([p], lr=.001)
        self.assertAlmostEqual(adapt_lr(opt, .03, .01), .001 / 1.5)
        self.assertAlmostEqual(adapt_lr(opt, .001, .01), .001)

    def test_network_shapes_and_raw_action_likelihood(self):
        torch.set_num_threads(1)
        a = ActorCritic(n_features=16)
        for shape in ((1,48), (3,48), (3,1,48)):
            x = torch.randn(shape)
            self.assertEqual(a.actor(x).shape, (shape[0],12))
            self.assertEqual(a.value(x).shape, (shape[0],))
        # PPO evaluates the original sample, including samples outside action bounds.
        x = torch.zeros(2,48)
        raw = torch.full((2,12), 2.)
        _, logprob, entropy, _, _, _ = a.action_value(x, raw)
        torch.testing.assert_close(logprob, a.distribution(x).log_prob(raw).sum(-1))
        self.assertTrue(bool(torch.isfinite(entropy).all()))
        self.assertNotEqual(logprob[0].item(), a.distribution(x).log_prob(raw.clamp(-1,1)).sum(-1)[0].item())

    def test_reference_defaults(self):
        a = parse_args([])
        self.assertEqual((a.num_envs,a.n_steps_per_fit,a.n_minibatches,a.n_epochs_policy),(100,1000,20,10))
        self.assertEqual((a.lr,a.gamma,a.lam,a.eps_ppo,a.ent_coeff),(1e-3,.99,.95,.2,.01))
        self.assertEqual(a.n_features,256)

    def test_cpu_train_save_resume_and_evaluation_logs(self):
        import csv
        with tempfile.TemporaryDirectory() as folder:
            common = ['--backend','mujoco','--device','cpu','--n-envs','2','--n-steps','16',
                      '--n-steps-per-fit','8','--n-minibatches','3','--n-epochs-policy','2',
                      '--n-features','16','--n-steps-test','3','--eval-envs','2',
                      '--episode-seconds','.08','--collision-only','--output-dir',folder+'/runs',
                      '--log-dir',folder+'/logs','--save-epochs','--save-every','1']
            args = parse_args(common + ['--n-epochs','1'])
            torch.manual_seed(args.seed)
            initial = {k: v.clone() for k, v in ActorCritic(n_features=16).state_dict().items()}
            first = train(args)
            ckpt = torch.load(first/'last.pt',map_location='cpu',weights_only=False)
            self.assertEqual(ckpt['global_step'],32)
            self.assertEqual(ckpt['update'],2)
            self.assertEqual(first, Path(folder)/'runs'/'LLC_PPO'/'run_0')
            self.assertTrue((first/'best.pt').exists())
            self.assertTrue((first/'epoch_0001.pt').exists())
            evaluation_log = Path(folder)/'logs'/'LLC_PPO'/'run_0'/'evaluation.csv'
            with evaluation_log.open() as f:
                logs = list(csv.DictReader(f))
            self.assertEqual(len(logs),2)  # Initial and end-of-epoch evaluation.
            self.assertEqual(int(logs[-1]['eval_steps']),3)
            self.assertEqual(int(logs[-1]['eval_transitions']),6)
            self.assertTrue(float(logs[-1]['R']) == float(logs[-1]['R']))
            self.assertTrue(any(not torch.equal(ckpt['agent'][k], initial[k]) for k in ckpt['agent']))
            resumed = train(parse_args(common + ['--n-epochs','2','--resume',str(first/'last.pt')]))
            final = torch.load(resumed/'last.pt',map_location='cpu',weights_only=False)
            self.assertEqual((final['epoch'],final['global_step'],final['update']),(2,64,4))
            self.assertTrue(all(bool(torch.isfinite(t).all()) for t in final['agent'].values()))
            with evaluation_log.open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 4)


if __name__ == '__main__':
    unittest.main()
