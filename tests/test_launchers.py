import contextlib
from dataclasses import is_dataclass
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import uuid

import torch

from src import ROOT
from src.algorithms.config import (parse_train_args, parse_play_args,
                                    resolve_play_config, saved_values)
from src.algorithms.paths import output_paths
from src.envs import Config, make_env
from src.envs.model import load_model


class LauncherTests(unittest.TestCase):
    def test_dataclass_flags_ranges_aliases_and_retired_exports(self):
        cfg = parse_train_args(['--name-exp','test', '--run-id','7', '--n-envs','4',
                                '--solver-iterations','75','--no-physics-graph',
                                '--no-fast-contacts','--target-vx','1.2',
                                '--dr-com-range','-0.03','0.04'])
        self.assertTrue(is_dataclass(cfg))
        self.assertEqual((cfg.experiment_name,cfg.run_id,cfg.num_envs),('test',7,4))
        self.assertEqual(cfg.dr_com_range,(-.03,.04))
        self.assertEqual(cfg.command_vx,1.2)
        self.assertFalse(cfg.use_physics_graph); self.assertFalse(cfg.fast_contacts)
        self.assertEqual(cfg.solver_iterations,75)
        with patch.dict(os.environ, {'GO1_WARP_SOLVER_ITERATIONS':'2', 'GO1_WARP_PHYSICS_GRAPH':'invalid'}):
            with contextlib.redirect_stdout(io.StringIO()):
                env = make_env(Config(num_envs=1,backend='mujoco',device='cpu',solver_iterations=75))
            self.addCleanup(env.close)
            self.assertEqual(env.model.opt.iterations,75)

    def test_numbered_checkpoint_options_and_default_paths(self):
        cfg = parse_train_args([])
        self.assertEqual((cfg.experiment_name,cfg.run_id),('LLC_PPO',0))
        self.assertEqual(output_paths(cfg),(ROOT/'runs/LLC_PPO/run_0', ROOT/'logs/LLC_PPO/run_0'))
        self.assertFalse(cfg.save_epochs)
        self.assertTrue(parse_train_args(['--save-every','2']).save_epochs)
        self.assertFalse(parse_train_args(['--save-every','0']).save_epochs)
        self.assertFalse(parse_train_args(['--save-every','2','--no-save-epochs']).save_epochs)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit): parse_train_args(['--experiment-name','../outside'])
            with self.assertRaises(SystemExit): parse_train_args(['--run-id','-1'])

    def test_checkpoint_physics_and_explicit_play_override(self):
        saved = dict(args=dict(n_features=16, gamma=.99), env_config=dict(
            environment='domain_randomized',num_envs=4096,device='cuda:0',backend='warp',
            solver_iterations=55,use_physics_graph=False,fast_contacts=False,
            dr_friction_range=(.2,1.4)))
        config = resolve_play_config(parse_play_args(['example.pt','--num-envs','2','--physics-graph',
                                                      '--solver-iterations','88','--headless']), saved)
        self.assertEqual(config.environment,'domain_randomized')
        self.assertEqual(config.dr_friction_range,(.2,1.4))
        self.assertEqual(config.num_envs,2)
        self.assertEqual(config.solver_iterations,88)
        self.assertTrue(config.use_physics_graph)
        self.assertFalse(config.fast_contacts)
        self.assertEqual(config.seed,10000)

    def test_legacy_checkpoint_arguments_and_asset_paths(self):
        values = saved_values(dict(args=dict(n_envs=5,target_vx=1.2,name_exp='OLD'),
                                   env_config=dict(xml_path='models/walk_go1.xml')))
        self.assertEqual((values['num_envs'],values['command_vx'],values['experiment_name']),(5,1.2,'OLD'))
        self.assertTrue(Path(values['xml_path']).exists())
        with tempfile.TemporaryDirectory() as cwd:
            previous = Path.cwd()
            try:
                os.chdir(cwd)
                self.assertEqual(load_model().nu,12)
                self.assertEqual(load_model('src/models/walk_go1.xml').nu,12)
                self.assertEqual(load_model(collision_only=True).nmesh,0)
            finally:
                os.chdir(previous)

    def run_script(self, filename, arguments, cwd):
        result = subprocess.run([sys.executable, str(ROOT/filename), *arguments],cwd=cwd,
                                 env=os.environ.copy(),capture_output=True,text=True,timeout=60)
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        return result

    def test_direct_scripts_from_foreign_cwd_and_old_checkpoint_playback(self):
        experiment = 'test-'+uuid.uuid4().hex[:10]
        self.addCleanup(shutil.rmtree, ROOT/'runs'/experiment, True)
        self.addCleanup(shutil.rmtree, ROOT/'logs'/experiment, True)
        with tempfile.TemporaryDirectory() as folder:
            common = ['--backend','mujoco','--device','cpu','--num-envs','2',
                       '--n-steps','4','--n-steps-per-fit','2','--n-minibatches','2',
                       '--n-epochs','1','--n-epochs-policy','1','--n-features','8',
                       '--n-steps-test','2','--eval-envs','1','--episode-seconds','.04',
                       '--experiment-name',experiment,'--run-id','12']
            self.run_script('train.py',common,folder)
            run=ROOT/'runs'/experiment/'run_12'
            self.assertTrue((run/'last.pt').exists()); self.assertTrue((run/'best.pt').exists())
            self.assertFalse(list(run.glob('epoch_*.pt')))
            self.assertTrue((ROOT/'logs'/experiment/'run_12'/'evaluation.csv').exists())
            result=self.run_script('play.py',[str(run/'last.pt'),'--backend','mujoco',
                                                '--device','cpu','--headless','--steps','3',
                                                '--num-envs','2'],folder)
            self.assertIn('"eval_transitions": 6',result.stdout)
            # Recreate the old checkpoint's flat dict/state_dict format.
            old=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
            old['env_config'].pop('environment')
            old['args']['n_envs']=old['args'].pop('num_envs')
            old['args']['name_exp']=old['args'].pop('experiment_name')
            old['args']['target_vx']=old['args'].pop('command_vx')
            old['env_config']['xml_path']='models/walk_go1.xml'
            legacy=Path(folder)/'legacy.pt'; torch.save(old,legacy)
            self.run_script('play.py',[str(legacy),'--backend','mujoco','--device','cpu',
                                       '--headless','--steps','2'],folder)
            self.run_script('benchmark_eval.py',[str(legacy),'--backend','mujoco','--device','cpu',
                                                 '--env-counts','1','2','--steps','2','--warmups','0',
                                                 '--repeats','1','--output',str(Path(folder)/'benchmark.csv')],folder)
            for filename in ('train.py','play.py','benchmark_eval.py','demo.py'):
                self.run_script(filename,['--help'],folder)
