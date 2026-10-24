import unittest
from unittest.mock import patch
import numpy as np
import torch
import mujoco
import glfw
from src.envs import Config, Go1WarpEnv
from src.envs.viewer import MujocoViewer


class ViewerTests(unittest.TestCase):
    def test_mouse_drag_and_scroll_use_current_mujoco_api(self):
        e = Go1WarpEnv(Config(num_envs=1, device='cpu', backend='mujoco'))
        v = MujocoViewer.__new__(MujocoViewer)
        v.model = e.model
        v.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(e.model, v.camera)
        v.button_left, v.button_right, v.button_middle = True, False, False
        v.last_x = v.last_y = 0
        before = v.camera.elevation
        with patch('src.envs.viewer.glfw.get_window_size', return_value=(800, 600)), \
             patch('src.envs.viewer.glfw.get_key', return_value=glfw.RELEASE):
            v.mouse_move(None, 40, 30)
        self.assertNotEqual(v.camera.elevation, before)
        distance = v.camera.distance
        v.scroll(None, 0, 1)
        self.assertNotEqual(v.camera.distance, distance)
        e.close()

    def test_selected_snapshot_is_independent_and_does_not_step(self):
        e = Go1WarpEnv(Config(num_envs=2, device='cpu', backend='mujoco'))
        e.step(torch.zeros(2, 12))
        e.cpu_data[1].qpos[0] = 4.
        before = [d.qpos.copy() for d in e.cpu_data]
        times = [d.time for d in e.cpu_data]
        d = e.copy_render_data(1)
        np.testing.assert_array_equal(d.qpos, before[1])
        self.assertEqual(d.time, times[1])
        # mjv_updateScene must accept the snapshot and track the chosen robot.
        scene, opt, cam = mujoco.MjvScene(e.model, maxgeom=1000), mujoco.MjvOption(), mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(e.model, cam)
        mujoco.mjv_updateScene(e.model, d, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
        self.assertGreater(scene.ngeom, 0)
        d.qpos[0] = 100.
        for i, cpu in enumerate(e.cpu_data):
            np.testing.assert_array_equal(cpu.qpos, before[i])
            self.assertEqual(cpu.time, times[i])
        full = e.copy_render_data(1, full_state=True)
        self.assertEqual(full.ncon, e.cpu_data[1].ncon)
        np.testing.assert_array_equal(full.efc_force, e.cpu_data[1].efc_force)
        with self.assertRaises(IndexError): e.copy_render_data(2)
        e.close()

    def test_camera_and_keyboard_without_window(self):
        e = Go1WarpEnv(Config(num_envs=1, device='cpu', backend='mujoco'))
        v = MujocoViewer.__new__(MujocoViewer)
        v.model = e.model
        v.camera = mujoco.MjvCamera()
        v.camera_mode = None
        v.camera_mode_target = 'follow'
        v.scene_option = mujoco.MjvOption()
        v.set_camera()
        self.assertEqual(v.camera.trackbodyid, e.model.body('trunk').id)
        v.keyboard(None, glfw.KEY_TAB, 0, glfw.RELEASE, 0)
        v.set_camera()
        self.assertEqual(v.camera.type, mujoco.mjtCamera.mjCAMERA_FREE)
        self.assertEqual(v.camera.trackbodyid, -1)
        groups_before = v.scene_option.geomgroup.copy()
        for key in range(glfw.KEY_0, glfw.KEY_9 + 1):
            v.keyboard(None, key, 0, glfw.RELEASE, 0)
        np.testing.assert_array_equal(v.scene_option.geomgroup, 1 - groups_before)
        v.keyboard(None, glfw.KEY_C, 0, glfw.RELEASE, 0)
        self.assertTrue(v.show_contacts)
        e.close()

    def test_no_readback_on_skipped_frames_and_clean_close(self):
        from unittest.mock import Mock
        e = Go1WarpEnv(Config(num_envs=1, device='cpu', backend='mujoco'))
        v = MujocoViewer.__new__(MujocoViewer)
        v._closed = False
        v.window = object()
        v.paused = False
        v.throttle = False
        v.frames = 1
        v._next_frame = float('inf')
        getter = Mock(return_value=e.copy_render_data())
        with patch('src.envs.viewer.glfw.window_should_close', return_value=False), \
             patch('src.envs.viewer.glfw.poll_events'):
            self.assertTrue(v.render(getter))
        getter.assert_not_called()
        v._closed = True
        self.assertFalse(v.render(getter))
        e.close()


if __name__ == '__main__':
    unittest.main()
