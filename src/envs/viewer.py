"""GLFW viewer adapted for a single CPU snapshot of a batched environment.

Use on the main thread. `render` accepts MjData or a zero-argument snapshot
callback. The callback is called only when a frame will actually be drawn.
"""
import time
import glfw
import mujoco


class MujocoViewer:
    def __init__(self, model, dt, width=1280, height=720, throttle=True):
        if dt <= 0:
            raise ValueError('dt must be positive')
        self.model, self.dt = model, float(dt)
        self.button_left = self.button_right = self.button_middle = False
        self.last_x = self.last_y = 0
        self.frames = 0
        self.target_render_time = 1 / 60.
        self.time_per_render = self.target_render_time
        self.run_speed_factor = 1.
        self.paused = self.hide_menu = False
        self.overlay = {}
        self.font_scale = 100
        self.throttle = throttle
        self._budget = 0.
        self._last_frame = None
        self._next_frame = time.perf_counter()
        self._closed = False
        self.world_id = 0
        self.window = self.context = self.scene = None
        if not glfw.init():
            raise RuntimeError('GLFW initialization failed; a desktop display is required')
        try:
            glfw.window_hint(glfw.SCALE_TO_MONITOR, glfw.TRUE)
            self.window = glfw.create_window(width, height, 'Go1 MuJoCo Warp', None, None)
            if not self.window:
                raise RuntimeError('GLFW could not create an OpenGL window')
            glfw.make_context_current(self.window)
            glfw.swap_interval(0)  # Explicit frame pacing below.
            glfw.set_mouse_button_callback(self.window, self.mouse_button)
            glfw.set_cursor_pos_callback(self.window, self.mouse_move)
            glfw.set_key_callback(self.window, self.keyboard)
            glfw.set_scroll_callback(self.window, self.scroll)
            self.scene = mujoco.MjvScene(model, maxgeom=10000)
            self.scene_option = mujoco.MjvOption()
            # Visual meshes are group 2, collision primitives are group 3.
            # Expose the robot's collision geometry when visual meshes are stripped.
            if model.nmesh == 0:
                self.scene_option.geomgroup[3] = 1
            self.camera = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(model, self.camera)
            self.all_camera_modes = ('static', 'follow')
            self.camera_mode = None
            self.camera_mode_target = 'follow'
            self.set_camera()
            w, h = glfw.get_framebuffer_size(self.window)
            self.viewport = mujoco.MjrRect(0, 0, w, h)
            self.context = mujoco.MjrContext(model, mujoco.mjtFontScale(self.font_scale))
        except Exception:
            self.stop()
            raise

    @property
    def is_running(self):
        return not self._closed and not glfw.window_should_close(self.window)

    @property
    def show_contacts(self):
        return bool(self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE]
                    or self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONSTRAINT])

    def mouse_button(self, window, button, act, mods):
        self.button_left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        self.button_right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        self.button_middle = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        self.last_x, self.last_y = glfw.get_cursor_pos(window)

    def mouse_move(self, window, x_pos, y_pos):
        if not (self.button_left or self.button_right or self.button_middle):
            return
        dx, dy = x_pos - self.last_x, y_pos - self.last_y
        self.last_x, self.last_y = x_pos, y_pos
        width, height = glfw.get_window_size(window)
        if width <= 0 or height <= 0:
            return
        shift = (glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
                 or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)
        if self.button_right:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H if shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif self.button_left:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        mujoco.mjv_moveCamera(self.model, action, dx / width, dy / height, self.camera)

    def keyboard(self, window, key, scancode, act, mods):
        if act != glfw.RELEASE:
            return
        if key == glfw.KEY_ESCAPE:
            self.close()
        elif key == glfw.KEY_SPACE:
            self.paused = not self.paused
        elif key == glfw.KEY_H:
            self.hide_menu = not self.hide_menu
        elif key == glfw.KEY_TAB:
            # Cycle from the actual active mode: first TAB switches follow -> static.
            self.camera_mode_target = 'static' if self.camera_mode_target == 'follow' else 'follow'
        elif key == glfw.KEY_S:
            self.run_speed_factor = max(1 / 128., self.run_speed_factor / 2.)
        elif key == glfw.KEY_F:
            self.run_speed_factor = min(128., self.run_speed_factor * 2.)
        elif key == glfw.KEY_C:
            for flag in (mujoco.mjtVisFlag.mjVIS_CONTACTFORCE, mujoco.mjtVisFlag.mjVIS_CONSTRAINT):
                self.scene_option.flags[flag] = not self.scene_option.flags[flag]
        elif key == glfw.KEY_T:
            flag = mujoco.mjtVisFlag.mjVIS_TRANSPARENT
            self.scene_option.flags[flag] = not self.scene_option.flags[flag]
        elif glfw.KEY_0 <= key <= glfw.KEY_9:
            group = key - glfw.KEY_0
            # MuJoCo has six geom groups; keys 6-9 must not index beyond the array.
            if group < len(self.scene_option.geomgroup):
                self.scene_option.geomgroup[group] ^= 1

    def scroll(self, window, x_offset, y_offset):
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM,
                             0, .05 * y_offset, self.camera)

    def set_camera(self):
        if self.camera_mode_target == self.camera_mode:
            return
        self.camera.fixedcamid = -1
        if self.camera_mode_target == 'static':
            self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.camera.trackbodyid = -1
            self.camera.lookat[:] = [0., 0., .25]
            self.camera.distance, self.camera.elevation, self.camera.azimuth = 15., -45., 90.
        else:
            trunk = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'trunk')
            if trunk < 0:
                raise ValueError('Follow camera requires a body named trunk')
            self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.camera.trackbodyid = trunk
            self.camera.distance, self.camera.elevation, self.camera.azimuth = 3.5, -15., 90.
        self.camera_mode = self.camera_mode_target

    def create_overlay(self):
        self.overlay.clear()
        self.overlay[mujoco.mjtGridPos.mjGRID_TOPLEFT] = [
            'SPACE: pause\nH: hide menu\nTAB: camera\nCamera\nWorld\nSpeed [S/F]\nC: contacts / constraints\nT: transparent\n0-5: geom groups\nESC: close',
            f'\n\n\n{self.camera_mode}\n{self.world_id}\n{self.run_speed_factor:.3f}x\n\n\n\n',
        ]
        self.overlay[mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT] = [
            'Display FPS', f'{1 / max(self.time_per_render, 1e-9):.0f}']

    def _draw(self, data):
        glfw.make_context_current(self.window)
        self.set_camera()
        self.create_overlay()
        mujoco.mjv_updateScene(self.model, data, self.scene_option, None,
                              self.camera, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        self.viewport.width, self.viewport.height = glfw.get_framebuffer_size(self.window)
        if self.viewport.width > 0 and self.viewport.height > 0:
            mujoco.mjr_render(self.viewport, self.scene, self.context)
            if not self.hide_menu:
                for gridpos, (t1, t2) in self.overlay.items():
                    mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_SHADOW, gridpos,
                                       self.viewport, t1, t2, self.context)
            glfw.swap_buffers(self.window)
        now = time.perf_counter()
        if self._last_frame is not None:
            self.time_per_render = max(now - self._last_frame, 1e-9)
        self._last_frame = now
        self.frames += 1

    def render(self, data):
        """Return False on close. Pause blocks the calling simulation loop.

        `data` may be MjData or a callback returning a refreshed MjData snapshot.
        With throttle=False simulation runs freely; frames are capped at 60 Hz.
        With throttle=True simulation is paced to the requested speed factor.
        """
        if not self.is_running:
            self.stop()
            return False
        glfw.poll_events()  # Still process close/pause even when skipping a frame.
        while self.paused and self.is_running:
            self._draw(data() if callable(data) else data)
            glfw.wait_events_timeout(self.target_render_time)
            self._next_frame = time.perf_counter()
        if not self.is_running:
            self.stop()
            return False
        if self.throttle:
            self._budget += self.dt / self.run_speed_factor
            count = int(self._budget / self.target_render_time)
            self._budget -= count * self.target_render_time
        else:
            count = int(time.perf_counter() >= self._next_frame)
        if self.frames == 0:
            count = max(1, count)
        for _ in range(count):
            # Process events at display rate even when slow-motion repeats a state.
            glfw.poll_events()
            while self.paused and self.is_running:
                self._draw(data() if callable(data) else data)
                glfw.wait_events_timeout(self.target_render_time)
                self._next_frame = time.perf_counter()
            if not self.is_running:
                self.stop()
                return False
            if self.throttle:
                time.sleep(max(0., self._next_frame - time.perf_counter()))
            # Refresh for each drawn frame so toggling contacts while paused works.
            snapshot = data() if callable(data) else data
            self._draw(snapshot)
            self._next_frame = time.perf_counter() + self.target_render_time
        return True

    def stop(self):
        if self._closed:
            return
        self._closed = True
        if self.window:
            glfw.make_context_current(self.window)
        if self.context is not None:
            self.context.free()
        self.scene = None  # MjvScene releases its native storage on destruction.
        if self.window:
            glfw.destroy_window(self.window)
        # This viewer owns GLFW; use a shared lifecycle if embedding in another GUI.
        glfw.terminate()

    def close(self):
        self.stop()
