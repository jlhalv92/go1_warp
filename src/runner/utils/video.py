"""Stream one evaluation world's RGB frames to MP4, paced by simulation time.

The independent offscreen context works with or without a GLFW viewer. Only
scheduled frames copy a selected-world snapshot; the simulation is never stepped
by this module. Encoding streams to FFmpeg rather than retaining frames in RAM.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib import import_module
import math
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import threading
from types import SimpleNamespace

import mujoco
import numpy as np
import glfw

from .paths import rooted
from src.envs.viewer import current_glfw_context


def video_path(directory, label):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S-%fZ')
    return rooted(directory) / f'{label}_{stamp}.mp4'


def ffmpeg_executable():
    executable = shutil.which('ffmpeg')
    if executable:
        return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as error:
        raise RuntimeError('Video recording needs FFmpeg. Install it, or run '
                           'python -m pip install -r requirements-video.txt') from error


class _MP4Writer:
    def __init__(self, path, width, height, fps):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(self.path)
        self._errors = tempfile.TemporaryFile()
        try:
            self._process = subprocess.Popen([
                ffmpeg_executable(), '-nostdin', '-loglevel', 'error', '-y',
                '-f', 'rawvideo', '-pixel_format', 'rgb24',
                '-video_size', f'{width}x{height}', '-framerate', str(fps),
                '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'fast',
                '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(self.path),
            ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._errors)
        except Exception:
            self._errors.close()
            raise
        self._closed = False

    def write(self, pixels):
        try:
            self._process.stdin.write(pixels.tobytes())
        except (BrokenPipeError, OSError) as error:
            self.close()
            raise RuntimeError(f'FFmpeg could not write {self.path}') from error

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            try:
                self._process.stdin.close()
            except BrokenPipeError:
                pass
            try:
                code = self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
                raise RuntimeError(f'FFmpeg did not finish {self.path}') from None
            if code:
                self._errors.seek(0)
                reason = self._errors.read().decode(errors='replace').strip()
                raise RuntimeError(f'FFmpeg failed for {self.path}: {reason}')
        finally:
            self._errors.close()


class EvaluationVideo:
    def __init__(self, env, path, *, world_id=0, fps=30., width=1280, height=720, backend='auto'):
        if not 0 <= world_id < env.num_envs:
            raise ValueError('Video world must be within num_envs')
        if not math.isfinite(fps) or fps <= 0 or min(width, height) < 2 or width % 2 or height % 2:
            raise ValueError('Video needs positive FPS and positive even frame dimensions')
        # Resolve dependency failures before an evaluation starts.
        ffmpeg_executable()
        self.env, self.path, self.world_id = env, Path(path), world_id
        self.fps, self.width, self.height = fps, width, height
        self.backend = ('egl' if platform.system() == 'Linux' else 'glfw') if backend == 'auto' else backend
        self.frames = 0
        self._closed = False
        self._writer = self._gl = self._context = self.scene = None
        self._last_step = 0
        self._thread = threading.get_ident()
        self._pixels = np.empty((height, width, 3), dtype=np.uint8)
        try:
            # Import the requested context directly: no MUJOCO_GL export or
            # pre-import environment configuration is needed by the CLI.
            self._gl_module = import_module(f'mujoco.{self.backend}')
            with self._current_context():
                self._context = mujoco.MjrContext(env.model, mujoco.mjtFontScale.mjFONTSCALE_100)
                mujoco.mjr_resizeOffscreen(width, height, self._context)
                mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self._context)
            self.scene = mujoco.MjvScene(env.model, maxgeom=max(10000, env.model.ngeom + 1000))
            self.option = mujoco.MjvOption()
            self.option.geomgroup[5] = 1
            if env.model.nmesh == 0:
                self.option.geomgroup[3] = 1
            self.camera = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(env.model, self.camera)
            self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.camera.trackbodyid = env.model.body('trunk').id
            self.camera.fixedcamid = -1
            self.camera.distance, self.camera.elevation, self.camera.azimuth = 3.5, -15., 90.
            self.viewport = mujoco.MjrRect(0, 0, width, height)
        except Exception as error:
            self.close()
            raise RuntimeError(f'Could not create the {self.backend} video renderer: {error}. '
                               'On Linux use --video-backend egl with an EGL driver, or '
                               '--video-backend osmesa with libOSMesa installed.') from error

    def _check_thread(self):
        if threading.get_ident() != self._thread:
            raise RuntimeError('Create, use and close the video renderer on the same thread')

    @contextmanager
    def _current_context(self):
        """Detach GLFW, bind the recorder, release it, then restore GLFW.

        GLX and EGL must not both be current on one thread. GLFW cannot release
        an EGL context created outside GLFW, so release it through its own API.
        The scope also covers context creation and exception/cleanup paths.
        """
        self._check_thread()
        previous = current_glfw_context()
        if previous:
            glfw.make_context_current(None)
        bound = False
        try:
            if self._gl is None:
                self._gl = self._gl_module.GLContext(self.width, self.height)
            self._gl.make_current()
            bound = True
            yield
        finally:
            if bound:
                self._release_context()
            if previous:
                glfw.make_context_current(previous)

    def _release_context(self):
        if self.backend == 'egl':
            egl = self._gl_module.EGL
            if not egl.eglMakeCurrent(self._gl_module.EGL_DISPLAY, egl.EGL_NO_SURFACE,
                                      egl.EGL_NO_SURFACE, egl.EGL_NO_CONTEXT):
                raise RuntimeError('Failed to release the video EGL context')
        elif self.backend == 'osmesa':
            self._gl_module.osmesa.OSMesaMakeCurrent(None, None, self._gl_module.GL.GL_FLOAT, 0, 0)
        elif self.backend == 'glfw':
            glfw.make_context_current(None)

    def capture(self, env, step):
        """Post-step callback; call before resetting terminated/truncated worlds.

        FPS follows simulation time, independent of rendering or wall-clock
        speed. Frames above the control rate duplicate the latest state. Duration
        differs from simulated time by less than one output frame.
        """
        self._check_thread()
        if self._closed or env is not self.env or step <= self._last_step:
            raise ValueError('Video capture needs its original env and increasing step indices')
        self._last_step = step
        target = math.ceil(step * env.dt * self.fps - 1e-9)
        if target <= self.frames:
            return
        viewer = env.viewer
        active_viewer = viewer is not None and not getattr(viewer, '_closed', False)
        camera = viewer.camera if active_viewer else self.camera
        option = viewer.scene_option if active_viewer else self.option
        full_state = active_viewer and viewer.show_contacts
        snapshot = env.copy_render_data(self.world_id, full_state=full_state)
        mujoco.mjv_updateScene(env.model, snapshot, option, None, camera,
                              mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        if getattr(env.cfg, 'height_markers', False) and hasattr(env, '_draw_markers'):
            env._draw_markers(SimpleNamespace(scene=self.scene, world_id=self.world_id))
        with self._current_context():
            mujoco.mjr_render(self.viewport, self.scene, self._context)
            mujoco.mjr_readPixels(self._pixels, None, self.viewport, self._context)
        pixels = np.ascontiguousarray(self._pixels[::-1])  # OpenGL origin is bottom-left.
        if self._writer is None:
            self._writer = _MP4Writer(self.path, self.width, self.height, self.fps)
        while self.frames < target:
            self._writer.write(pixels)
            self.frames += 1

    def close(self):
        if self._closed:
            return
        self._check_thread()
        self._closed = True
        try:
            if self._writer is not None:
                self._writer.close()
        finally:
            try:
                if self._context is not None:
                    with self._current_context():
                        self._context.free()
            finally:
                self._context = self.scene = None
                if self._gl is not None:
                    self._gl.free()
                    self._gl = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def make_video(env, config, label, world_id):
    return EvaluationVideo(env, video_path(config.video_dir, label), world_id=world_id,
                           fps=config.video_fps, width=config.video_width, height=config.video_height,
                           backend=config.video_backend)