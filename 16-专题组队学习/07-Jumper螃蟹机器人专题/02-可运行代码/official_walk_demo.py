import sys, runpy, csv
sys.path.insert(0, 'scripts')
from mjrl.app_play import AppPlayer
_original = AppPlayer._person
samples = []
def inject(self, now_us):
    _original(self, now_us)
    if self._tick == 400:
        self._fsm.set_key('key_w', True, now_us)
        print('[demo] key_w down at tick', self._tick, flush=True)
    if self._tick == 1800:
        self._fsm.set_key('key_w', False, now_us)
        print('[demo] key_w up at tick', self._tick, flush=True)
    if self._tick % 10 == 0:
        data = self._robot.data
        p = data.root_link_pos_w[0]
        v = data.root_link_lin_vel_w[0]
        samples.append([now_us / 1e6, *[float(x) for x in p], *[float(x) for x in v], self._fsm.mode, self._fsm.running_policy])
AppPlayer._person = inject
sys.argv = ['scripts/play.py', '--app', 'out/bundle_example/jumper.app', '--backend', 'native', '--device', 'cpu', '--headless', '--steps', '2000', '--video', '/tmp/jumper-official-walk.mp4', '--video-width', '640', '--video-height', '480', '--no-video-follow', '--video-distance', '3', '--speed', '0']
try:
    runpy.run_path('scripts/play.py', run_name='__main__')
finally:
    with open('/tmp/jumper-official-walk.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['time_s', 'world_x_m', 'world_y_m', 'world_z_m', 'world_vx_m_s', 'world_vy_m_s', 'world_vz_m_s', 'mode', 'running_policy'])
        w.writerows(samples)