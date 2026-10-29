import sys, runpy
sys.path.insert(0, 'scripts')
from mjrl.app_play import AppPlayer
_original = AppPlayer._person
def inject(self, now_us):
    _original(self, now_us)
    if self._tick == 400:
        self._fsm.set_key('key_left_ctrl', True, now_us)
        self._fsm.set_key('key_1', True, now_us)
        print('[demo] ctrl + 1 down at tick', self._tick, flush=True)
    if self._tick == 405:
        self._fsm.set_key('key_1', False, now_us)
        self._fsm.set_key('key_left_ctrl', False, now_us)
        print('[demo] ctrl + 1 up at tick', self._tick, flush=True)
AppPlayer._person = inject
sys.argv = ['scripts/play.py', '--app', 'out/bundle_example/jumper.app', '--backend', 'native', '--device', 'cpu', '--headless', '--steps', '1800', '--video', '/tmp/jumper-official-dance.mp4', '--video-width', '480', '--video-height', '320', '--speed', '0']
runpy.run_path('scripts/play.py', run_name='__main__')