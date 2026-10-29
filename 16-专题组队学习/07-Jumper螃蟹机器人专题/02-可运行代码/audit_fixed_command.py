"""Audit the command manager during a Jumper fixed-command replay (upstream b2e5866)."""
import runpy
import sys

import mjrl.replay as replay

original = replay.FixedCommandEvaluation.update

def update(self, dones):
    if self.steps in (0, 49, 99):
        print('[audit] step', self.steps, 'command', self.env.command_manager.get_command('twist')[0].tolist(), flush=True)
    return original(self, dones)

replay.FixedCommandEvaluation.update = update
sys.path.insert(0, 'scripts')
sys.argv = ['scripts/play.py', '--task', 'jumper.tripod', '--backend', 'native', '--device', 'cpu', '--checkpoint', sys.argv[1], '--headless', '--steps', '100', '--physics-hz', '200', '--command', '0.3', '0', '0', '--evaluation-out', '/tmp/jumper-command-audit.json', '--speed', '0']
runpy.run_path('scripts/play.py', run_name='__main__')
