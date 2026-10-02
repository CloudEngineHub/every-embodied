"""CPU-only checks for the ladder pair's observation and action contract."""

import ast
from pathlib import Path

import numpy as np
import pytest


def _pair_type():
    script = Path(__file__).parents[1] / "scripts/bpu_official_ladder_pair_video.py"
    tree = ast.parse(script.read_text(encoding="utf-8"))
    controller = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "OfficialPair"
    )
    # Isolate the controller from the optional Torch/MuJoCo runtime imports.
    namespace = {"np": np}
    exec(compile(ast.Module(body=[controller], type_ignores=[]), str(script), "exec"), namespace)
    return namespace["OfficialPair"]


class Actor:
    def __init__(self, action):
        self.action = np.full(14, action, dtype=np.float32)
        self.observations = []
        self.closed = False

    def infer(self, observation):
        self.observations.append(observation.copy())
        return self.action.copy()

    def close(self):
        self.closed = True


def test_climber_is_unfiltered_and_not_clipped():
    climber, getup = Actor(2.0), Actor(4.0)
    pair = _pair_type()(climber, getup)
    action, recovering = pair.step(np.zeros(61, dtype=np.float32), False)
    np.testing.assert_array_equal(action, climber.action)
    assert not recovering
    assert len(climber.observations) == 1
    assert not getup.observations


def test_handoff_latches_and_observation_uses_previous_raw_action():
    climber, getup = Actor(2.0), Actor(4.0)
    pair = _pair_type()(climber, getup)
    observation = np.zeros(61, dtype=np.float32)
    pair.step(observation, False)
    action, recovering = pair.step(observation, True)
    expected = np.full(14, 3.4, dtype=np.float32)
    expected[5:9] = 3.0
    np.testing.assert_allclose(action, expected)
    np.testing.assert_array_equal(getup.observations[0][34:48], climber.action)
    assert recovering
    _, recovering = pair.step(observation, False)
    np.testing.assert_array_equal(getup.observations[1][34:48], getup.action)
    assert recovering
    assert len(climber.observations) == 1
    pair.close()
    assert climber.closed and getup.closed


@pytest.mark.parametrize("slot,value", [(48, 1.0), (60, -0.1), (0, np.nan)])
def test_invalid_observation_is_rejected(slot, value):
    pair = _pair_type()(Actor(0.0), Actor(0.0))
    observation = np.zeros(61, dtype=np.float32)
    observation[slot] = value
    with pytest.raises(ValueError, match="observation contract"):
        pair.step(observation, False)


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.1, np.nan])
def test_invalid_filter_scale_is_rejected(alpha):
    with pytest.raises(ValueError, match="alpha scale"):
        _pair_type()(Actor(0.0), Actor(0.0), alpha_scale=alpha)
