import test from "node:test";
import assert from "node:assert/strict";

import {
  VISION_KICK_TARGET,
  VisionChaseSource,
  relativeBallMeasurement,
} from "../src/game/controls/vision-chase.js";

test("transforms a world-space ball into the duck frame", () => {
  const m = relativeBallMeasurement([1, 2, Math.PI / 2], [1, 3]);
  assert.ok(Math.abs(m.localX - 1) < 1e-9);
  assert.ok(Math.abs(m.localY) < 1e-9);
  assert.equal(m.visible, true);
});

test("selects the visible-side foot, aligns and dispatches one kick", () => {
  let mode = "walk";
  const actions = [];
  const measurement = {
    active: true,
    visible: true,
    localX: VISION_KICK_TARGET.x,
    localY: VISION_KICK_TARGET.lateral,
    distance: 0.11,
    bearing: Math.atan2(VISION_KICK_TARGET.lateral, VISION_KICK_TARGET.x),
  };
  const source = new VisionChaseSource({
    getMeasurement: () => measurement,
    getVelocityLimits: () => [0.25, -0.2, 1],
    getBusyState: () => mode,
    getManualOverride: () => false,
  });
  source.onAction = (name) => { actions.push(name); if (name === "kickL") mode = "kickL"; };
  source.setEnabled(true);
  for (let i = 0; i < 50; i++) source.poll(1 / 60);
  assert.deepEqual(actions, ["kickL"]);
  assert.equal(source.status.foot, "left");
  assert.equal(source.status.state, "kicking");
  mode = "walk";
  source.poll(1 / 60);
  assert.equal(source.status.state, "complete");
});

test("reacquires the same target after a completed kick", () => {
  let mode = "walk";
  const actions = [];
  const measurement = {
    active: true,
    visible: true,
    localX: VISION_KICK_TARGET.x,
    localY: -VISION_KICK_TARGET.lateral,
    distance: 0.11,
    bearing: Math.atan2(-VISION_KICK_TARGET.lateral, VISION_KICK_TARGET.x),
  };
  const source = new VisionChaseSource({
    getMeasurement: () => measurement,
    getVelocityLimits: () => [0.25, -0.2, 1],
    getBusyState: () => mode,
    getManualOverride: () => false,
  });
  source.onAction = (name) => { actions.push(name); mode = "kickR"; };
  source.setEnabled(true);
  for (let i = 0; i < 8; i++) source.poll(0.1);
  assert.deepEqual(actions, ["kickR"]);
  mode = "walk";
  for (let i = 0; i < 10; i++) source.poll(0.1);
  assert.notEqual(source.status.state, "complete");
  for (let i = 0; i < 8; i++) source.poll(0.1);
  assert.deepEqual(actions, ["kickR", "kickR"]);
});

test("manual input and recovery park the autonomous command", () => {
  let manual = true;
  let mode = "walk";
  const source = new VisionChaseSource({
    getMeasurement: () => ({ active: true, visible: true, localX: 0.4, localY: 0, bearing: 0 }),
    getVelocityLimits: () => [0.25, -0.2, 1],
    getBusyState: () => mode,
    getManualOverride: () => manual,
  });
  source.setEnabled(true);
  source.poll();
  assert.equal(source.status.state, "manual-override");
  assert.deepEqual([...source.command], [0, 0, 0]);
  manual = false;
  mode = "recovery";
  source.poll();
  assert.equal(source.status.state, "recovering");
  assert.deepEqual([...source.command], [0, 0, 0]);
});

test("requests a ball once while the target is absent", () => {
  const actions = [];
  const source = new VisionChaseSource({
    getMeasurement: () => ({ active: false, visible: false }),
    getVelocityLimits: () => [0.25, -0.2, 1],
    getBusyState: () => "walk",
    getManualOverride: () => false,
  });
  source.onAction = (name) => actions.push(name);
  source.setEnabled(true);
  source.poll();
  source.poll();
  assert.deepEqual(actions, ["spawnBall"]);
  assert.equal(source.status.state, "searching");
});
