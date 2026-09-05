// Visual-servo soccer controller. It consumes detector-shaped ball data plus
// a goal target and drives three layers: route behind the ball, align the
// ball-to-goal ray, then execute the trained one-shot kick policy.

const ZERO = 1e-4;
const REACQUIRE_DELAY_S = 0.9;
const SHOT_STANDOFF = 0.27;
const ORBIT_OFFSET = 0.20;
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const wrapPi = (a) => Math.atan2(Math.sin(a), Math.cos(a));

export const VISION_KICK_TARGET = Object.freeze({ x: 0.095, lateral: 0.043 });

export function relativePointMeasurement(robotPose, point) {
  const [rx, ry, yaw] = robotPose;
  const dx = point[0] - rx;
  const dy = point[1] - ry;
  const c = Math.cos(yaw);
  const s = Math.sin(yaw);
  const localX = c * dx + s * dy;
  const localY = -s * dx + c * dy;
  return {
    localX,
    localY,
    distance: Math.hypot(localX, localY),
    bearing: Math.atan2(localY, localX),
  };
}

export function relativeBallMeasurement(robotPose, ballPosition, halfFov = 0.62) {
  const relative = relativePointMeasurement(robotPose, ballPosition);
  return {
    active: true,
    visible: relative.localX > 0.025 && Math.abs(relative.bearing) <= halfFov,
    ...relative,
  };
}

export function relativeShotMeasurement(
  robotPose,
  ballPosition,
  goalTarget,
  { scored = false, halfFov = 0.62 } = {},
) {
  const ball = relativeBallMeasurement(robotPose, ballPosition, halfFov);
  const gx = goalTarget[0] - ballPosition[0];
  const gy = goalTarget[1] - ballPosition[1];
  const goalDistance = Math.max(1e-6, Math.hypot(gx, gy));
  const dx = gx / goalDistance;
  const dy = gy / goalDistance;
  const px = -dy;
  const py = dx;
  const rbx = robotPose[0] - ballPosition[0];
  const rby = robotPose[1] - ballPosition[1];
  const robotAlong = rbx * dx + rby * dy;
  const robotLateral = rbx * px + rby * py;
  const stagingPoint = [
    ballPosition[0] - dx * SHOT_STANDOFF,
    ballPosition[1] - dy * SHOT_STANDOFF,
  ];
  const staging = relativePointMeasurement(robotPose, stagingPoint);
  const sideWaypoints = {
    left: [stagingPoint[0] + px * ORBIT_OFFSET, stagingPoint[1] + py * ORBIT_OFFSET],
    right: [stagingPoint[0] - px * ORBIT_OFFSET, stagingPoint[1] - py * ORBIT_OFFSET],
  };
  const shotYaw = Math.atan2(dy, dx);
  return {
    ...ball,
    scored,
    robotPose: [...robotPose],
    ballPosition: [...ballPosition],
    goalTarget: [...goalTarget],
    goalDistance,
    shotPlan: {
      direction: [dx, dy],
      shotYaw,
      yawError: wrapPi(shotYaw - robotPose[2]),
      robotAlong,
      robotLateral,
      behindBall: robotAlong < -0.14,
      stagingPoint,
      staging,
      sideWaypoints,
      ready: robotAlong < -0.14 && Math.abs(robotLateral) < 0.11 && staging.distance < 0.13,
    },
  };
}

export class VisionChaseSource {
  id = "vision";
  connected = true;
  command = new Float32Array(3);
  axes = { jaw: 0, orbitX: 0, orbitY: 0 };
  pressed = {};
  onAction = () => {};

  #enabled = false;
  #getMeasurement;
  #getVelocityLimits;
  #getBusyState;
  #getManualOverride;
  #foot = null;
  #state = "idle";
  #stableS = 0;
  #kickIssued = false;
  #kickSeen = false;
  #settling = false;
  #spawnRequested = false;
  #measurement = null;
  #drivePhase = "align";
  #completeS = 0;
  #repeatEnabled = true;
  #routeSide = null;
  #staged = false;
  #goal = false;
  #clearing = false;

  constructor({ getMeasurement, getVelocityLimits, getBusyState, getManualOverride }) {
    this.#getMeasurement = getMeasurement;
    this.#getVelocityLimits = getVelocityLimits;
    this.#getBusyState = getBusyState;
    this.#getManualOverride = getManualOverride;
  }

  init() {}
  dispose() {}
  isActive() { return this.#enabled; }
  get enabled() { return this.#enabled; }
  setRepeatEnabled(value) { this.#repeatEnabled = !!value; }
  markGoal() {
    this.command.fill(0);
    this.#goal = true;
    this.#state = "goal";
  }
  get status() {
    return {
      enabled: this.#enabled,
      state: this.#state,
      foot: this.#foot,
      measurement: this.#measurement,
      command: [...this.command],
    };
  }

  setEnabled(value) {
    const next = !!value;
    if (next === this.#enabled) return;
    this.#enabled = next;
    this.reset();
  }

  reset() {
    this.command.fill(0);
    this.#state = this.#enabled ? "searching" : "idle";
    this.#foot = null;
    this.#stableS = 0;
    this.#kickIssued = false;
    this.#kickSeen = false;
    this.#settling = false;
    this.#spawnRequested = false;
    this.#measurement = null;
    this.#drivePhase = "align";
    this.#completeS = 0;
    this.#routeSide = null;
    this.#staged = false;
    this.#goal = false;
    this.#clearing = false;
  }

  #driveTo(target, state, velocityLimits) {
    const [forward, , angular] = velocityLimits;
    const turnLimit = Math.min(angular, 1);
    this.#state = state;
    if (Math.abs(target.bearing) > 0.22) {
      this.command[2] = Math.sign(target.bearing || 1) * turnLimit;
      return;
    }
    this.command[0] = Math.min(forward, 0.25);
    this.command[2] = clamp(target.bearing * 0.45, -0.09, 0.09);
  }

  poll(dt = 1 / 60) {
    this.command.fill(0);
    if (!this.#enabled) return;
    if (this.#goal) {
      this.#state = "goal";
      return;
    }

    const busy = this.#getBusyState?.() ?? "walk";
    if (busy === "recovery") {
      this.#state = "recovering";
      return;
    }
    if (busy === "kickL" || busy === "kickR") {
      this.#kickSeen = true;
      this.#state = "kicking";
      return;
    }
    if (this.#kickIssued && this.#kickSeen && busy === "walk") {
      this.#state = "complete";
      this.#completeS += Math.min(dt, 0.1);
      if (this.#repeatEnabled && this.#completeS >= REACQUIRE_DELAY_S) this.reset();
      return;
    }
    if (this.#getManualOverride?.()) {
      this.#state = "manual-override";
      return;
    }

    const m = this.#getMeasurement?.() ?? null;
    this.#measurement = m;
    if (m?.scored) {
      this.markGoal();
      return;
    }
    if (!m?.active) {
      this.#state = "searching";
      if (!this.#spawnRequested) {
        this.#spawnRequested = true;
        this.onAction("spawnBall", { automatic: true });
      }
      this.command[2] = 0.42;
      return;
    }
    this.#spawnRequested = false;
    const [forward, back, angular] = this.#getVelocityLimits?.() ?? [0.25, -0.2, 1];
    const turnLimit = Math.min(angular, 1);

    const plan = m.shotPlan;
    // Turning in place is only approximate for the learned gait and can
    // translate the body. If that drift invalidates the staging pose, route
    // behind the ball again instead of spinning farther away indefinitely.
    if (plan && this.#staged &&
        (m.distance > 0.46 || plan.robotAlong > -0.12 || Math.abs(plan.robotLateral) > 0.18)) {
      this.#staged = false;
      this.#routeSide = null;
    }
    if (plan && !this.#staged) {
      // A failed approach can leave the duck almost touching the ball. Move
      // radially away first; trying to turn toward an orbit waypoint at this
      // range sweeps a foot through the ball and destroys the shot setup.
      if (m.distance < 0.27) this.#clearing = true;
      if (this.#clearing && m.distance < 0.38) {
        this.#state = "clearing-ball";
        this.command[0] = Math.abs(m.bearing) < Math.PI / 2
          ? back
          : forward;
        return;
      }
      this.#clearing = false;
      if (plan.ready) {
        this.#staged = true;
      } else {
        if (!this.#routeSide) this.#routeSide = plan.robotLateral >= 0 ? "left" : "right";
        const targetPoint = plan.behindBall
          ? plan.stagingPoint
          : plan.sideWaypoints[this.#routeSide];
        const target = relativePointMeasurement(m.robotPose, targetPoint);
        this.#driveTo(target, plan.behindBall ? "staging-shot" : "circling-ball",
          [forward, back, angular]);
        return;
      }
    }
    if (plan && this.#staged && m.distance > 0.15 && Math.abs(plan.yawError) > 0.35) {
      this.#state = "aligning-shot";
      this.command[2] = Math.sign(plan.yawError || 1) * turnLimit;
      return;
    }

    if (!this.#kickIssued && m.distance < 0.24) {
      this.#foot = m.localY > 0.006 ? "left" : "right";
    }
    const targetY = this.#foot === "left"
      ? VISION_KICK_TARGET.lateral
      : -VISION_KICK_TARGET.lateral;
    const angleError = wrapPi(m.bearing);
    const xError = m.localX - VISION_KICK_TARGET.x;
    const yError = m.localY - targetY;

    if (!m.visible) {
      this.#state = "searching";
      this.command[2] = clamp(m.bearing * 1.8, -0.55, 0.55);
      if (Math.abs(this.command[2]) < 0.16) this.command[2] = m.bearing >= 0 ? 0.16 : -0.16;
      return;
    }
    if (this.#foot && m.localX < 0.17 && Math.abs(yError) > 0.018) {
      const desiredBearing = Math.atan2(targetY, Math.max(0.06, m.localX));
      const footAngleError = wrapPi(m.bearing - desiredBearing);
      this.#state = "aligning-foot";
      this.command[2] = Math.sign(footAngleError || 1) * turnLimit;
      return;
    }

    const aligned = Math.abs(angleError) < 0.55;
    const inKickWindow = this.#foot &&
      m.localX > 0.07 && m.localX < 0.122 && Math.abs(yError) < 0.03;
    if (aligned && inKickWindow) this.#settling = true;
    const inSettleWindow = this.#foot && aligned &&
      m.localX > 0.065 && m.localX < 0.15 && Math.abs(yError) < 0.05;
    if (this.#settling && inSettleWindow) {
      this.#state = "kick-ready";
      this.#stableS += Math.min(dt, 0.1);
      if (this.#stableS >= 0.55 && !this.#kickIssued) {
        this.#kickIssued = true;
        this.onAction(this.#foot === "left" ? "kickL" : "kickR", { automatic: true });
      }
      return;
    }
    if (this.#settling) this.#settling = false;
    this.#stableS = 0;

    if (this.#foot && m.localX < 0.17 && Math.abs(yError) <= 0.03) {
      this.#state = "approaching-foot";
      this.command[0] = Math.min(forward, 0.11);
      return;
    }

    if (Math.abs(angleError) > 0.26) this.#drivePhase = "align";
    if (this.#drivePhase === "align" && Math.abs(angleError) < 0.13) this.#drivePhase = "advance";
    if (this.#drivePhase === "align") {
      this.#state = "aligning";
      this.command[2] = Math.sign(angleError || 1) * turnLimit;
      return;
    }
    if (xError > 0.018) {
      const cap = xError > 0.16 ? 0.25 : xError > 0.07 ? 0.18 : 0.11;
      this.command[0] = Math.min(forward, cap, 0.055 + xError * 1.15);
      this.command[2] = clamp(angleError * 0.35, -0.08, 0.08);
      this.#state = "approaching";
    } else if (xError < -0.018) {
      this.command[0] = Math.max(back, xError * 0.65);
      this.#state = "backing-up";
    } else {
      this.#state = "aligning";
    }
    if (Math.abs(this.command[0]) < ZERO) this.command[0] = 0;
    if (Math.abs(this.command[2]) < ZERO) this.command[2] = 0;
  }
}
