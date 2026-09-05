// Vision-guided ball approach controller. The browser demo feeds this source
// a detector-shaped measurement produced by projecting the MuJoCo ball into
// the first-person camera. A real detector can replace getMeasurement without
// changing the search/approach/align/kick state machine.

const ZERO = 1e-4;
const REACQUIRE_DELAY_S = 0.9;
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const wrapPi = (a) => Math.atan2(Math.sin(a), Math.cos(a));

export const VISION_KICK_TARGET = Object.freeze({ x: 0.095, lateral: 0.043 });

export function relativeBallMeasurement(robotPose, ballPosition, halfFov = 0.62) {
  const [rx, ry, yaw] = robotPose;
  const [bx, by] = ballPosition;
  const dx = bx - rx;
  const dy = by - ry;
  const c = Math.cos(yaw);
  const s = Math.sin(yaw);
  const localX = c * dx + s * dy;
  const localY = -s * dx + c * dy;
  const bearing = Math.atan2(localY, localX);
  return {
    active: true,
    visible: localX > 0.025 && Math.abs(bearing) <= halfFov,
    localX,
    localY,
    distance: Math.hypot(localX, localY),
    bearing,
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
  get status() {
    return {
      enabled: this.#enabled,
      state: this.#state,
      foot: this.#foot,
      measurement: this.#measurement,
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
  }

  poll(dt = 1 / 60) {
    this.command.fill(0);
    if (!this.#enabled) return;

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
      if (this.#repeatEnabled && this.#completeS >= REACQUIRE_DELAY_S) {
        // Keep the ball untouched and begin a new perception-control cycle.
        // This turns AUTO into continuous pursuit instead of a one-shot demo.
        this.reset();
      }
      return;
    }
    if (this.#getManualOverride?.()) {
      this.#state = "manual-override";
      return;
    }

    const m = this.#getMeasurement?.() ?? null;
    this.#measurement = m;
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

    // Choose the kicking foot only after the final approach. The apparent
    // side can change while turning, so committing at first sight is brittle.
    if (!this.#kickIssued && m.distance < 0.24) {
      this.#foot = m.localY > 0.006 ? "left" : "right";
    }
    const targetY = this.#foot === "left"
      ? VISION_KICK_TARGET.lateral
      : -VISION_KICK_TARGET.lateral;
    // Navigation aims the body centreline at the ball. Foot selection is a
    // separate terminal decision; steering toward a foot-specific bearing
    // leaves this gait with a large steady-state yaw error.
    const angleError = wrapPi(m.bearing);
    const xError = m.localX - VISION_KICK_TARGET.x;
    const yError = m.localY - targetY;
    const [forward, back, angular] = this.#getVelocityLimits?.() ?? [0.25, -0.2, 1];
    const turnLimit = Math.min(angular, 1);

    if (!m.visible) {
      this.#state = "searching";
      this.command[2] = clamp(m.bearing * 1.8, -0.55, 0.55);
      if (Math.abs(this.command[2]) < 0.16) this.command[2] = m.bearing >= 0 ? 0.16 : -0.16;
      return;
    }

    // At toe range, pivot until the ball sits in front of the chosen foot.
    // This is the terminal visual-servo stage: the walking policy has no
    // lateral command, so trying to correct y while advancing only pushes the
    // ball along the body centreline and misses both kicking arcs.
    if (this.#foot && m.localX < 0.17 && Math.abs(yError) > 0.018) {
      const desiredBearing = Math.atan2(targetY, Math.max(0.06, m.localX));
      const footAngleError = wrapPi(m.bearing - desiredBearing);
      this.#state = "aligning-foot";
      this.command[2] = Math.sign(footAngleError || 1) * turnLimit;
      return;
    }

    // The kick policies tolerate a wider lateral window than the walking
    // policy can reliably servo to. Select the nearer foot and trigger once
    // the ball is safely ahead of the toes; the trained kick performs the
    // final contact correction.
    const aligned = Math.abs(angleError) < 0.55;
    const inKickWindow = this.#foot &&
      m.localX > 0.07 && m.localX < 0.122 && Math.abs(yError) < 0.03;
    if (aligned && inKickWindow) this.#settling = true;
    const inSettleWindow = this.#foot && aligned &&
      m.localX > 0.065 && m.localX < 0.15 && Math.abs(yError) < 0.05;
    if (this.#settling && inSettleWindow) {
      this.#state = "kick-ready";
      this.#stableS += Math.min(dt, 0.1);
      // Let the locomotion policy return to a balanced stance before the
      // one-shot policy swap. Launching directly from a gait phase can put
      // the selected foot behind the ball even when the geometric window is
      // correct.
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

    // Discrete turn/advance phases are intentional. The locomotion network
    // tracks either command well, but a simultaneous low-speed arc can settle
    // into a nearly static pose. Hysteresis prevents rapid phase chatter.
    if (Math.abs(angleError) > 0.26) this.#drivePhase = "align";
    if (this.#drivePhase === "align" && Math.abs(angleError) < 0.13) {
      this.#drivePhase = "advance";
    }
    if (this.#drivePhase === "align") {
      this.#state = "aligning";
      this.command[2] = Math.sign(angleError || 1) * turnLimit;
      return;
    }

    // Turn mostly in place for a large bearing error, then advance slowly
    // enough for the 50 Hz walking policy to settle inside the kick window.
    const turnScale = clamp(1 - Math.abs(angleError) / 0.62, 0.08, 1);
    if (xError > 0.018) {
      const cap = xError > 0.16 ? 0.25 : xError > 0.07 ? 0.18 : 0.11;
      this.command[0] = Math.min(forward, cap, 0.055 + xError * 1.15);
      // Only a small trim is allowed during forward gait; large corrections
      // are handled by the next align phase.
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
