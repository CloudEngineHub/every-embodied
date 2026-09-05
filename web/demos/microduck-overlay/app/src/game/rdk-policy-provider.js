const SLOT_TO_RDK = Object.freeze({
  walk: "walking",
  stand: "stand",
  sitstand: "sitstand",
  roll: "roll",
  kickL: "kick_left",
  kickR: "kick_right",
  groundpick: "groundpick",
  drive: "drive",
  crouch: "crouch",
});

export const RDK_REQUIRED_POLICIES = Object.freeze(Object.values(SLOT_TO_RDK));

export function requestedRdkBridge() {
  const raw = new URLSearchParams(window.location.search).get("rdk");
  if (!raw) return null;
  if (raw === "self") return window.location.origin;
  return raw.replace(/\/$/, "");
}

export class RdkPolicyProvider {
  static async connect(baseUrl, required = RDK_REQUIRED_POLICIES) {
    const response = await fetch(`${baseUrl}/catalog`, { cache: "no-store" });
    if (!response.ok) throw new Error(`RDK catalog HTTP ${response.status}`);
    const catalog = await response.json();
    if (catalog.protocol !== "MDP2") throw new Error(`unsupported RDK protocol: ${catalog.protocol}`);
    const missing = required.filter((name) => !catalog.policies?.[name]);
    if (missing.length) throw new Error(`RDK missing policies: ${missing.join(", ")}`);
    for (const name of required) {
      const contract = catalog.policies[name];
      if (contract.observations !== 61 || contract.actions !== 14) {
        throw new Error(`RDK ${name} contract must be [1,61] -> [1,14]`);
      }
    }
    return new RdkPolicyProvider(baseUrl, catalog);
  }

  constructor(baseUrl, catalog) {
    this.baseUrl = baseUrl;
    this.catalog = catalog;
    this.lastLatencyMs = 0;
  }

  async run(slot, observation) {
    const policy = SLOT_TO_RDK[slot];
    if (!policy) throw new Error(`no RDK policy mapping for slot: ${slot}`);
    const body = observation instanceof Float32Array
      ? observation.slice().buffer
      : Float32Array.from(observation).buffer;
    const started = performance.now();
    const response = await fetch(`${this.baseUrl}/infer/${encodeURIComponent(policy)}`, {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body,
      cache: "no-store",
    });
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(`RDK ${policy} inference HTTP ${response.status}: ${detail}`);
    }
    const payload = await response.arrayBuffer();
    if (payload.byteLength !== 14 * 4) {
      throw new Error(`RDK ${policy} returned ${payload.byteLength} bytes, expected 56`);
    }
    this.lastLatencyMs = performance.now() - started;
    const actions = new Float32Array(payload.slice(0));
    if (!actions.every(Number.isFinite)) throw new Error(`RDK ${policy} returned NaN or Inf`);
    return { policy, actions, latencyMs: this.lastLatencyMs };
  }
}
