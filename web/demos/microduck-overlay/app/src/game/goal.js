import * as THREE from "three";

// One source of truth for the MuJoCo colliders, Three.js dressing, visual
// target and scoring gate. MuJoCo is Z-up; Three.js maps (x, y, z) to
// (x, z, -y).
export const GOAL = Object.freeze({
  lineX: 1.18,
  backX: 1.44,
  halfWidth: 0.40,
  height: 0.36,
  postRadius: 0.025,
  targetY: 0,
  targetZ: 0.14,
});

export function appendGoalPhysics({ doc, worldbody, el }) {
  const post = (name, y) => worldbody.appendChild(el("geom", {
    name, type: "cylinder",
    pos: `${GOAL.lineX} ${y} ${GOAL.height / 2}`,
    size: `${GOAL.postRadius} ${GOAL.height / 2}`,
    friction: "0.8 0.01 0.001",
  }));
  post("goal_post_left", GOAL.halfWidth);
  post("goal_post_right", -GOAL.halfWidth);
  worldbody.appendChild(el("geom", {
    name: "goal_crossbar", type: "cylinder",
    pos: `${GOAL.lineX} 0 ${GOAL.height}`,
    size: `${GOAL.postRadius} ${GOAL.halfWidth}`,
    euler: "90 0 0", friction: "0.8 0.01 0.001",
  }));

  // Thin physical back and side nets keep a scored ball in the goal. They
  // are boxes because MuJoCo cloth would add needless state and cost here.
  const depth = GOAL.backX - GOAL.lineX;
  worldbody.appendChild(el("geom", {
    name: "goal_back_net", type: "box",
    pos: `${GOAL.backX} 0 ${GOAL.height / 2}`,
    size: `0.012 ${GOAL.halfWidth} ${GOAL.height / 2}`,
    friction: "0.35 0.01 0.003",
  }));
  for (const [name, y] of [["goal_side_left", GOAL.halfWidth], ["goal_side_right", -GOAL.halfWidth]]) {
    worldbody.appendChild(el("geom", {
      name, type: "box",
      pos: `${GOAL.lineX + depth / 2} ${y} ${GOAL.height / 2}`,
      size: `${depth / 2} 0.012 ${GOAL.height / 2}`,
      friction: "0.35 0.01 0.003",
    }));
  }
}

export function goalCrossing(previousX, ballPosition, radius) {
  const [x, y, z] = ballPosition;
  const wholeBallPastLine = previousX - radius <= GOAL.lineX && x - radius > GOAL.lineX;
  const betweenPosts = Math.abs(y) + radius < GOAL.halfWidth - GOAL.postRadius;
  const belowBar = z + radius < GOAL.height - GOAL.postRadius;
  return wholeBallPastLine && betweenPosts && belowBar;
}

function lineSegments(points, material) {
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  return new THREE.LineSegments(geometry, material);
}

export function createGoalVisual() {
  const group = new THREE.Group();
  group.name = "precision-goal";
  const frameMat = new THREE.MeshStandardMaterial({
    color: 0xfaf8f2, roughness: 0.52, metalness: 0.08,
  });
  const netMat = new THREE.LineBasicMaterial({
    color: 0x9fabb5, transparent: true, opacity: 0.42,
  });
  const targetMat = new THREE.LineBasicMaterial({
    color: 0xff7a2f, transparent: true, opacity: 0.92,
  });
  const cylinder = (radius, length) => new THREE.Mesh(
    new THREE.CylinderGeometry(radius, radius, length, 18), frameMat,
  );

  for (const y of [-GOAL.halfWidth, GOAL.halfWidth]) {
    const post = cylinder(GOAL.postRadius, GOAL.height);
    post.position.set(GOAL.lineX, GOAL.height / 2, -y);
    post.castShadow = true;
    group.add(post);
  }
  const bar = cylinder(GOAL.postRadius, GOAL.halfWidth * 2);
  bar.rotation.x = Math.PI / 2;
  bar.position.set(GOAL.lineX, GOAL.height, 0);
  bar.castShadow = true;
  group.add(bar);

  const net = [];
  const x0 = GOAL.lineX, x1 = GOAL.backX;
  const z0 = -GOAL.halfWidth, z1 = GOAL.halfWidth;
  for (let i = 0; i <= 8; i++) {
    const z = z0 + (z1 - z0) * i / 8;
    net.push(new THREE.Vector3(x1, 0, z), new THREE.Vector3(x1, GOAL.height, z));
  }
  for (let i = 0; i <= 5; i++) {
    const y = GOAL.height * i / 5;
    net.push(new THREE.Vector3(x1, y, z0), new THREE.Vector3(x1, y, z1));
  }
  for (const z of [z0, z1]) {
    for (let i = 0; i <= 5; i++) {
      const y = GOAL.height * i / 5;
      net.push(new THREE.Vector3(x0, y, z), new THREE.Vector3(x1, y, z));
    }
    for (let i = 0; i <= 4; i++) {
      const x = x0 + (x1 - x0) * i / 4;
      net.push(new THREE.Vector3(x, 0, z), new THREE.Vector3(x, GOAL.height, z));
    }
  }
  group.add(lineSegments(net, netMat));

  // Physical aim point: this is the target consumed by the visual servo,
  // shown in-world so the screen reticle has an auditable geometric source.
  const target = new THREE.Group();
  target.position.set(GOAL.lineX + 0.006, GOAL.targetZ, -GOAL.targetY);
  const ring = new THREE.Mesh(
    new THREE.TorusGeometry(0.075, 0.005, 8, 32),
    new THREE.MeshBasicMaterial({ color: 0xff7a2f }),
  );
  ring.rotation.y = Math.PI / 2;
  target.add(ring);
  const cross = [
    new THREE.Vector3(0, -0.095, 0), new THREE.Vector3(0, 0.095, 0),
    new THREE.Vector3(0, 0, -0.095), new THREE.Vector3(0, 0, 0.095),
  ];
  target.add(lineSegments(cross, targetMat));
  group.add(target);
  return { group, target };
}
