import Box from "@mui/material/Box";
import { useGame } from "../store.js";
import { ORANGE, MONO } from "../theme.js";
import { ANTON, CREAM } from "./comic.jsx";

const stateLabel = {
  idle: "STANDBY",
  searching: "SEARCH BALL",
  approaching: "AUTO APPROACH",
  aligning: "ALIGN TARGET",
  "backing-up": "ADJUST RANGE",
  "kick-ready": "KICK WINDOW",
  kicking: "KICK",
  complete: "TARGET CLEARED",
  recovering: "FALL RECOVERY",
  "aligning-foot": "ALIGN FOOT",
  "approaching-foot": "FINAL APPROACH",
  "circling-ball": "ROUTE BEHIND BALL",
  "staging-shot": "SET SHOT LINE",
  "clearing-ball": "CREATE SPACE",
  "aligning-shot": "AIM AT GOAL",
  goal: "GOAL SCORED",
  "manual-override": "MANUAL OVERRIDE",
};

export default function VisionOverlay() {
  const entered = useGame((s) => s.entered);
  const menuOpen = useGame((s) => s.menuOpen);
  const vision = useGame((s) => s.vision);
  const policyBrain = useGame((s) => s.policyBrain);
  if (!entered || menuOpen || !vision.enabled) return null;

  const box = vision.bbox;
  const label = stateLabel[vision.state] ?? String(vision.state).toUpperCase();
  return (
    <Box
      aria-live="polite"
      sx={{ position: "fixed", inset: 0, zIndex: 8, pointerEvents: "none", overflow: "hidden" }}
    >
      <Box
        sx={{
          position: "absolute", top: "5.2rem", left: "1.5rem",
          fontFamily: MONO, color: CREAM, fontSize: "0.7rem", lineHeight: 1.65,
          letterSpacing: "0.08em", textShadow: "0 1px 5px #000",
        }}
      >
        <Box sx={{ fontFamily: ANTON, color: ORANGE, fontSize: "0.95rem", letterSpacing: "0.08em" }}>
          VISION AUTO / {label}
        </Box>
        {vision.distance > 0 ? `${vision.distance.toFixed(2)} M  ${Math.round(vision.bearing * 180 / Math.PI)} DEG` : "NO TARGET"}
        <br />{vision.foot ? `${vision.foot.toUpperCase()} FOOT SELECTED` : "FOOT PENDING"}
        <br />GOAL {vision.goalDistance > 0 ? `${vision.goalDistance.toFixed(2)} M` : "LOCKING"} / AIM {Math.round(vision.shotError * 180 / Math.PI)} DEG
        <br />SCORE {vision.score} {vision.scored ? "/ VERIFIED OVER LINE" : "/ LIVE"}
        <br />CMD {(vision.command?.[0] ?? 0).toFixed(2)} / {(vision.command?.[2] ?? 0).toFixed(2)}
        <br />CAM {vision.cameraMode === "first" ? "1P SENSOR" : vision.cameraMode === "second" ? "2P FOLLOW" : "3P CHASE"}
        <br />BRAIN {policyBrain.name.toUpperCase()} / {policyBrain.status.toUpperCase()}
        {policyBrain.status === "online" && policyBrain.latencyMs > 0
          ? ` / ${policyBrain.latencyMs.toFixed(1)} MS`
          : ""}
      </Box>

      {vision.cameraMode === "first" && (
        <Box sx={{
          position: "absolute", left: "50%", top: "50%", width: 26, height: 26,
          transform: "translate(-50%, -50%)",
          borderLeft: `1px solid ${CREAM}`, borderRight: `1px solid ${CREAM}`,
          "&::before": { content: '""', position: "absolute", left: -7, right: -7, top: 12, borderTop: `1px solid ${CREAM}` },
        }} />
      )}

      {vision.goalTarget && (
        <Box sx={{
          position: "absolute",
          left: `${vision.goalTarget.x * 100}%`, top: `${vision.goalTarget.y * 100}%`,
          width: 34, height: 34, transform: "translate(-50%, -50%)",
          border: `2px solid ${CREAM}`, borderRadius: "50%",
          boxShadow: "0 0 0 1px rgba(0,0,0,0.75), 0 0 18px rgba(255,122,47,0.55)",
          "&::before": { content: '\"\"', position: "absolute", left: -8, right: -8, top: 15, borderTop: `1px solid ${ORANGE}` },
          "&::after": { content: '\"\"', position: "absolute", top: -8, bottom: -8, left: 15, borderLeft: `1px solid ${ORANGE}` },
        }}>
          <Box sx={{
            position: "absolute", top: -22, left: "50%", transform: "translateX(-50%)",
            px: "5px", whiteSpace: "nowrap", background: CREAM, color: "#09090d",
            fontFamily: MONO, fontWeight: 900, fontSize: "0.56rem",
          }}>GOAL TARGET</Box>
        </Box>
      )}

      {box && (
        <Box sx={{
          position: "absolute",
          left: `${box.left * 100}%`, top: `${box.top * 100}%`,
          width: `${(box.right - box.left) * 100}%`, height: `${(box.bottom - box.top) * 100}%`,
          minWidth: 32, minHeight: 32, border: `2px solid ${ORANGE}`,
          boxShadow: "0 0 0 1px rgba(0,0,0,0.7), 0 0 14px rgba(255,122,47,0.35)",
        }}>
          <Box sx={{
            position: "absolute", left: -2, top: -25, height: 23, px: "6px",
            display: "flex", alignItems: "center", background: ORANGE, color: "#09090d",
            fontFamily: MONO, fontWeight: 800, fontSize: "0.62rem", letterSpacing: "0.06em",
          }}>
            BALL {Math.round(vision.confidence * 100)}%
          </Box>
        </Box>
      )}
    </Box>
  );
}
