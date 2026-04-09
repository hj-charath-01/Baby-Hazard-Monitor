"""
test_patents.py  —  Standalone test harness for all 6 patent claims.
No camera or video file required. Run from project root.
"""

import sys, time, json, shutil
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
HEAD = "\033[1m"
END  = "\033[0m"

def section(title):
    print(f"\n{HEAD}{'='*60}\nPATENT CLAIM: {title}\n{'='*60}{END}")

results = {}

# ─────────────────────────────────────────────────────────────
# CLAIM 1 — Adaptive Alert Cadence
# Novel aspect: cooldown SHORTENS when response latency trends UP
# ─────────────────────────────────────────────────────────────
section("Adaptive Alert Cadence  (adaptive_alert_cadence.py)")

# Clean up any persisted state from previous runs
Path("config/alert_cadence.json").unlink(missing_ok=True)

from adaptive_alert_cadence import AdaptiveAlertCadence, COOLDOWN_MIN, COOLDOWN_MAX

cadence = AdaptiveAlertCadence(base_cooldown=60)

# Simulate a FATIGUED caregiver: inject slow, worsening responses
alert_type = "fire_hazard"
for i, latency in enumerate([15, 25, 40, 65, 90, 120]):   # increasing latency
    aid = f"A{i}"
    cadence.record_sent(aid, alert_type, "urgent")
    cadence._pending[aid].sent_ts -= latency          # fake the sent timestamp
    cadence.record_ack(aid)

cd_after_fatigue, urg_after, meta = cadence.get_params(alert_type)
fatigue_detected = meta["fatigue_detected"]
urgency_escalated = meta["urgency_bump"] > 0

print(f"  Latency trend : increasing (simulated fatigue)")
print(f"  Cooldown after: {cd_after_fatigue:.1f}s  (base was 60s)")
print(f"  Fatigue flag  : {fatigue_detected}")
print(f"  Urgency bump  : {meta['urgency_bump']}")

ok_shorten   = cd_after_fatigue < 60
ok_fatigue   = fatigue_detected
ok_urgency   = urgency_escalated

print(f"  {PASS if ok_shorten  else FAIL} Cooldown shortened when fatigue detected")
print(f"  {PASS if ok_fatigue  else FAIL} Fatigue flag raised")
print(f"  {PASS if ok_urgency  else FAIL} Urgency level escalated")

# Now simulate an ATTENTIVE caregiver on a fresh instance
cadence2 = AdaptiveAlertCadence(base_cooldown=60)
for i, latency in enumerate([30, 25, 20, 15, 12, 10]):    # decreasing latency
    aid = f"B{i}"
    cadence2.record_sent(aid, alert_type, "gentle")
    cadence2._pending[aid].sent_ts -= latency
    cadence2.record_ack(aid)

cd_after_responsive, _, meta2 = cadence2.get_params(alert_type)
ok_lengthen = cd_after_responsive > 60

print(f"\n  Latency trend : decreasing (attentive caregiver)")
print(f"  Cooldown after: {cd_after_responsive:.1f}s  (base was 60s)")
print(f"  {PASS if ok_lengthen else FAIL} Cooldown lengthened for attentive caregiver")

results["adaptive_cadence"] = all([ok_shorten, ok_fatigue, ok_urgency, ok_lengthen])
Path("config/alert_cadence.json").unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────
# CLAIM 2 — Caregiver Attention Estimation
# Novel aspect: risk multiplier changes with gaze direction
# ─────────────────────────────────────────────────────────────
section("Caregiver Attention  (caregiver_attention.py)")

from caregiver_attention import (
    CaregiverAttentionEstimator, AttentionState,
    ATTENTION_RISK_MULTIPLIER
)

estimator = CaregiverAttentionEstimator()
frame = np.zeros((720, 1280, 3), dtype=np.uint8)   # blank frame

def make_person(x1, y1, x2, y2):
    return [{"class_name": "person", "bbox": [x1, y1, x2, y2],
             "confidence": 0.95}]

# ABSENT: no detections at all
r_absent = estimator.estimate(frame, [])
ok_absent = r_absent["state"] == AttentionState.ABSENT
ok_absent_mult = r_absent["risk_multiplier"] >= 1.5
print(f"  Absent caregiver  → multiplier={r_absent['risk_multiplier']:.2f}")
print(f"  {PASS if ok_absent      else FAIL} State correctly classified as ABSENT")
print(f"  {PASS if ok_absent_mult else FAIL} Multiplier ≥ 1.5 (highest risk)")

# WATCHING: large adult bounding box, yaw ~0 (face toward camera)
# We can't force mediapipe's output in a blank frame, but we can test the
# multiplier table contract directly.
mult_watching    = ATTENTION_RISK_MULTIPLIER[AttentionState.WATCHING]
mult_distracted  = ATTENTION_RISK_MULTIPLIER[AttentionState.DISTRACTED]

ok_watching_low  = mult_watching < 1.0
ok_distracted_hi = mult_distracted > 1.0
ok_ordering      = mult_watching < mult_distracted

print(f"\n  WATCHING multiplier   : {mult_watching}")
print(f"  DISTRACTED multiplier : {mult_distracted}")
print(f"  {PASS if ok_watching_low  else FAIL} WATCHING reduces risk (multiplier < 1.0)")
print(f"  {PASS if ok_distracted_hi else FAIL} DISTRACTED increases risk (multiplier > 1.0)")
print(f"  {PASS if ok_ordering      else FAIL} WATCHING < DISTRACTED ordering preserved")

results["caregiver_attention"] = all([ok_absent, ok_absent_mult,
                                       ok_watching_low, ok_distracted_hi, ok_ordering])


# ─────────────────────────────────────────────────────────────
# CLAIM 3 — Developmental Stage & Adaptive Zone Radii
# Novel aspect: zone radii scale with observed mobility stage
# ─────────────────────────────────────────────────────────────
section("Developmental Stage  (developmental_stage.py)")

from developmental_stage import (
    DevelopmentalStageEstimator, DevelopmentalStage, STAGE_ZONE_SCALES
)

base_zones = {"critical": 1.0, "warning": 2.5, "safe": 5.0}

est = DevelopmentalStageEstimator(frame_wh=(1280, 720))

def simulate_stage(aspect_ratios, velocities):
    """Feed synthetic bbox sequences to force a stage classification."""
    e = DevelopmentalStageEstimator(frame_wh=(1280, 720))
    cx, cy = 640, 360
    for ar, vel in zip(aspect_ratios, velocities):
        w = 80
        h = int(w * ar)
        # advance position to simulate velocity
        cx += int(vel * 1280)
        x1, y1 = cx - w//2, cy - h//2
        e.update([x1, y1, x1+w, y1+h])
    return e

# LYING: wide bbox, no movement
e_lying = simulate_stage([0.6]*20, [0.0]*20)
zones_lying = e_lying.get_adapted_zones(base_zones)

# RUNNING: tall bbox, fast movement
e_running = simulate_stage([2.5]*20, [0.09]*20)
zones_running = e_running.get_adapted_zones(base_zones)

ok_lying_smaller   = zones_lying["critical"] < base_zones["critical"]
ok_running_larger  = zones_running["critical"] > base_zones["critical"]
ok_running_gt_lying = zones_running["critical"] > zones_lying["critical"]

print(f"  LYING critical zone   : {zones_lying['critical']:.2f}m  (base 1.0m)")
print(f"  RUNNING critical zone : {zones_running['critical']:.2f}m  (base 1.0m)")
print(f"  {PASS if ok_lying_smaller   else FAIL} Lying → zones contracted below baseline")
print(f"  {PASS if ok_running_larger  else FAIL} Running → zones expanded above baseline")
print(f"  {PASS if ok_running_gt_lying else FAIL} Running zones > lying zones")

# Verify all 5 stages are ordered correctly on critical scale
scales = [STAGE_ZONE_SCALES[s][0] for s in [
    DevelopmentalStage.LYING, DevelopmentalStage.CRAWLING,
    DevelopmentalStage.CRUISING, DevelopmentalStage.WALKING,
    DevelopmentalStage.RUNNING
]]
ok_monotone = all(scales[i] <= scales[i+1] for i in range(len(scales)-1))
print(f"  {PASS if ok_monotone else FAIL} Critical scale monotonically increases: lying→running")

results["developmental_stage"] = all([ok_lying_smaller, ok_running_larger,
                                        ok_running_gt_lying, ok_monotone])


# ─────────────────────────────────────────────────────────────
# CLAIM 4 — Hazard Habituation (cross-session)
# Novel aspect: approach count trend across sessions escalates risk
# ─────────────────────────────────────────────────────────────
section("Hazard Habituation  (hazard_habituation.py)")

HABITUATION_DB_PATH = Path("config/habituation_db.json")
HABITUATION_DB_PATH.unlink(missing_ok=True)

from hazard_habituation import HazardHabituationDetector, HABITUATION_WEIGHT_SCALE

hazard = {"type": "fire", "location": (40, 40), "bbox": [200, 150, 300, 280],
          "confidence": 0.85}

def run_fake_session(n_approaches, session_id_suffix):
    """Simulate a session with n approach events toward the hazard."""
    det = HazardHabituationDetector(approach_distance_m=5.0)
    det._session_id = f"S_TEST_{session_id_suffix}"
    prox = {"zone": "warning", "closest_distance": 1.2,
            "closest_hazard": hazard, "speed": 0.3}
    child_bbox = [220, 170, 260, 280]
    ts = time.time()
    for i in range(n_approaches):
        det._last_approach_ts = {}        # reset cooldown for each synthetic event
        det.observe(child_bbox, prox, [hazard], current_time_ts=ts + i * 35)
    det.close_session()

# Simulate 4 sessions with an INCREASING approach count (habituation pattern)
for idx, count in enumerate([1, 2, 3, 5]):
    run_fake_session(count, idx)

# New session — should now detect habituation
det_final = HazardHabituationDetector(approach_distance_m=5.0)
det_final._session_id = "S_TEST_FINAL"
prox = {"zone": "warning", "closest_distance": 1.0,
        "closest_hazard": hazard, "speed": 0.5}
child_bbox = [220, 170, 260, 280]
ts = time.time()
for i in range(6):
    det_final._last_approach_ts = {}
    result = det_final.observe(child_bbox, prox, [hazard], current_time_ts=ts + i * 35)

habituated = len(result.get("habituated_hazards", [])) > 0

# Escalated confidence
base_conf = 0.85
escalated = det_final.get_escalated_confidence(hazard, base_conf)
ok_escalated = escalated > base_conf
ok_habituation = habituated

report = det_final.get_habituation_report()
print(f"  Sessions simulated: 5 (counts: 1, 2, 3, 5, 6)")
print(f"  Habituation detected: {habituated}")
print(f"  Base confidence: {base_conf:.2f}  →  Escalated: {escalated:.2f}")
print(f"  {PASS if ok_habituation else FAIL} Habituation flag raised after increasing trend")
print(f"  {PASS if ok_escalated   else FAIL} Confidence weight escalated (×{HABITUATION_WEIGHT_SCALE})")

# Also test that a DECREASING trend does NOT flag habituation
HABITUATION_DB_PATH.unlink(missing_ok=True)
for idx, count in enumerate([5, 3, 2, 1]):
    run_fake_session(count, f"DEC{idx}")
det_no_hab = HazardHabituationDetector(approach_distance_m=5.0)
det_no_hab._session_id = "S_TEST_NOHAB"
ts = time.time()
for i in range(1):
    det_no_hab._last_approach_ts = {}
    r = det_no_hab.observe(child_bbox, prox, [hazard], current_time_ts=ts + i * 35)
ok_no_false_positive = len(r.get("habituated_hazards", [])) == 0
print(f"  {PASS if ok_no_false_positive else FAIL} Decreasing trend does NOT trigger habituation")

results["hazard_habituation"] = all([ok_habituation, ok_escalated, ok_no_false_positive])
HABITUATION_DB_PATH.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────
# CLAIM 5 — Adaptive Room Mapping
# Novel aspect: persistent vs temporary hazard classification
# ─────────────────────────────────────────────────────────────
section("Adaptive Room Mapping  (adaptive_room_mapper.py)")

from adaptive_room_mapper import AdaptiveRoomMapper

mapper = AdaptiveRoomMapper()
mapper.learning_duration_frames = 50   # speed up for testing

frame = np.zeros((720, 1280, 3), dtype=np.uint8)

# Feed 50 frames: fire appears in the same spot every frame (persistent)
for i in range(50):
    dets = [{
        "class_name": "fire",
        "bbox": [400, 200, 480, 320],
        "confidence": 0.90,
    }]
    mapper.process_frame_for_mapping(frame, dets)

ok_not_learning = not mapper.learning_mode
ok_persistent   = len(mapper.room_map["persistent_hazards"]) > 0

# Feed a NEW hazard once after learning — should become temporary
mapper.room_map["temporary_hazards"].clear()
mapper._detect_dynamic_hazards([{
    "class_name": "fire",
    "bbox": [100, 100, 150, 180],     # far from the persistent one
    "confidence": 0.85,
}])
ok_temporary = len(mapper.room_map["temporary_hazards"]) > 0

print(f"  Learning phase     : finished after 50 frames")
print(f"  Persistent hazards : {len(mapper.room_map['persistent_hazards'])}")
print(f"  Temporary hazards  : {len(mapper.room_map['temporary_hazards'])}")
print(f"  {PASS if ok_not_learning else FAIL} Learning mode exits after frame limit")
print(f"  {PASS if ok_persistent   else FAIL} Repeatedly-seen hazard classified as persistent")
print(f"  {PASS if ok_temporary    else FAIL} Novel hazard after learning classified as temporary")

# Proximity alert fires when child is close to a persistent hazard
child_bbox = [420, 210, 460, 310]     # inside the persistent hazard bbox
alerts = mapper.get_hazard_proximity_alert(child_bbox)
ok_alert = any(a["distance_meters"] < 1.0 for a in alerts)
print(f"  {PASS if ok_alert else FAIL} Proximity alert fires when child near persistent hazard")

results["adaptive_room_mapping"] = all([ok_not_learning, ok_persistent,
                                          ok_temporary, ok_alert])


# ─────────────────────────────────────────────────────────────
# CLAIM 6 — Privacy-Preserving Edge Processing
# Novel aspect: features extracted, raw frame never stored
# ─────────────────────────────────────────────────────────────
section("Privacy Edge Processing  (privacy_edge_processor.py)")

from privacy_edge_processor import PrivacyPreservingProcessor

proc = PrivacyPreservingProcessor()
frame = np.zeros((720, 1280, 3), dtype=np.uint8)

dets = [{"bbox": [100, 100, 200, 300], "class_name": "child", "confidence": 0.95}]

features = proc.extract_privacy_preserving_features(frame, dets)

# Core privacy assertions
ok_no_pixel_data = "image" not in str(features) and "pixel" not in str(features)
ok_has_bbox      = features["detections"][0]["bbox"] is not None
ok_no_crop       = "crop" not in str(features)

# Bandwidth: features must be << 1 MB
savings = proc.estimate_bandwidth(features)
ok_tiny = savings["features_size_kb"] < 10       # < 10 KB

# Privacy zone masking changes the frame in place
proc.add_privacy_zone("bedroom", [(50, 50), (400, 50), (400, 300), (50, 300)])
test_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 200
masked = proc.mask_privacy_zones(test_frame)
zone_region = masked[100:200, 100:300]
original_region = test_frame[100:200, 100:300]
ok_masked = not np.array_equal(zone_region, original_region)

# Report confirms raw video never stored or transmitted
report = proc.get_privacy_report()
ok_no_raw_store  = report["compliance"]["raw_video_stored"] is False
ok_no_raw_tx     = report["compliance"]["raw_video_transmitted"] is False
ok_feature_only  = report["compliance"]["features_only"] is True

print(f"  Features size : {savings['features_size_kb']:.2f} KB  "
      f"(raw video would be {savings['raw_video_size_mb']:.1f} MB/min)")
print(f"  Reduction factor: {savings['reduction_factor']:.0f}×")
print(f"  {PASS if ok_no_pixel_data else FAIL} No raw pixel data in feature dict")
print(f"  {PASS if ok_has_bbox      else FAIL} Bounding box coordinates retained")
print(f"  {PASS if ok_no_crop       else FAIL} No image crop stored")
print(f"  {PASS if ok_tiny          else FAIL} Features < 10 KB")
print(f"  {PASS if ok_masked        else FAIL} Privacy zone masking alters pixels")
print(f"  {PASS if ok_no_raw_store  else FAIL} raw_video_stored = False")
print(f"  {PASS if ok_no_raw_tx     else FAIL} raw_video_transmitted = False")
print(f"  {PASS if ok_feature_only  else FAIL} features_only = True")

results["privacy_edge"] = all([ok_no_pixel_data, ok_has_bbox, ok_no_crop,
                                 ok_tiny, ok_masked, ok_no_raw_store,
                                 ok_no_raw_tx, ok_feature_only])


# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
print(f"\n{HEAD}{'='*60}\nSUMMARY\n{'='*60}{END}")
all_passed = True
for claim, passed in results.items():
    status = PASS if passed else FAIL
    print(f"  {status}  {claim.replace('_', ' ').title()}")
    if not passed:
        all_passed = False

print()
if all_passed:
    print(f"  {HEAD}All 6 patent claims verified.{END}")
else:
    print(f"  Some claims failed — check output above.")