"""
Temporal Reasoning Module
"""

import numpy as np
from collections import deque
import yaml

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class _LSTMAttentionNet(nn.Module):
        def __init__(self, input_dim, hidden_size, num_heads):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_dim, hidden_size=hidden_size,
                num_layers=2, batch_first=True,
                dropout=0.2, bidirectional=True)
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size * 2, num_heads=num_heads,
                dropout=0.1, batch_first=True)
            self.fc1     = nn.Linear(hidden_size * 2, 64)
            self.fc2     = nn.Linear(64, 32)
            self.fc3     = nn.Linear(32, 1)
            self.relu    = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            self.dropout = nn.Dropout(0.3)

        def forward(self, x, hidden=None):
            lstm_out, new_hidden = self.lstm(x, hidden)
            attn_out, attn_w = self.attention(lstm_out, lstm_out, lstm_out)
            last = attn_out[:, -1, :]
            out  = self.dropout(self.relu(self.fc1(last)))
            out  = self.dropout(self.relu(self.fc2(out)))
            score = self.sigmoid(self.fc3(out))
            return score, attn_w, new_hidden


class TemporalReasoningModule:
    # Features: [child, fire, cx, cy, vel, proximity]  (pool removed → 6 dims)
    INPUT_DIM = 6

    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        temporal_cfg = self.config.get('temporal', {})
        self.buffer_size  = temporal_cfg.get('frame_buffer_size', 30)
        self.hidden_size  = temporal_cfg.get('lstm_hidden_size',  64)
        self.num_heads    = temporal_cfg.get('attention_heads',    4)

        self.frame_buffer: deque = deque(maxlen=self.buffer_size)
        self.hidden = None

        if TORCH_AVAILABLE:
            self.net = _LSTMAttentionNet(self.INPUT_DIM, self.hidden_size, self.num_heads)
            self.net.eval()
        else:
            self.net = None

    def extract_features(self, detections):
        feat = np.zeros(self.INPUT_DIM, dtype=np.float32)

        feat[0] = 1.0 if detections.get('child') else 0.0
        feat[1] = 1.0 if detections.get('fire')  else 0.0

        children = detections.get('child', [])
        if children:
            cx, cy = children[0]['center']
            feat[2] = cx / 1280.0
            feat[3] = cy / 720.0
            if self.frame_buffer and self.frame_buffer[-1][0] == 1.0:
                feat[4] = float(np.hypot(feat[2] - self.frame_buffer[-1][2],
                                         feat[3] - self.frame_buffer[-1][3]))

        feat[5] = self._proximity_score(detections)
        return feat

    def _proximity_score(self, detections):
        if not detections.get('child'):
            return 0.0
        child_c = np.array(detections['child'][0]['center'])
        diag    = np.hypot(1280, 720)
        best    = 0.0
        for h in detections.get('fire', []):
            d = np.linalg.norm(child_c - np.array(h['center']))
            best = max(best, 1.0 - d / diag)
        return float(best)

    def update_buffer(self, features):
        self.frame_buffer.append(features)

    def forward(self, sequence_tensor):
        if self.net is None or not TORCH_AVAILABLE:
            return 0.0, None

        import torch
        with torch.no_grad():
            if self.hidden is not None:
                self.hidden = tuple(h.detach() for h in self.hidden)
            score, attn_w, self.hidden = self.net(sequence_tensor, self.hidden)
        return float(score.item()), attn_w

    def analyze_temporal_pattern(self, detections):
        features = self.extract_features(detections)
        self.update_buffer(features)

        min_frames = 10
        if len(self.frame_buffer) < min_frames:
            return 0.0, "insufficient_data"

        if self.net is not None and TORCH_AVAILABLE:
            import torch
            seq    = torch.FloatTensor(np.array(self.frame_buffer)).unsqueeze(0)
            risk,_ = self.forward(seq)
        else:
            seq  = np.array(self.frame_buffer)
            risk = float(np.mean(seq[:, 5]) * (1 + float(np.mean(seq[:, 4])) * 5))
            risk = min(1.0, risk)

        pattern = self._classify_pattern(np.array(self.frame_buffer), risk)
        return risk, pattern

    def _classify_pattern(self, sequence, risk_score):
        n = len(sequence)
        if n < 5:
            return "insufficient_data"

        prox = sequence[:, 5]
        half = max(1, n // 2)

        if np.mean(prox[-half:]) > np.mean(prox[:half]):
            return "dangerous_approach" if risk_score > 0.7 else "cautious_approach"

        if np.mean(prox) > 0.6 and np.std(prox) < 0.1:
            return "lingering_near_hazard"

        if n >= 10 and np.mean(sequence[-10:, 4]) > 0.05:
            return "rapid_movement"

        child_frames = sequence[-min(10, n):, 0]
        if np.sum(child_frames) < len(child_frames) / 2:
            return "transient_detection"

        return "stable_safe" if risk_score < 0.3 else "normal_activity"

    def reset_buffer(self):
        self.frame_buffer.clear()
        self.hidden = None

    def get_trajectory_prediction(self):
        if len(self.frame_buffer) < 5:
            return None
        recent = np.array(self.frame_buffer)[-5:]
        positions = recent[:, 2:4]
        if len(positions) >= 2:
            vel = positions[-1] - positions[-2]
            preds, cur = [], positions[-1].copy()
            for _ in range(60):
                cur = cur + vel
                preds.append(cur.copy())
            return np.array(preds)
        return None


class TemporalPatternAnalyzer:
    def __init__(self, config_path='config/config.yaml'):
        self.reasoning_module = TemporalReasoningModule(config_path)
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        temporal_cfg = self.config.get('temporal', {})
        self.temporal_threshold = temporal_cfg.get('temporal_threshold', 0.6)
        self.pattern_history: deque = deque(maxlen=100)

    def analyze(self, detections):
        temporal_risk, pattern_type = self.reasoning_module.analyze_temporal_pattern(detections)
        trajectory = self.reasoning_module.get_trajectory_prediction()
        self.pattern_history.append({'risk': temporal_risk, 'pattern': pattern_type})

        is_hazardous = (
            temporal_risk > self.temporal_threshold
            and pattern_type in ('dangerous_approach', 'lingering_near_hazard')
        )

        return {
            'temporal_risk':  temporal_risk,
            'pattern_type':   pattern_type,
            'is_hazardous':   is_hazardous,
            'trajectory':     trajectory,
            'confidence':     self._confidence(),
        }

    def _confidence(self):
        if len(self.pattern_history) < 10:
            return 0.5
        from collections import Counter
        recent  = [p['pattern'] for p in list(self.pattern_history)[-10:]]
        top_cnt = Counter(recent).most_common(1)[0][1]
        return top_cnt / 10.0

    def reset(self):
        self.reasoning_module.reset_buffer()
        self.pattern_history.clear()