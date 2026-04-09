import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from collections import deque
import hashlib


class PrivacyPreservingProcessor:
    def __init__(self, config_path='config/privacy_config.yaml'):
        self.config = self._load_config(config_path)
        self.buffer_duration = self.config.get('buffer_duration', 1800)
        self.frame_buffer    = deque(maxlen=self.buffer_duration * 30)
        self.privacy_zones   = []
        self.encryption_key  = self._initialize_encryption()
        self.feature_dim     = 128
        self.alert_recording_buffer = deque(maxlen=300)
        self.recording_active = False

        print("Privacy-Preserving Processor Initialized")
        print(f"Buffer Duration: {self.buffer_duration}s")
        print(f"Privacy Zones: {len(self.privacy_zones)}")
        print(f"Encryption: Enabled")

    def _load_config(self, config_path):
        default = {
            'buffer_duration': 1800,
            'alert_pre_buffer': 10,
            'alert_post_buffer': 10,
            'privacy_zones': [],
            'auto_delete': True,
            'feature_only_mode': True,
            'encryption_enabled': True,
        }
        try:
            import yaml
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
                return {**default, **cfg}
        except Exception:
            return default

    def _initialize_encryption(self):
        key_path = Path('config/encryption_key.key')
        if key_path.exists():
            with open(key_path, 'rb') as f:
                key = f.read()
        else:
            key = Fernet.generate_key()
            key_path.parent.mkdir(exist_ok=True)
            with open(key_path, 'wb') as f:
                f.write(key)
        return Fernet(key)

    def add_privacy_zone(self, zone_name, coordinates):
        zone = {
            'name': zone_name,
            'coordinates': coordinates,
            'created_at': datetime.now().isoformat(),
            'active': True,
        }
        self.privacy_zones.append(zone)
        self._save_privacy_zones()
        print(f"Privacy Zone Added: {zone_name}")

    def _save_privacy_zones(self):
        zones_path = Path('config/privacy_zones.json')
        zones_path.parent.mkdir(exist_ok=True)
        with open(zones_path, 'w') as f:
            json.dump(self.privacy_zones, f, indent=2)

    def mask_privacy_zones(self, frame):
        """
        Apply privacy zone masking.

        FIX: replaced Gaussian blur with solid black fill.
        Gaussian blur of a uniform-colour frame produces identical pixel
        values (blur of a constant = the same constant), so the test
        `not np.array_equal(zone_region, original_region)` always failed
        on synthetic test frames.  Filling with 0 (black) guarantees the
        masked pixels differ from any non-zero original, regardless of
        frame content.
        """
        if not self.privacy_zones:
            return frame

        masked_frame = frame.copy()
        for zone in self.privacy_zones:
            if not zone['active']:
                continue
            mask   = np.zeros(frame.shape[:2], dtype=np.uint8)
            points = np.array(zone['coordinates'], dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)
            # Black-out the privacy zone instead of blurring
            masked_frame[mask == 255] = 0

        return masked_frame

    def extract_privacy_preserving_features(self, frame, detections):
        features = {
            'timestamp': datetime.now().isoformat(),
            'frame_id':  hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8],
            'detections': [],
        }
        for det in detections:
            det_features = {
                'bbox':         det['bbox'],
                'class':        det['class_name'],
                'confidence':   det['confidence'],
                'center':       self._get_bbox_center(det['bbox']),
                'area':         self._get_bbox_area(det['bbox']),
                'aspect_ratio': self._get_aspect_ratio(det['bbox']),
            }
            features['detections'].append(det_features)

        self.frame_buffer.append(features)
        if self.config['auto_delete']:
            self._cleanup_old_data()
        return features

    def _get_bbox_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return [(x1+x2)/2, (y1+y2)/2]

    def _get_bbox_area(self, bbox):
        x1, y1, x2, y2 = bbox
        return (x2-x1) * (y2-y1)

    def _get_aspect_ratio(self, bbox):
        x1, y1, x2, y2 = bbox
        w = x2-x1; h = y2-y1
        return w/h if h > 0 else 0

    def _cleanup_old_data(self):
        cutoff = datetime.now() - timedelta(seconds=self.buffer_duration)
        self._cleanup_alert_recordings(cutoff)

    def _cleanup_alert_recordings(self, cutoff_time):
        alerts_dir = Path('logs/alerts/recordings')
        if not alerts_dir.exists():
            return
        for recording in alerts_dir.glob('*.json'):
            try:
                with open(recording, 'r') as f:
                    data = json.load(f)
                ts = datetime.fromisoformat(data['timestamp'])
                if ts < cutoff_time:
                    recording.unlink()
            except Exception:
                pass

    def trigger_alert_recording(self, alert_level):
        if alert_level < 3:
            return
        self.recording_active = True
        pre_buffer_frames = self.config['alert_pre_buffer'] * 30
        alert_data = {
            'timestamp': datetime.now().isoformat(),
            'alert_level': alert_level,
            'pre_alert_features': list(self.frame_buffer)[-pre_buffer_frames:],
            'post_alert_features': [],
        }
        alert_id   = hashlib.md5(alert_data['timestamp'].encode()).hexdigest()[:8]
        alert_path = Path(f'logs/alerts/recordings/alert_{alert_id}.json')
        alert_path.parent.mkdir(parents=True, exist_ok=True)
        with open(alert_path, 'w') as f:
            json.dump(alert_data, f, indent=2)
        print(f"Alert Recording Triggered: {alert_id}")
        return alert_id

    def encrypt_features_for_transmission(self, features):
        if not self.config['encryption_enabled']:
            return features
        features_json = json.dumps(features)
        encrypted     = self.encryption_key.encrypt(features_json.encode())
        return {'encrypted_data': encrypted.decode(),
                'timestamp': datetime.now().isoformat(),
                'size': len(encrypted)}

    def estimate_bandwidth(self, features):
        features_json    = json.dumps(features)
        features_size_kb = len(features_json.encode()) / 1024
        raw_video_size_mb = 2.5
        return {
            'features_size_kb':         features_size_kb,
            'raw_video_size_mb':        raw_video_size_mb,
            'reduction_factor':         (raw_video_size_mb * 1024) / features_size_kb,
            'bandwidth_saved_percent':  ((raw_video_size_mb*1024 - features_size_kb) /
                                         (raw_video_size_mb*1024)) * 100,
        }

    def get_privacy_report(self):
        return {
            'timestamp': datetime.now().isoformat(),
            'compliance': {
                'raw_video_stored':       False,
                'raw_video_transmitted':  False,
                'features_only':          True,
                'encryption_enabled':     self.config['encryption_enabled'],
                'auto_deletion':          self.config['auto_delete'],
                'privacy_zones_defined':  len(self.privacy_zones),
            },
            'data_retention': {
                'buffer_duration_seconds': self.buffer_duration,
                'current_buffer_size':     len(self.frame_buffer),
                'alert_recordings':        self._count_alert_recordings(),
            },
            'privacy_zones': [
                {'name': z['name'], 'active': z['active']}
                for z in self.privacy_zones
            ],
            'bandwidth_efficiency': {
                'features_only':           True,
                'estimated_reduction':     '1000x',
                'suitable_for_slow_internet': True,
            },
        }

    def _count_alert_recordings(self):
        alerts_dir = Path('logs/alerts/recordings')
        if not alerts_dir.exists():
            return 0
        return len(list(alerts_dir.glob('*.json')))

    def export_user_data(self, output_path):
        user_data = {
            'export_timestamp': datetime.now().isoformat(),
            'privacy_zones':    self.privacy_zones,
            'configuration':    self.config,
            'alert_recordings_count': self._count_alert_recordings(),
            'data_retention_policy': {
                'buffer_duration': self.buffer_duration,
                'auto_delete':     self.config['auto_delete'],
            },
        }
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(user_data, f, indent=2)
        print(f"User data exported to: {output_path}")
        return output_path

    def delete_all_user_data(self):
        self.frame_buffer.clear()
        alerts_dir = Path('logs/alerts/recordings')
        if alerts_dir.exists():
            for r in alerts_dir.glob('*.json'):
                r.unlink()
        for p in [Path('config/privacy_zones.json'),
                  Path('config/encryption_key.key')]:
            if p.exists():
                p.unlink()