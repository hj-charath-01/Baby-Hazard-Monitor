"""
Alert Management System
Handles multi-level alerts, notification delivery, cooldown periods,
and alert escalation logic

FIXED BUGS:
- Added fallback defaults when notification config section is missing
- Fixed config key lookup with .get() for safety
- Fixed escalation logic (was modifying dict during iteration)
"""

import time
from datetime import datetime, timedelta
from collections import deque
import json
from pathlib import Path
import yaml


class AlertLevel:
    """Alert level configuration"""
    SILENT = 0
    GENTLE = 1
    MEDIUM = 2
    URGENT = 3
    EMERGENCY = 4

    @staticmethod
    def get_name(level):
        names = {0: "Silent", 1: "Gentle", 2: "Medium", 3: "Urgent", 4: "Emergency"}
        return names.get(level, "Unknown")


class AlertManager:
    """Manages alert generation, notification delivery, and escalation"""

    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Alert settings — use .get() with sane defaults to avoid KeyError
        alerts_cfg = self.config.get('alerts', {})
        self.cooldown_period = alerts_cfg.get('cooldown_period', 30)
        self.escalation_time = alerts_cfg.get('escalation_time', 120)

        # Alert history
        self.alert_history = deque(maxlen=100)
        self.last_alert_time = {}
        self.active_alerts = {}

        # FIX: use .get() so missing 'notification' key doesn't crash
        self.notification_config = self.config.get('notification', {
            'mobile_app': True,
            'push_notification': True,
            'email': False,
            'sms': False,
        })

        # Logging
        self.log_dir = Path('logs/alerts')
        self.log_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Core alert logic
    # ------------------------------------------------------------------

    def should_send_alert(self, alert_type, current_level):
        if current_level == AlertLevel.EMERGENCY:
            return True, "Emergency alert — no cooldown"

        if alert_type in self.last_alert_time:
            elapsed = time.time() - self.last_alert_time[alert_type]
            if elapsed < self.cooldown_period:
                remaining = int(self.cooldown_period - elapsed)
                return False, f"Cooldown active ({remaining}s remaining)"

        if alert_type in self.active_alerts:
            if current_level > self.active_alerts[alert_type]['level']:
                return True, "Alert escalation detected"

        return True, "Normal alert conditions met"

    def create_alert(self, assessment, detections):
        urgency_to_level = {
            'emergency': AlertLevel.EMERGENCY,
            'urgent':    AlertLevel.URGENT,
            'medium':    AlertLevel.MEDIUM,
            'gentle':    AlertLevel.GENTLE,
        }
        alert_level = urgency_to_level.get(assessment.get('alert_urgency', 'gentle'),
                                           AlertLevel.SILENT)
        alert_type = self._determine_alert_type(detections)

        alert = {
            'id':               self._generate_alert_id(),
            'timestamp':        datetime.now().isoformat(),
            'type':             alert_type,
            'level':            alert_level,
            'level_name':       AlertLevel.get_name(alert_level),
            'risk_score':       assessment.get('risk_score', 0.0),
            'risk_level':       assessment.get('risk_level_name', 'UNKNOWN'),
            'explanation':      assessment.get('explanation', {}),
            'recommended_actions': [],
            'detections':       detections,
            'acknowledged':     False,
            'resolved':         False,
        }
        return alert

    def _determine_alert_type(self, detections):
        if len(detections.get('fire', [])) > 0:
            return 'fire_hazard'
        if len(detections.get('pool', [])) > 0:
            return 'pool_hazard'
        return 'general_hazard'

    def _generate_alert_id(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"ALERT_{ts}_{len(self.alert_history)}"

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------

    def send_alert(self, alert):
        alert_type  = alert['type']
        alert_level = alert['level']

        should_send, reason = self.should_send_alert(alert_type, alert_level)
        if not should_send and alert_level < AlertLevel.EMERGENCY:
            print(f"Alert suppressed: {reason}")
            return {'suppressed': True, 'reason': reason}

        delivery_status = {}

        if self.notification_config.get('mobile_app', False):
            delivery_status['mobile_app'] = self._send_mobile_notification(alert)

        if self.notification_config.get('push_notification', False):
            delivery_status['push'] = self._send_push_notification(alert)

        if self.notification_config.get('email', False) and alert_level >= AlertLevel.MEDIUM:
            delivery_status['email'] = self._send_email_notification(alert)

        if self.notification_config.get('sms', False) and alert_level >= AlertLevel.URGENT:
            delivery_status['sms'] = self._send_sms_notification(alert)

        self.last_alert_time[alert_type] = time.time()
        self.active_alerts[alert_type] = alert
        self._log_alert(alert, delivery_status)
        self.alert_history.append(alert)

        return delivery_status

    # ------------------------------------------------------------------
    # Notification channels (stubs — replace with real integrations)
    # ------------------------------------------------------------------

    def _send_mobile_notification(self, alert):
        notification = {
            'title': f"{alert['level_name']} Alert: {alert['type'].replace('_', ' ').title()}",
            'body':  self._format_alert_message(alert),
            'priority': 'high' if alert['level'] >= AlertLevel.URGENT else 'normal',
            'data': {'alert_id': alert['id'], 'type': alert['type'],
                     'risk_score': alert['risk_score']},
        }
        print(f"\nMobile Notification: {notification['title']}")
        print(f"   {notification['body'][:120]}")
        return {'success': True, 'notification': notification}

    def _send_push_notification(self, alert):
        print(f"Push Notification sent for alert {alert['id']}")
        return {'success': True}

    def _send_email_notification(self, alert):
        subject = f"[{alert['level_name']}] Baby Safety Alert: {alert['type']}"
        print(f"Email sent: {subject}")
        return {'success': True}

    def _send_sms_notification(self, alert):
        msg = f"URGENT: {alert['level_name']} baby safety alert. {self._format_alert_message(alert)}"
        print(f"SMS sent: {msg[:60]}...")
        return {'success': True}

    def _format_alert_message(self, alert):
        explanation = alert.get('explanation', {})
        factors = explanation.get('primary_factors', [])
        msg = f"Risk Level: {alert['risk_level']} (Score: {alert['risk_score']:.2f})"
        if factors:
            msg += "\nKey Factors: " + "; ".join(factors[:3])
        return msg

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_alert(self, alert, delivery_status):
        log_entry = {
            'alert': alert,
            'delivery_status': delivery_status,
            'logged_at': datetime.now().isoformat(),
        }
        log_file = self.log_dir / f"alerts_{datetime.now().strftime('%Y%m%d')}.json"
        logs = []
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            except Exception:
                logs = []
        logs.append(log_entry)
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)

    # ------------------------------------------------------------------
    # Acknowledgement / resolution
    # ------------------------------------------------------------------

    def acknowledge_alert(self, alert_id):
        for alert in self.alert_history:
            if alert['id'] == alert_id:
                alert['acknowledged'] = True
                alert['acknowledged_at'] = datetime.now().isoformat()
                print(f"✓ Alert {alert_id} acknowledged")
                return True
        return False

    def resolve_alert(self, alert_id, resolution_note=None):
        for alert in self.alert_history:
            if alert['id'] == alert_id:
                alert['resolved'] = True
                alert['resolved_at'] = datetime.now().isoformat()
                if resolution_note:
                    alert['resolution_note'] = resolution_note
                self.active_alerts.pop(alert['type'], None)
                print(f"✓ Alert {alert_id} resolved")
                return True
        return False

    # ------------------------------------------------------------------
    # Escalation — FIX: iterate over a snapshot to avoid RuntimeError
    # ------------------------------------------------------------------

    def check_escalation(self):
        current_time = time.time()
        escalated = []

        for alert_type, alert in list(self.active_alerts.items()):   # FIX: list()
            if alert.get('acknowledged'):
                continue
            alert_ts = datetime.fromisoformat(alert['timestamp']).timestamp()
            if current_time - alert_ts >= self.escalation_time:
                alert['level'] = min(AlertLevel.EMERGENCY, alert['level'] + 1)
                alert['level_name'] = AlertLevel.get_name(alert['level'])
                alert['escalated'] = True
                alert['escalation_time'] = datetime.now().isoformat()
                print(f"\nALERT ESCALATION: {alert['id']} → {alert['level_name']}")
                self.send_alert(alert)
                escalated.append(alert)

        return escalated

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_active_alerts(self):
        return list(self.active_alerts.values())

    def get_alert_statistics(self):
        total = len(self.alert_history)
        if total == 0:
            return {'total_alerts': 0, 'by_level': {}, 'by_type': {},
                    'acknowledged_rate': 0.0, 'resolution_rate': 0.0, 'active_alerts': 0}

        by_level = {}
        by_type = {}
        for a in self.alert_history:
            by_level[a['level_name']] = by_level.get(a['level_name'], 0) + 1
            by_type[a['type']]        = by_type.get(a['type'], 0)        + 1

        acknowledged = sum(1 for a in self.alert_history if a.get('acknowledged'))
        resolved     = sum(1 for a in self.alert_history if a.get('resolved'))

        return {
            'total_alerts':       total,
            'by_level':           by_level,
            'by_type':            by_type,
            'acknowledged_rate':  acknowledged / total,
            'resolution_rate':    resolved / total,
            'active_alerts':      len(self.active_alerts),
        }