"""
Alert Management System
Handles multi-level alerts, notification delivery, cooldown periods,
and alert escalation logic
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
        names = {
            0: "Silent",
            1: "Gentle",
            2: "Medium",
            3: "Urgent",
            4: "Emergency"
        }
        return names.get(level, "Unknown")


class AlertManager:
    """
    Manages alert generation, notification delivery, and escalation
    """
    
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Alert settings
        self.cooldown_period = self.config['alerts']['cooldown_period']
        self.escalation_time = self.config['alerts']['escalation_time']
        
        # Alert history
        self.alert_history = deque(maxlen=100)
        self.last_alert_time = {}  # Track last alert time for each hazard type
        self.active_alerts = {}  # Currently active alerts
        
        # Notification channels
        self.notification_config = self.config['notification']
        
        # Logging
        self.log_dir = Path('logs/alerts')
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def should_send_alert(self, alert_type, current_level):
        """
        Determine if alert should be sent based on cooldown and previous alerts
        
        Returns:
            should_send: Boolean
            reason: String explaining decision
        """
        # Always send emergency alerts
        if current_level == AlertLevel.EMERGENCY:
            return True, "Emergency alert - no cooldown"
        
        # Check if we're in cooldown period
        if alert_type in self.last_alert_time:
            time_since_last = time.time() - self.last_alert_time[alert_type]
            if time_since_last < self.cooldown_period:
                return False, f"Cooldown active ({self.cooldown_period - int(time_since_last)}s remaining)"
        
        # Check if this is an escalation
        if alert_type in self.active_alerts:
            previous_level = self.active_alerts[alert_type]['level']
            if current_level > previous_level:
                return True, "Alert escalation detected"
        
        return True, "Normal alert conditions met"
    
    def create_alert(self, assessment, detections):
        """
        Create alert from risk assessment
        
        Returns:
            alert: Alert object
        """
        alert_urgency = assessment['alert_urgency']
        
        # Map urgency to alert level
        urgency_to_level = {
            'emergency': AlertLevel.EMERGENCY,
            'urgent': AlertLevel.URGENT,
            'medium': AlertLevel.MEDIUM,
            'gentle': AlertLevel.GENTLE
        }
        
        alert_level = urgency_to_level.get(alert_urgency, AlertLevel.SILENT)
        
        # Determine alert type
        alert_type = self._determine_alert_type(detections)
        
        # Create alert object
        alert = {
            'id': self._generate_alert_id(),
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'level': alert_level,
            'level_name': AlertLevel.get_name(alert_level),
            'risk_score': assessment['risk_score'],
            'risk_level': assessment['risk_level_name'],
            'explanation': assessment['explanation'],
            'recommended_actions': [],
            'detections': detections,
            'acknowledged': False,
            'resolved': False
        }
        
        return alert
    
    def _determine_alert_type(self, detections):
        """Determine the type of hazard causing the alert"""
        if len(detections.get('fire', [])) > 0:
            return 'fire_hazard'
        elif len(detections.get('pool', [])) > 0:
            return 'pool_hazard'
        else:
            return 'general_hazard'
    
    def _generate_alert_id(self):
        """Generate unique alert ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"ALERT_{timestamp}_{len(self.alert_history)}"
    
    def send_alert(self, alert):
        """
        Send alert through configured notification channels
        
        Returns:
            delivery_status: Dictionary of channel delivery results
        """
        alert_type = alert['type']
        alert_level = alert['level']
        
        # Check if we should send
        should_send, reason = self.should_send_alert(alert_type, alert_level)
        
        if not should_send and alert_level < AlertLevel.EMERGENCY:
            print(f"Alert suppressed: {reason}")
            return {'suppressed': True, 'reason': reason}
        
        delivery_status = {}
        
        # Mobile app notification
        if self.notification_config['mobile_app']:
            delivery_status['mobile_app'] = self._send_mobile_notification(alert)
        
        # Push notification
        if self.notification_config['push_notification']:
            delivery_status['push'] = self._send_push_notification(alert)
        
        # Email
        if self.notification_config['email'] and alert_level >= AlertLevel.MEDIUM:
            delivery_status['email'] = self._send_email_notification(alert)
        
        # SMS (for urgent/emergency only)
        if self.notification_config['sms'] and alert_level >= AlertLevel.URGENT:
            delivery_status['sms'] = self._send_sms_notification(alert)
        
        # Update tracking
        self.last_alert_time[alert_type] = time.time()
        self.active_alerts[alert_type] = alert
        
        # Log alert
        self._log_alert(alert, delivery_status)
        
        # Add to history
        self.alert_history.append(alert)
        
        return delivery_status
    
    def _send_mobile_notification(self, alert):
        """Send notification to mobile app"""
        # In real implementation, this would connect to mobile app backend
        notification = {
            'title': f"{alert['level_name']} Alert: {alert['type'].replace('_', ' ').title()}",
            'body': self._format_alert_message(alert),
            'priority': 'high' if alert['level'] >= AlertLevel.URGENT else 'normal',
            'data': {
                'alert_id': alert['id'],
                'type': alert['type'],
                'risk_score': alert['risk_score']
            }
        }
        
        print(f"\n📱 Mobile Notification: {notification['title']}")
        print(f"   {notification['body']}")
        
        return {'success': True, 'notification': notification}
    
    def _send_push_notification(self, alert):
        """Send push notification"""
        # Similar to mobile notification
        print(f"🔔 Push Notification sent for alert {alert['id']}")
        return {'success': True}
    
    def _send_email_notification(self, alert):
        """Send email notification"""
        email_content = {
            'subject': f"[{alert['level_name']}] Baby Safety Alert: {alert['type']}",
            'body': self._format_alert_email(alert),
            'to': 'caregiver@example.com'  # From config in real implementation
        }
        
        print(f"📧 Email sent: {email_content['subject']}")
        return {'success': True, 'email': email_content}
    
    def _send_sms_notification(self, alert):
        """Send SMS notification"""
        sms_content = {
            'to': '+1234567890',  # From config
            'message': f"URGENT: {alert['level_name']} baby safety alert. {self._format_alert_message(alert)}"
        }
        
        print(f"📱 SMS sent: {sms_content['message'][:50]}...")
        return {'success': True, 'sms': sms_content}
    
    def _format_alert_message(self, alert):
        """Format alert message for notifications"""
        explanation = alert['explanation']
        factors = explanation.get('primary_factors', [])
        
        message = f"Risk Level: {alert['risk_level']} (Score: {alert['risk_score']:.2f})"
        
        if factors:
            message += f"\n\nKey Factors:"
            for factor in factors[:3]:  # Top 3 factors
                message += f"\n• {factor}"
        
        return message
    
    def _format_alert_email(self, alert):
        """Format detailed email content"""
        explanation = alert['explanation']
        
        email = f"""
BABY SAFETY MONITORING ALERT
{'='*60}

Alert ID: {alert['id']}
Timestamp: {alert['timestamp']}
Alert Type: {alert['type'].replace('_', ' ').title()}
Alert Level: {alert['level_name']}

RISK ASSESSMENT
Risk Score: {alert['risk_score']:.3f}
Risk Level: {alert['risk_level']}
Confidence: {explanation.get('confidence', 0):.2f}

PRIMARY RISK FACTORS:
"""
        for factor in explanation.get('primary_factors', []):
            email += f"  • {factor}\n"
        
        email += f"""
COMPONENT SCORES:
  • Temporal Component: {explanation.get('temporal_component', 0):.3f}
  • Spatial Component: {explanation.get('spatial_component', 0):.3f}

RECOMMENDED ACTIONS:
"""
        for action in alert.get('recommended_actions', []):
            email += f"  {action}\n"
        
        email += f"""
{'='*60}
This is an automated alert from your Baby Safety Monitoring System.
Please check the mobile app for live video feed and more details.
"""
        return email
    
    def _log_alert(self, alert, delivery_status):
        """Log alert to file"""
        log_entry = {
            'alert': alert,
            'delivery_status': delivery_status,
            'logged_at': datetime.now().isoformat()
        }
        
        # Daily log file
        log_file = self.log_dir / f"alerts_{datetime.now().strftime('%Y%m%d')}.json"
        
        # Append to log file
        logs = []
        if log_file.exists():
            with open(log_file, 'r') as f:
                try:
                    logs = json.load(f)
                except:
                    logs = []
        
        logs.append(log_entry)
        
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
    
    def acknowledge_alert(self, alert_id):
        """Mark alert as acknowledged by caregiver"""
        for alert in self.alert_history:
            if alert['id'] == alert_id:
                alert['acknowledged'] = True
                alert['acknowledged_at'] = datetime.now().isoformat()
                print(f"✓ Alert {alert_id} acknowledged")
                return True
        return False
    
    def resolve_alert(self, alert_id, resolution_note=None):
        """Mark alert as resolved"""
        for alert in self.alert_history:
            if alert['id'] == alert_id:
                alert['resolved'] = True
                alert['resolved_at'] = datetime.now().isoformat()
                if resolution_note:
                    alert['resolution_note'] = resolution_note
                
                # Remove from active alerts
                alert_type = alert['type']
                if alert_type in self.active_alerts:
                    del self.active_alerts[alert_type]
                
                print(f"✓ Alert {alert_id} resolved")
                return True
        return False
    
    def check_escalation(self):
        """Check if any active alerts need escalation"""
        current_time = time.time()
        escalated_alerts = []
        
        for alert_type, alert in list(self.active_alerts.items()):
            if alert['acknowledged']:
                continue
            
            alert_time = datetime.fromisoformat(alert['timestamp']).timestamp()
            time_elapsed = current_time - alert_time
            
            if time_elapsed >= self.escalation_time:
                # Escalate alert
                alert['level'] = min(AlertLevel.EMERGENCY, alert['level'] + 1)
                alert['level_name'] = AlertLevel.get_name(alert['level'])
                alert['escalated'] = True
                alert['escalation_time'] = datetime.now().isoformat()
                
                print(f"\n⚠️  ALERT ESCALATION: {alert['id']}")
                print(f"   Level increased to: {alert['level_name']}")
                
                # Resend with higher priority
                self.send_alert(alert)
                escalated_alerts.append(alert)
        
        return escalated_alerts
    
    def get_active_alerts(self):
        """Get all currently active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_statistics(self):
        """Get alert statistics"""
        total_alerts = len(self.alert_history)
        
        if total_alerts == 0:
            return {
                'total_alerts': 0,
                'by_level': {},
                'by_type': {},
                'acknowledged_rate': 0,
                'resolution_rate': 0
            }
        
        # Count by level
        by_level = {}
        for alert in self.alert_history:
            level_name = alert['level_name']
            by_level[level_name] = by_level.get(level_name, 0) + 1
        
        # Count by type
        by_type = {}
        for alert in self.alert_history:
            alert_type = alert['type']
            by_type[alert_type] = by_type.get(alert_type, 0) + 1
        
        # Rates
        acknowledged = sum(1 for a in self.alert_history if a.get('acknowledged'))
        resolved = sum(1 for a in self.alert_history if a.get('resolved'))
        
        stats = {
            'total_alerts': total_alerts,
            'by_level': by_level,
            'by_type': by_type,
            'acknowledged_rate': acknowledged / total_alerts,
            'resolution_rate': resolved / total_alerts,
            'active_alerts': len(self.active_alerts)
        }
        
        return stats


def main():
    """Test alert management system"""
    print("\n" + "="*60)
    print("ALERT MANAGEMENT SYSTEM TEST")
    print("="*60)
    
    manager = AlertManager()
    
    # Test different alert levels
    test_alerts = [
        {
            'name': 'Gentle Alert',
            'assessment': {
                'alert_urgency': 'gentle',
                'risk_score': 0.4,
                'risk_level_name': 'LOW',
                'explanation': {'primary_factors': ['Child near monitored area'], 'confidence': 0.6}
            }
        },
        {
            'name': 'Medium Alert',
            'assessment': {
                'alert_urgency': 'medium',
                'risk_score': 0.65,
                'risk_level_name': 'MEDIUM',
                'explanation': {'primary_factors': ['Child in warning zone (1.2m from hazard)'], 'confidence': 0.7}
            }
        },
        {
            'name': 'Urgent Alert',
            'assessment': {
                'alert_urgency': 'urgent',
                'risk_score': 0.85,
                'risk_level_name': 'HIGH',
                'explanation': {'primary_factors': ['Child approaching fire', 'Collision predicted in 20 frames'], 'confidence': 0.8}
            }
        },
        {
            'name': 'Emergency Alert',
            'assessment': {
                'alert_urgency': 'emergency',
                'risk_score': 0.95,
                'risk_level_name': 'CRITICAL',
                'explanation': {'primary_factors': ['Child in critical zone (0.3m from pool)', 'Dangerous approach detected'], 'confidence': 0.9}
            }
        }
    ]
    
    for i, test in enumerate(test_alerts):
        print(f"\n{'='*60}")
        print(f"Test {i+1}: {test['name']}")
        print(f"{'='*60}")
        
        detections = {'child': [{}], 'fire': [{}] if i >= 2 else [], 'pool': [{}] if i == 3 else []}
        
        alert = manager.create_alert(test['assessment'], detections)
        delivery_status = manager.send_alert(alert)
        
        print(f"\nDelivery Status: {json.dumps(delivery_status, indent=2)}")
        
        time.sleep(1)  # Small delay between tests
    
    # Test alert statistics
    print(f"\n{'='*60}")
    print("ALERT STATISTICS")
    print(f"{'='*60}")
    stats = manager.get_alert_statistics()
    print(json.dumps(stats, indent=2))
    
    print("\n✓ Alert management system test complete")


if __name__ == "__main__":
    main()