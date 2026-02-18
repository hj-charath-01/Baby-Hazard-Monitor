"""
Context-Aware Risk Assessment Module
Combines temporal patterns, spatial proximity, and environmental context
to provide comprehensive hazard assessment with multi-level risk scoring
"""

import numpy as np
import cv2
import yaml
from enum import Enum
from datetime import datetime


class RiskLevel(Enum):
    """Risk level enumeration"""
    SAFE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class EnvironmentalContext:
    """Environmental context analysis"""
    
    def __init__(self):
        self.time_of_day = None
        self.lighting_conditions = None
        self.number_of_people = 0
        self.supervision_present = False
        
    def analyze_context(self, detections, frame):
        """Analyze environmental context from detections and frame"""
        # Time of day
        current_hour = datetime.now().hour
        if 6 <= current_hour < 12:
            self.time_of_day = 'morning'
        elif 12 <= current_hour < 18:
            self.time_of_day = 'afternoon'
        elif 18 <= current_hour < 22:
            self.time_of_day = 'evening'
        else:
            self.time_of_day = 'night'
        
        # Lighting conditions (from frame brightness)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        mean_brightness = np.mean(gray)
        
        if mean_brightness > 150:
            self.lighting_conditions = 'bright'
        elif mean_brightness > 80:
            self.lighting_conditions = 'normal'
        else:
            self.lighting_conditions = 'dim'
        
        # Count people (adults could supervise)
        # This is simplified - in real system, would use person detector
        self.number_of_people = len(detections.get('child', []))
        
        # Determine if supervision present (would need adult detection)
        self.supervision_present = False  # Simplified
        
        context = {
            'time_of_day': self.time_of_day,
            'lighting': self.lighting_conditions,
            'num_people': self.number_of_people,
            'supervised': self.supervision_present
        }
        
        return context
    
    def get_context_risk_modifier(self):
        """Calculate risk modifier based on environmental context"""
        modifier = 0.0
        
        # Higher risk at night or in dim lighting
        if self.time_of_day == 'night' or self.lighting_conditions == 'dim':
            modifier += 0.1
        
        # Lower risk if supervised
        if self.supervision_present:
            modifier -= 0.2
        
        return max(-0.3, min(0.3, modifier))  # Clamp between -0.3 and 0.3


class RiskAssessmentModule:
    """
    Context-aware risk assessment combining multiple analysis streams
    """
    
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Risk weights from config
        self.weights = self.config['risk']['weights']
        self.thresholds = self.config['risk']['thresholds']
        
        # Environmental context
        self.env_context = EnvironmentalContext()
        
        # Risk history for smoothing
        self.risk_history = []
        self.max_history = 10
        
    def calculate_risk_score(self, temporal_analysis, spatial_analysis, 
                            environmental_context):
        """
        Calculate overall risk score using weighted combination
        
        Args:
            temporal_analysis: Results from temporal reasoning module
            spatial_analysis: Results from spatial analysis module
            environmental_context: Environmental context data
            
        Returns:
            risk_score: Overall risk score (0-1)
        """
        # Extract individual risk components
        temporal_risk = temporal_analysis.get('temporal_risk', 0.0)
        spatial_risk = spatial_analysis.get('spatial_risk', 0.0)
        
        # Environmental context modifier
        context_modifier = self.env_context.get_context_risk_modifier()
        
        # Weighted combination
        base_risk = (
            self.weights['proximity'] * spatial_risk +
            self.weights['temporal_pattern'] * temporal_risk +
            self.weights['environment_context'] * abs(context_modifier)
        )
        
        # Apply context modifier
        adjusted_risk = base_risk + context_modifier
        
        # Clamp to [0, 1]
        risk_score = max(0.0, min(1.0, adjusted_risk))
        
        # Apply temporal smoothing
        self.risk_history.append(risk_score)
        if len(self.risk_history) > self.max_history:
            self.risk_history.pop(0)
        
        # Exponential moving average for smooth transitions
        if len(self.risk_history) > 1:
            alpha = 0.3  # Smoothing factor
            smoothed_risk = alpha * risk_score + (1 - alpha) * np.mean(self.risk_history[:-1])
        else:
            smoothed_risk = risk_score
        
        return smoothed_risk
    
    def classify_risk_level(self, risk_score):
        """
        Classify risk score into discrete risk level
        
        Returns:
            risk_level: RiskLevel enum
            level_name: String name of risk level
        """
        if risk_score >= self.thresholds['critical']:
            return RiskLevel.CRITICAL, 'CRITICAL'
        elif risk_score >= self.thresholds['high']:
            return RiskLevel.HIGH, 'HIGH'
        elif risk_score >= self.thresholds['medium']:
            return RiskLevel.MEDIUM, 'MEDIUM'
        elif risk_score >= self.thresholds['low']:
            return RiskLevel.LOW, 'LOW'
        else:
            return RiskLevel.SAFE, 'SAFE'
    
    def generate_risk_explanation(self, temporal_analysis, spatial_analysis,
                                  risk_score, risk_level_name):
        """
        Generate human-readable explanation of risk assessment
        
        Returns:
            explanation: Dictionary with risk explanation
        """
        factors = []
        
        # Spatial factors
        if spatial_analysis.get('proximity_analysis'):
            prox = spatial_analysis['proximity_analysis']
            if prox['zone'] == 'critical':
                factors.append(f"Child in critical zone ({prox['closest_distance']:.2f}m from hazard)")
            elif prox['zone'] == 'warning':
                factors.append(f"Child in warning zone ({prox['closest_distance']:.2f}m from hazard)")
        
        # Temporal factors
        pattern = temporal_analysis.get('pattern_type', '')
        if 'approach' in pattern:
            factors.append(f"Child showing {pattern} behavior")
        
        # Trajectory factors
        if spatial_analysis.get('collision_warning'):
            collision_time = spatial_analysis.get('collision_time', 0)
            factors.append(f"Collision predicted in {collision_time} frames (~{collision_time/30:.1f}s)")
        
        # Approach pattern
        if spatial_analysis.get('approach_pattern') == 'approaching':
            rate = spatial_analysis.get('approach_rate', 0)
            factors.append(f"Approaching hazard at {rate:.3f} m/frame")
        
        explanation = {
            'risk_score': risk_score,
            'risk_level': risk_level_name,
            'primary_factors': factors,
            'temporal_component': temporal_analysis.get('temporal_risk', 0.0),
            'spatial_component': spatial_analysis.get('spatial_risk', 0.0),
            'confidence': temporal_analysis.get('confidence', 0.5)
        }
        
        return explanation
    
    def assess_comprehensive_risk(self, detections, temporal_analysis, 
                                 spatial_analysis, frame=None):
        """
        Perform comprehensive risk assessment
        
        Returns:
            assessment: Complete risk assessment result
        """
        # Analyze environmental context
        if frame is not None:
            import cv2  # Import here to avoid dependency if not needed
            env_context = self.env_context.analyze_context(detections, frame)
        else:
            env_context = {}
        
        # Calculate overall risk score
        risk_score = self.calculate_risk_score(
            temporal_analysis,
            spatial_analysis,
            env_context
        )
        
        # Classify risk level
        risk_level_enum, risk_level_name = self.classify_risk_level(risk_score)
        
        # Generate explanation
        explanation = self.generate_risk_explanation(
            temporal_analysis,
            spatial_analysis,
            risk_score,
            risk_level_name
        )
        
        # Determine if alert should be triggered
        should_alert = risk_level_enum.value >= RiskLevel.MEDIUM.value
        
        # Alert urgency
        if risk_level_enum == RiskLevel.CRITICAL:
            alert_urgency = 'emergency'
        elif risk_level_enum == RiskLevel.HIGH:
            alert_urgency = 'urgent'
        elif risk_level_enum == RiskLevel.MEDIUM:
            alert_urgency = 'medium'
        else:
            alert_urgency = 'gentle'
        
        assessment = {
            'risk_score': risk_score,
            'risk_level': risk_level_enum,
            'risk_level_name': risk_level_name,
            'should_alert': should_alert,
            'alert_urgency': alert_urgency,
            'explanation': explanation,
            'environmental_context': env_context,
            'timestamp': datetime.now().isoformat()
        }
        
        return assessment
    
    def get_recommended_actions(self, assessment):
        """
        Get recommended actions based on risk assessment
        
        Returns:
            actions: List of recommended actions
        """
        risk_level = assessment['risk_level']
        actions = []
        
        if risk_level == RiskLevel.CRITICAL:
            actions.extend([
                "IMMEDIATE ACTION REQUIRED",
                "Alert all caregivers immediately",
                "Activate emergency response",
                "Sound local alarm",
                "Record incident for review"
            ])
        elif risk_level == RiskLevel.HIGH:
            actions.extend([
                "Alert primary caregiver urgently",
                "Send push notification with video snapshot",
                "Prepare for escalation if situation persists",
                "Monitor closely for next 30 seconds"
            ])
        elif risk_level == RiskLevel.MEDIUM:
            actions.extend([
                "Send notification to caregiver",
                "Continue close monitoring",
                "Log event for review"
            ])
        elif risk_level == RiskLevel.LOW:
            actions.extend([
                "Silent logging",
                "Continue monitoring",
                "No immediate action needed"
            ])
        else:  # SAFE
            actions.append("Normal monitoring")
        
        return actions


def main():
    """Test risk assessment module"""
    print("\n" + "="*60)
    print("CONTEXT-AWARE RISK ASSESSMENT MODULE TEST")
    print("="*60)
    
    assessor = RiskAssessmentModule()
    
    # Test scenarios
    scenarios = [
        {
            'name': 'Critical: Child in pool',
            'temporal': {'temporal_risk': 0.9, 'pattern_type': 'dangerous_approach', 'confidence': 0.8},
            'spatial': {'spatial_risk': 0.95, 'proximity_analysis': {'zone': 'critical', 'closest_distance': 0.3}, 'collision_warning': True, 'collision_time': 15}
        },
        {
            'name': 'High: Approaching fire',
            'temporal': {'temporal_risk': 0.7, 'pattern_type': 'cautious_approach', 'confidence': 0.7},
            'spatial': {'spatial_risk': 0.75, 'proximity_analysis': {'zone': 'warning', 'closest_distance': 0.8}, 'approach_pattern': 'approaching', 'approach_rate': 0.05}
        },
        {
            'name': 'Medium: Lingering near hazard',
            'temporal': {'temporal_risk': 0.5, 'pattern_type': 'lingering_near_hazard', 'confidence': 0.6},
            'spatial': {'spatial_risk': 0.6, 'proximity_analysis': {'zone': 'warning', 'closest_distance': 1.2}}
        },
        {
            'name': 'Low: Normal activity',
            'temporal': {'temporal_risk': 0.2, 'pattern_type': 'normal_activity', 'confidence': 0.7},
            'spatial': {'spatial_risk': 0.3, 'proximity_analysis': {'zone': 'safe', 'closest_distance': 2.5}}
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario['name']}")
        print(f"{'='*60}")
        
        detections = {'child': [{'center': (640, 360)}], 'fire': [], 'pool': []}
        
        assessment = assessor.assess_comprehensive_risk(
            detections,
            scenario['temporal'],
            scenario['spatial']
        )
        
        print(f"Risk Score: {assessment['risk_score']:.3f}")
        print(f"Risk Level: {assessment['risk_level_name']}")
        print(f"Should Alert: {assessment['should_alert']}")
        print(f"Alert Urgency: {assessment['alert_urgency']}")
        print(f"\nExplanation:")
        for factor in assessment['explanation']['primary_factors']:
            print(f"  - {factor}")
        
        print(f"\nRecommended Actions:")
        actions = assessor.get_recommended_actions(assessment)
        for action in actions:
            print(f"  • {action}")
    
    print("\n" + "="*60)
    print("✓ Risk assessment module test complete")
    print("="*60)


if __name__ == "__main__":
    main()