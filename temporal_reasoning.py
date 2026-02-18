"""
Temporal Reasoning Module
Implements LSTM-Attention Network for temporal pattern analysis
and false alarm reduction through behavior understanding
"""

import torch
import torch.nn as nn
import numpy as np
from collections import deque
import yaml


class TemporalReasoningModule(nn.Module):
    """
    LSTM-Attention Network for analyzing temporal patterns
    Reduces false alarms by understanding behavior over time
    """
    
    def __init__(self, config_path='config/config.yaml'):
        super(TemporalReasoningModule, self).__init__()
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Temporal parameters
        self.buffer_size = self.config['temporal']['frame_buffer_size']
        self.hidden_size = self.config['temporal']['lstm_hidden_size']
        self.num_heads = self.config['temporal']['attention_heads']
        
        # Feature dimension (from detections)
        # [child_present, fire_present, pool_present, 
        #  child_x, child_y, child_velocity, proximity_score]
        self.input_dim = 7
        
        # Frame buffer for temporal analysis
        self.frame_buffer = deque(maxlen=self.buffer_size)
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size * 2,  # *2 for bidirectional
            num_heads=self.num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Output layers
        self.fc1 = nn.Linear(self.hidden_size * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # Risk score output
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.3)
        
        # Hidden state
        self.hidden = None
        
    def extract_features(self, detections):
        """Extract temporal features from detection results"""
        features = np.zeros(self.input_dim)
        
        # Binary presence indicators
        features[0] = 1.0 if len(detections['child']) > 0 else 0.0
        features[1] = 1.0 if len(detections['fire']) > 0 else 0.0
        features[2] = 1.0 if len(detections['pool']) > 0 else 0.0
        
        # Child position and velocity (if detected)
        if len(detections['child']) > 0:
            child = detections['child'][0]  # Primary child
            center = child['center']
            
            # Normalized position (0-1)
            features[3] = center[0] / 1280.0  # Assuming 1280x720
            features[4] = center[1] / 720.0
            
            # Calculate velocity from previous frame
            if len(self.frame_buffer) > 0:
                prev_features = self.frame_buffer[-1]
                if prev_features[0] == 1.0:  # Child was detected before
                    dx = features[3] - prev_features[3]
                    dy = features[4] - prev_features[4]
                    features[5] = np.sqrt(dx**2 + dy**2)  # Velocity magnitude
        
        # Proximity score (calculated from spatial analysis)
        features[6] = self._calculate_proximity_score(detections)
        
        return features
    
    def _calculate_proximity_score(self, detections):
        """Calculate proximity between child and hazards"""
        if len(detections['child']) == 0:
            return 0.0
        
        child_center = np.array(detections['child'][0]['center'])
        max_proximity = 0.0
        
        # Check proximity to fire
        for fire in detections['fire']:
            fire_center = np.array(fire['center'])
            distance = np.linalg.norm(child_center - fire_center)
            # Normalize by image diagonal
            normalized_dist = distance / np.sqrt(1280**2 + 720**2)
            proximity = 1.0 - normalized_dist
            max_proximity = max(max_proximity, proximity)
        
        # Check proximity to pool
        for pool in detections['pool']:
            pool_center = np.array(pool['center'])
            distance = np.linalg.norm(child_center - pool_center)
            normalized_dist = distance / np.sqrt(1280**2 + 720**2)
            proximity = 1.0 - normalized_dist
            max_proximity = max(max_proximity, proximity)
        
        return max_proximity
    
    def update_buffer(self, features):
        """Add new frame features to temporal buffer"""
        self.frame_buffer.append(features)
    
    def forward(self, sequence):
        """
        Forward pass through temporal reasoning network
        
        Args:
            sequence: Tensor of shape (batch, seq_len, input_dim)
            
        Returns:
            risk_score: Temporal risk assessment (0-1)
        """
        # LSTM processing
        lstm_out, self.hidden = self.lstm(sequence, self.hidden)
        
        # Self-attention over temporal sequence
        attn_out, attn_weights = self.attention(
            lstm_out, lstm_out, lstm_out
        )
        
        # Take the last timestep output
        last_out = attn_out[:, -1, :]
        
        # Fully connected layers
        x = self.relu(self.fc1(last_out))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        risk_score = self.sigmoid(self.fc3(x))
        
        return risk_score, attn_weights
    
    def analyze_temporal_pattern(self, detections):
        """
        Analyze temporal pattern and return risk assessment
        
        Args:
            detections: Current frame detections
            
        Returns:
            temporal_risk: Risk score based on temporal analysis
            pattern_type: Type of detected pattern
        """
        # Extract features
        features = self.extract_features(detections)
        self.update_buffer(features)
        
        # Need minimum frames for analysis
        if len(self.frame_buffer) < 10:
            return 0.0, "insufficient_data"
        
        # Convert buffer to tensor
        sequence = np.array(list(self.frame_buffer))
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
        
        # Get risk score
        with torch.no_grad():
            risk_score, attn_weights = self.forward(sequence_tensor)
        
        temporal_risk = risk_score.item()
        
        # Classify pattern type
        pattern_type = self._classify_pattern(sequence, temporal_risk)
        
        return temporal_risk, pattern_type
    
    def _classify_pattern(self, sequence, risk_score):
        """Classify the type of temporal pattern"""
        # Check for approach pattern (increasing proximity)
        proximity_trend = sequence[-10:, 6]  # Last 10 frames proximity
        
        if np.mean(proximity_trend[-5:]) > np.mean(proximity_trend[:5]):
            if risk_score > 0.7:
                return "dangerous_approach"
            else:
                return "cautious_approach"
        
        # Check for lingering near hazard
        if np.mean(proximity_trend) > 0.6 and np.std(proximity_trend) < 0.1:
            return "lingering_near_hazard"
        
        # Check for rapid movement
        velocity_trend = sequence[-10:, 5]
        if np.mean(velocity_trend) > 0.05:
            return "rapid_movement"
        
        # Check for transient detection (false alarm indicator)
        child_presence = sequence[-10:, 0]
        if np.sum(child_presence) < 5:  # Detected in less than half frames
            return "transient_detection"
        
        # Stable safe situation
        if risk_score < 0.3 and np.std(proximity_trend) < 0.1:
            return "stable_safe"
        
        return "normal_activity"
    
    def reset_buffer(self):
        """Clear temporal buffer"""
        self.frame_buffer.clear()
        self.hidden = None
    
    def get_trajectory_prediction(self):
        """Predict future trajectory based on recent movement"""
        if len(self.frame_buffer) < 5:
            return None
        
        recent_features = np.array(list(self.frame_buffer)[-5:])
        
        # Extract positions
        positions = recent_features[:, 3:5]
        
        # Simple linear extrapolation
        if len(positions) >= 2:
            velocity = positions[-1] - positions[-2]
            
            # Predict next 2 seconds (60 frames)
            predicted_positions = []
            current_pos = positions[-1]
            
            for i in range(60):
                next_pos = current_pos + velocity
                predicted_positions.append(next_pos)
                current_pos = next_pos
            
            return np.array(predicted_positions)
        
        return None


class TemporalPatternAnalyzer:
    """High-level temporal pattern analysis coordinator"""
    
    def __init__(self, config_path='config/config.yaml'):
        self.reasoning_module = TemporalReasoningModule(config_path)
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.temporal_threshold = self.config['temporal']['temporal_threshold']
        
        # Pattern history
        self.pattern_history = deque(maxlen=100)
    
    def analyze(self, detections):
        """
        Perform complete temporal analysis
        
        Returns:
            analysis_result: Dictionary with temporal analysis results
        """
        temporal_risk, pattern_type = self.reasoning_module.analyze_temporal_pattern(
            detections
        )
        
        # Get trajectory prediction
        trajectory = self.reasoning_module.get_trajectory_prediction()
        
        # Record pattern
        self.pattern_history.append({
            'risk': temporal_risk,
            'pattern': pattern_type
        })
        
        # Determine if situation is genuinely hazardous
        is_hazardous = (
            temporal_risk > self.temporal_threshold and
            pattern_type in ['dangerous_approach', 'lingering_near_hazard']
        )
        
        analysis_result = {
            'temporal_risk': temporal_risk,
            'pattern_type': pattern_type,
            'is_hazardous': is_hazardous,
            'trajectory': trajectory,
            'confidence': self._calculate_confidence()
        }
        
        return analysis_result
    
    def _calculate_confidence(self):
        """Calculate confidence in analysis based on consistency"""
        if len(self.pattern_history) < 10:
            return 0.5
        
        recent_patterns = [p['pattern'] for p in list(self.pattern_history)[-10:]]
        
        # Higher confidence if patterns are consistent
        from collections import Counter
        pattern_counts = Counter(recent_patterns)
        most_common_count = pattern_counts.most_common(1)[0][1]
        
        confidence = most_common_count / 10.0
        return confidence
    
    def reset(self):
        """Reset analyzer state"""
        self.reasoning_module.reset_buffer()
        self.pattern_history.clear()


def main():
    """Test temporal reasoning module"""
    print("\n" + "="*60)
    print("TEMPORAL REASONING MODULE TEST")
    print("="*60)
    
    analyzer = TemporalPatternAnalyzer()
    
    # Simulate detection sequence
    print("\nSimulating temporal pattern analysis...")
    
    for i in range(50):
        # Dummy detections
        detections = {
            'child': [{'center': (640 + i*5, 360), 'bbox': [600, 300, 680, 420]}] if i % 3 != 0 else [],
            'fire': [{'center': (800, 400)}] if i > 30 else [],
            'pool': []
        }
        
        result = analyzer.analyze(detections)
        
        if i % 10 == 0:
            print(f"\nFrame {i}:")
            print(f"  Temporal Risk: {result['temporal_risk']:.3f}")
            print(f"  Pattern Type: {result['pattern_type']}")
            print(f"  Is Hazardous: {result['is_hazardous']}")
            print(f"  Confidence: {result['confidence']:.3f}")
    
    print("\n✓ Temporal reasoning module test complete")


if __name__ == "__main__":
    main()