import gym
from gym import spaces

class SeismicEventDetectionEnv(gym.Env):
    def __init__(self, signals_array, annotations_array):
        super(SeismicEventDetectionEnv, self).__init__()
        
        # State: STA, LTA, RMS values, detection history
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        
        # Action: Change STA, LTA, Threshold, RMS Scale
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(4,), dtype=np.float32)
        
        # Signals and Annotations
        self.signals_array = signals_array
        self.annotations_array = annotations_array
        self.current_signal_index = 0
    
    def reset(self):
        # Reset environment state, start with a new signal
        self.current_signal = self.signals_array[self.current_signal_index]
        self.current_annotations = self.annotations_array[self.current_signal_index]
        self.current_signal_index = (self.current_signal_index + 1) % len(self.signals_array)
        return self._get_state()

    def step(self, action):
        # Apply action (adjust filtering parameters)
        new_sta, new_lta, new_threshold, new_rms_scale = action
        detections = self._apply_filter(new_sta, new_lta, new_threshold, new_rms_scale)
        
        # Calculate reward based on detections and annotations
        reward = self._calculate_reward(detections)
        done = False  # Define terminal state condition
        
        return self._get_state(), reward, done, {}
    
    def _get_state(self):
        # Return the current state (STA, LTA, RMS values)
        return np.array([self.current_sta, self.current_lta, self.rms_sta, self.rms_lta, self.detection_history])

    def _apply_filter(self, sta, lta, threshold, rms_window, rms_scale):
        # Compute detections using the STA/LTA algorithm
        return compute_sta_lta(self.current_signal, sta, lta, threshold, rms_window, rms_scale)
    
    def _calculate_reward(self, detections):
        # Calculate reward based on how close detections are to annotations
        return reward_function(detections, self.current_annotations)
