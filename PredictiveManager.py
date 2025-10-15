import logging
import os
import sys
import pandas as pd
from PredictiveIndices import PredictiveIndices
from DataIngestion import DataIngestion

class PredictiveManager:
    """
    Manages real-time predictive safety calculations by taking the latest
    data frame from the DataIngestion module and feeding it to the
    PredictiveIndices model.
    """
    def __init__(self, data_ingestion_instance):
        self.data_ingestion = data_ingestion_instance
        self.predictive_indices_model = PredictiveIndices()
        self.last_update_frame = -1
        # Update every 10 frames (~3 times per second at 30fps) for efficiency
        self.update_interval = 10
        self.current_indices = {}

    def tick(self, frame_number):
        """Called every frame from the main simulation loop."""
        if frame_number - self.last_update_frame >= self.update_interval:
            self._compute_indices()
            self.last_update_frame = frame_number
            
    def _compute_indices(self):
        """
        Fetches the latest data row, formats it, and runs the predictive model.
        """
        try:
            df = self.data_ingestion.get_dataframe()
            if df is None or df.empty:
                return

            # Get the most recent row of data
            latest_data = df.iloc[-1]

            # Map the DataFrame columns to the 'obs' dictionary expected by PredictiveIndices
            obs_for_predictor = {
                "v": latest_data.get('speed_kmh', 0) / 3.6,  # Convert km/h to m/s
                "ax": latest_data.get('acceleration_x', 0),
                "yaw_rate": latest_data.get('angular_velocity_z', 0),
                "lateral_offset": latest_data.get('distance_from_lane_center', 0),
                "lane_width": latest_data.get('lane_width', 3.6),
                "blinker": {0: None, 1: "left", 2: "right", 3: "hazard"}.get(latest_data.get('blinker_state')),
                "lead_distance": latest_data.get('dist_nearest_vehicle'),
                "lead_rel_speed": latest_data.get('relative_speed_nearest_vehicle'),
            }

            # Run the update with the latest timestamp and observation dictionary
            timestamp = latest_data.get('timestamp', 0.0)
            self.current_indices = self.predictive_indices_model.update(timestamp, obs_for_predictor)

        except Exception as e:
            logging.error(f"Failed to compute predictive indices: {e}", exc_info=False)

            
    def get_indices(self):
        """Returns the most recently computed indices for display or other use."""
        return self.current_indices