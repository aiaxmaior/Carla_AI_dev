"""
-----------------------------
### WindowProcessor.py
Author: Arjun Joshi
Date: 10.17.2025
-----------------------------
Description:

WindowProcessor module for aggregating DataIngestion data into time windows.
Uses pandas rolling windows to aggregate pre-flagged data from DataIngestion.
No violation detection logic - just statistical aggregation and JSONL export.
"""

# ============================================================================
# PERF CHECK (file-level):
# ============================================================================
# [X] | Role: Analytics / windowing - NOT in hot path (called periodically)
# [X] | Hot-path functions: NONE (called on-demand or end-of-session)
# [ ] |- Heavy allocs in hot path? N/A (not in hot path)
# [X] |- pandas/pyarrow/json/disk/net in hot path? Heavy pandas but NOT in tick loop
# [ ] | Graphics here? No
# [X] | Data produced (tick schema?): Window aggregations (5s windows w/ 2.5s overlap)
# [X] | Storage (Parquet/Arrow/CSV/none): JSONL export
# [ ] | Queue/buffer used?: No (processes DataFrame directly)
# [X] | Session-aware? Timestamps only (no session_id yet)
# [ ] | Debug-only heavy features?: None
# Top 3 perf risks:
# 1. [PERF_OK] pandas operations (rolling, resample) are NOT in tick loop - acceptable
# 2. [PERF_SPLIT] Could add session_id to window metadata for multi-session analysis
# 3. [PERF_OK] JSONL export only on-demand (export_to_jsonl) - acceptable
# ============================================================================

import json
import logging
import pandas as pd
from typing import Dict, List, Any, Optional


class WindowProcessor:
    """
    Aggregates DataIngestion's DataFrame into rolling time windows.
    Uses pandas' native windowing for performance.
    """
    
    def __init__(self, window_duration_s: float = 5.0, overlap_s: float = 2.5):
        """
        Args:
            window_duration_s: Duration of each window
            overlap_s: Overlap between windows (stride = duration - overlap)
        """
        self.window_duration_s = window_duration_s
        self.overlap_s = overlap_s
        self.stride_s = window_duration_s - overlap_s
        
        self._last_processed_idx = 0
        self._completed_windows: List[Dict[str, Any]] = []
        
        logging.info(f"WindowProcessor: {window_duration_s}s windows, {overlap_s}s overlap")
    
    def process_dataframe(self, df: pd.DataFrame, force_final: bool = False) -> List[Dict[str, Any]]:
        """
        Process the DataFrame and return completed windows since last call.
        
        Args:
            df: Current DataFrame from DataIngestion
            force_final: If True, process remaining data even if incomplete window
            
        Returns:
            List of newly completed window dicts
        """
        if df.empty or 'timestamp' not in df.columns:
            return []
        
        # Only process new data
        new_df = df.iloc[self._last_processed_idx:].copy()
        
        if new_df.empty:
            return []
        
        # Set timestamp as index for rolling operations
        new_df.set_index('timestamp', inplace=True)
        
        new_windows = self._create_windows(new_df, force_final)
        
        self._last_processed_idx = len(df)
        self._completed_windows.extend(new_windows)
        
        return new_windows
    
    def _create_windows(self, df: pd.DataFrame, force_final: bool) -> List[Dict[str, Any]]:
        """Create windows using pandas resample or rolling."""
        windows = []
        
        if df.empty:
            return windows
        
        start_time = df.index.min()
        end_time = df.index.max()
        total_duration = end_time - start_time
        
        # Not enough data for even one window
        if total_duration < self.window_duration_s and not force_final:
            return windows
        
        # Generate window boundaries
        window_starts = []
        current_start = start_time
        
        while current_start + self.window_duration_s <= end_time or force_final:
            window_starts.append(current_start)
            current_start += self.stride_s
            
            if force_final and current_start >= end_time:
                break
        
        # Create each window
        for i, win_start in enumerate(window_starts):
            win_end = win_start + self.window_duration_s
            
            # Filter data in window
            window_data = df[(df.index >= win_start) & (df.index < win_end)]
            
            if window_data.empty:
                continue
            
            window_dict = self._aggregate_window(window_data, i, win_start, win_end)
            windows.append(window_dict)
        
        return windows
    
    def _aggregate_window(self, window_df: pd.DataFrame, window_id: int, 
                         start_time: float, end_time: float) -> Dict[str, Any]:
        """
        Aggregate a single window using pandas built-in functions.
        """
        # Numeric columns for aggregation
        numeric_cols = window_df.select_dtypes(include='number').columns
        
        # Core metrics aggregation
        metrics_agg = {}
        for col in ['speed_kmh', 'accel_x', 'accel_y', 'norm_steer', 
                    'norm_throttle', 'norm_brake']:
            if col in window_df.columns:
                metrics_agg[col] = {
                    'mean': float(window_df[col].mean()),
                    'max': float(window_df[col].max()),
                    'min': float(window_df[col].min()),
                    'std': float(window_df[col].std()),
                    'final': float(window_df[col].iloc[-1])
                }
        
        # Score aggregation
        score_cols = [c for c in window_df.columns if c.startswith('score_')]
        scores_agg = {}
        for col in score_cols:
            score_name = col.replace('score_', '')
            scores_agg[score_name] = {
                'mean': float(window_df[col].mean()),
                'min': float(window_df[col].min()),
                'final': float(window_df[col].iloc[-1])
            }
        
        # Predictive indices aggregation
        predictive_agg = {}
        for col in ['p_lane_violation', 'p_collision', 'tlc_s', 'ttc_s']:
            if col in window_df.columns:
                predictive_agg[col] = {
                    'mean': float(window_df[col].mean()),
                    'max': float(window_df[col].max()),
                    'final': float(window_df[col].iloc[-1])
                }
        
        # Count violation flags (already flagged in DataIngestion)
        violation_cols = [c for c in window_df.columns if 
                         any(keyword in c for keyword in ['violation', 'exceeded', 'harsh', 'warning', 'critical', 'poor'])]
        
        violations = {}
        for col in violation_cols:
            if window_df[col].dtype == bool:
                violations[col] = {
                    'count': int(window_df[col].sum()),
                    'occurred': bool(window_df[col].any())
                }
        
        # Extract events that occurred (any True flags)
        events = []
        for col, stats in violations.items():
            if stats['occurred']:
                severity = 'high' if 'critical' in col or 'severe' in col else 'medium'
                events.append({
                    'type': col,
                    'count': stats['count'],
                    'severity': severity
                })
        
        # Environmental context (last values)
        context = {
            'speed_limit': float(window_df['speed_limit'].iloc[-1]) if 'speed_limit' in window_df.columns else 50.0,
            'traffic_light': window_df['traffic_light_state'].iloc[-1] if 'traffic_light_state' in window_df.columns else 'Unknown',
            'at_intersection': bool(window_df['is_at_traffic_light'].iloc[-1]) if 'is_at_traffic_light' in window_df.columns else False
        }
        
        # Build complete window dict
        window = {
            'window_id': window_id,
            'start_time': float(start_time),
            'end_time': float(end_time),
            'duration_s': self.window_duration_s,
            'frame_count': len(window_df),
            
            'metrics': metrics_agg,
            'scores': scores_agg,
            'predictive': predictive_agg,
            'violations': violations,
            'events': events,
            'context': context
        }
        
        return window
    
    def export_to_jsonl(self, filepath: str = "./Session_logs/windows.jsonl"):
        """Export all windows to JSONL."""
        try:
            with open(filepath, 'w') as f:
                for window in self._completed_windows:
                    json.dump(window, f, default=str)
                    f.write('\n')
            
            logging.info(f"Exported {len(self._completed_windows)} windows to {filepath}")
            
        except Exception as e:
            logging.error(f"Error exporting windows: {e}", exc_info=True)
    
    def get_latest_window(self) -> Optional[Dict[str, Any]]:
        """Returns most recent window."""
        return self._completed_windows[-1] if self._completed_windows else None
    
    def clear(self):
        """Reset state."""
        self._completed_windows.clear()
        self._last_processed_idx = 0