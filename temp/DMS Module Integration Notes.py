# DMS Module Integration Notes

#----------------------------------
#  ADD: controls_queue.py
#----------------------------------

def init_dms(self):
    """Initialize driver monitoring system"""
    from DMS_Module import DMS
    self.dms = DMS(camera_index=0)  # Your Logitech C615
    self.dms.start()
    logging.info("DMS initialized and started")

def process_dms_state(self):
    """Process DMS state in command loop"""
    if hasattr(self, 'dms'):
        state = self.dms.get_latest_state()
        if state and state.alert_level.value >= 2:  # WARNING or CRITICAL
            # Add to command queue for HUD notification
            self.commands.put({
                'type': 'dms_alert',
                'level': state.alert_level.name,
                'attention': state.attention_score,
                'message': self._get_dms_message(state)
            })

def _get_dms_message(self, state):
    """Generate context-aware alert message"""
    if state.microsleep_detected:
        return "MICROSLEEP DETECTED - Pull over safely"
    elif state.looking_at_phone:
        return "Eyes on road - Phone use detected"
    elif state.drowsiness_score > 0.7:
        return "Drowsiness detected - Take a break"
    elif state.distraction_score > 0.7:
        return "Stay focused on the road ahead"
    return "Driver attention required"

#----------------------------------
#  ADD: MVD.py
#----------------------------------
# Add to MVD.py scoring configuration
"dms_events": {
    "microsleep": {
        "base_penalty": -25,
        "duration_multiplier": 2.0
    },
    "distraction_high": {
        "base_penalty": -10,
        "duration_threshold_ms": 2000
    },
    "phone_use": {
        "base_penalty": -15
    },
    "drowsy_driving": {
        "base_penalty": -20,
        "escalation_rate": 1.5
    }
}

def process_dms_event(self, state):
    """Calculate DMS-based score adjustments"""
    penalties = []
    
    if state.microsleep_detected:
        penalties.append(self.config["dms_events"]["microsleep"])
    
    if state.looking_at_phone:
        penalties.append(self.config["dms_events"]["phone_use"])
    
    # Return total penalty
    return sum(p["base_penalty"] for p in penalties)


#----------------------------------
#  ADD: HUD.py
#----------------------------------

# Add to HUD.py
def render_dms_panel(self, display, dms_state):
    """Render DMS status panel"""
    if not dms_state:
        return
    
    # Position on right monitor
    panel_x = self.dim[0] - 300
    panel_y = 100
    
    # Draw attention meter
    meter_width = 200
    meter_height = 20
    attention_pct = dms_state.attention_score
    
    # Background
    pygame.draw.rect(display, (50, 50, 50),
                    (panel_x, panel_y, meter_width, meter_height))
    
    # Fill based on attention
    color = (0, 255, 0) if attention_pct > 0.7 else \
            (255, 255, 0) if attention_pct > 0.4 else \
            (255, 0, 0)
    
    pygame.draw.rect(display, color,
                    (panel_x, panel_y, 
                     int(meter_width * attention_pct), 
                     meter_height))
    
    # Text label
    font = self.panel_fonts['small_label']
    text = font.render(f"Driver Attention: {attention_pct:.0%}", 
                      True, (255, 255, 255))
    display.blit(text, (panel_x, panel_y - 25))
    
    # Alert banner if needed
    if dms_state.alert_level.value >= 2:
        self._show_dms_alert(display, dms_state)