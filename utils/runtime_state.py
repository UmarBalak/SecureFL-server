from datetime import datetime

class RuntimeState:
    def __init__(self):
        self.last_aggregation_timestamp = 0
        self.latest_version = 0
        self.last_checked_timestamp = 0
    
    def update_checked_timestamp(self):
        self.last_checked_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

runtime_state = RuntimeState()