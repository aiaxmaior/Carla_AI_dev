import pyarrow as pa
import pyarrow.compute as pc
from collections import deque

class ArrowTelemetryBuffer:
    """Zero-copy telemetry buffer using PyArrow"""
    
    def __init__(self, max_windows=1000):
        # Define schema for telemetry
        self.schema = pa.schema([
            ('frame_start', pa.int32()),
            ('frame_end', pa.int32()),
            ('avg_speed', pa.float32()),
            ('max_speed', pa.float32()),
            ('avg_mvd_score', pa.float32()),
            ('min_mvd_score', pa.float32()),
            ('total_harsh_events', pa.int8()),
            ('total_lane_violations', pa.int8()),
            ('min_ttc', pa.float32()),
            ('max_jerk', pa.float32()),
            ('max_g_force', pa.float32()),
            # Store compressed formats as binary
            ('compressed_v1', pa.binary()),
            ('compressed_llm', pa.binary()),
            ('timestamp', pa.timestamp('ms'))
        ])
        
        # Use RecordBatchBuilder for streaming appends
        self.batch_builder = []
        self.max_windows = max_windows
        
    def append_raw(self, sensor_data: dict):
        """Direct append from sensors without intermediate pandas"""
        # Compute compressed formats on-the-fly
        compressed = self._compress_formats(sensor_data)
        
        # Build row directly
        row = {
            'frame_start': sensor_data['frame_start'],
            'frame_end': sensor_data['frame_end'],
            'avg_speed': sensor_data['avg_speed'],
            # ... other fields ...
            'compressed_v1': compressed['v1'].encode(),
            'compressed_llm': compressed['llm'].encode(),
            'timestamp': pa.timestamp('ms').now()
        }
        
        self.batch_builder.append(row)
        
        # Flush to RecordBatch periodically
        if len(self.batch_builder) >= 100:
            self._flush_batch()
    
    def _flush_batch(self):
        """Convert accumulated rows to RecordBatch"""
        if not self.batch_builder:
            return
            
        # Create RecordBatch from accumulated data
        batch = pa.RecordBatch.from_pylist(
            self.batch_builder, 
            schema=self.schema
        )
        
        # Could write to Plasma shared memory here
        # Or to memory-mapped file
        self.batch_builder.clear()
        
        return batch
    

class ArrowCircularCache:
    """Memory-mapped circular buffer using Arrow"""
    
    def __init__(self, path='/dev/shm/telemetry.arrow', size_mb=100):
        self.path = path
        self.size_mb = size_mb
        
        # Create memory-mapped file
        self.mmap = pa.memory_map(path, size_mb * 1024 * 1024)
        
        # Track write position
        self.write_pos = 0
        self.read_pos = 0
        
    def write_telemetry(self, sensor_data: dict):
        """Write directly to mmap without intermediate storage"""
        # Compress inline
        compressed = format_as_compressed(sensor_data).encode()
        
        # Write length prefix + data
        length = len(compressed)
        self.mmap.write(struct.pack('>H', length))
        self.mmap.write(compressed)
        
        self.write_pos = self.mmap.tell()
        
    def read_batch(self, n=20):
        """Read n compressed windows for streaming"""
        batch = []
        self.mmap.seek(self.read_pos)
        
        for _ in range(n):
            # Read length prefix
            length_bytes = self.mmap.read(2)
            if not length_bytes:
                break
                
            length = struct.unpack('>H', length_bytes)[0]
            compressed = self.mmap.read(length)
            batch.append(compressed)
            
        self.read_pos = self.mmap.tell()
        return batch
    
class HybridTelemetryStore:
    """Store both columnar (for queries) and compressed (for streaming)"""
    
    def __init__(self):
        # Columnar storage for analytics
        self.table = None
        self.pending_batches = []
        
        # Compressed cache for streaming
        self.compressed_cache = deque(maxlen=1000)
        
    def ingest(self, sensor_data: dict, violations: list):
        """Single ingestion point, dual storage"""
        
        # 1. Generate compressed formats
        compressed = {
            'standard': format_as_compressed(sensor_data),
            'context': format_as_compressed_v2(sensor_data, violations),
            'llm': format_for_llm(sensor_data, violations),
            'adaptive': format_adaptive(sensor_data, violations)
        }
        
        # 2. Create Arrow array directly (no pandas!)
        arrays = [
            pa.array([sensor_data['frame_start']]),
            pa.array([sensor_data['avg_speed']]),
            pa.array([sensor_data['avg_mvd_score']]),
            # Store best compressed format as binary
            pa.array([compressed['adaptive'].encode()])
        ]
        
        # 3. Create RecordBatch
        batch = pa.RecordBatch.from_arrays(
            arrays,
            names=['frame_start', 'speed', 'score', 'compressed']
        )
        
        self.pending_batches.append(batch)
        
        # 4. Also cache compressed for immediate streaming
        self.compressed_cache.append(compressed['adaptive'])
        
        # 5. Periodically merge batches into table
        if len(self.pending_batches) >= 10:
            self._merge_batches()
    
    def get_streaming_batch(self, n=20):
        """Get compressed data for Jetson streaming"""
        return list(self.compressed_cache)[-n:]
    
    def query_analytics(self, start_frame, end_frame):
        """Query columnar data for analytics"""
        if self.table:
            # Use Arrow compute for filtering
            mask = pc.and_(
                pc.greater_equal(self.table['frame_start'], start_frame),
                pc.less_equal(self.table['frame_start'], end_frame)
            )
            return self.table.filter(mask)
        
class MultiScaleWindows:
    """
    Three concurrent windows for different analysis types
    """
    
    # Window definitions (in seconds)
    WINDOWS = {
        'immediate': {
            'duration': 3,      # Last 3 seconds
            'frames': 60,       # At 20 Hz
            'detects': ['harsh_brake', 'collision', 'sudden_swerve'],
            'compression': 'full'  # Need all details
        },
        'tactical': {
            'duration': 30,     # Last 30 seconds  
            'frames': 600,
            'detects': ['lane_weaving', 'tailgating_pattern', 'aggressive_driving'],
            'compression': 'compressed_v2'  # Need violation context
        },
        'strategic': {
            'duration': 300,    # Last 5 minutes
            'frames': 6000,
            'detects': ['fatigue', 'distraction_trend', 'risk_escalation'],
            'compression': 'adaptive'  # Mostly summary, expand on issues
        }
    }