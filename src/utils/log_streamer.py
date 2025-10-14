import logging
import asyncio
from collections import defaultdict
from typing import Dict, Set
from queue import Queue

class LogStreamer:
    def __init__(self):
        self.session_queues: Dict[str, Set[asyncio.Queue]] = defaultdict(set)
        
    def add_client(self, session_id: str) -> asyncio.Queue:
        """Add a client queue for a session"""
        queue = asyncio.Queue(maxsize=100)
        self.session_queues[session_id].add(queue)
        return queue
    
    def remove_client(self, session_id: str, queue: asyncio.Queue):
        """Remove a client queue"""
        if session_id in self.session_queues:
            self.session_queues[session_id].discard(queue)
            if not self.session_queues[session_id]:
                del self.session_queues[session_id]
    
    async def send_log(self, session_id: str, message: str):
        """Send log message to all clients subscribed to this session"""
        if session_id in self.session_queues:
            # Create list of queues to avoid modification during iteration
            queues = list(self.session_queues[session_id])
            for queue in queues:
                try:
                    await asyncio.wait_for(queue.put(message), timeout=1.0)
                except (asyncio.QueueFull, asyncio.TimeoutError):
                    # If queue is full or timeout, remove this client
                    self.remove_client(session_id, queue)

# Global instance
log_streamer = LogStreamer()

class SessionLogHandler(logging.Handler):
    """Custom handler that streams logs to clients"""
    
    def __init__(self, streamer: LogStreamer):
        super().__init__()
        self.streamer = streamer
        
    def emit(self, record):
        try:
            log_entry = self.format(record)
            # Extract session_id from the log record if available
            session_id = getattr(record, 'session_id', None)
            if session_id:
                # Use asyncio to send the log
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(self.streamer.send_log(session_id, log_entry))
                except RuntimeError:
                    pass
        except Exception:
            self.handleError(record)