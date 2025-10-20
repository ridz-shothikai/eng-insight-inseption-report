#src/utils/log_streamer.py

import logging
import asyncio
from collections import defaultdict, deque
from typing import Dict, Set, Deque
from datetime import datetime, timedelta

class LogStreamer:
    def __init__(self, max_history: int = 200, history_ttl_hours: int = 24):
        """
        Initialize the log streamer with history buffer support.
        
        Args:
            max_history: Maximum number of messages to keep per session
            history_ttl_hours: Hours to keep history after last update
        """
        self.session_queues: Dict[str, Set[asyncio.Queue]] = defaultdict(set)
        
        # NEW: History buffers
        self.history: Dict[str, Deque[str]] = {}
        self.history_timestamps: Dict[str, datetime] = {}
        self.max_history = max_history
        self.history_ttl = timedelta(hours=history_ttl_hours)
        
        self.logger = logging.getLogger(__name__)
    
    def add_client(self, session_id: str) -> asyncio.Queue:
        """Add a client queue for a session and replay history"""
        queue = asyncio.Queue(maxsize=100)
        
        # NEW: Initialize history if needed
        if session_id not in self.history:
            self.history[session_id] = deque(maxlen=self.max_history)
            self.history_timestamps[session_id] = datetime.now()
        
        # NEW: Replay history to new client
        history_count = len(self.history[session_id])
        if history_count > 0:
            self.logger.info(f"Replaying {history_count} messages to reconnected client for session {session_id}")
            for historical_message in self.history[session_id]:
                try:
                    queue.put_nowait(historical_message)
                except asyncio.QueueFull:
                    self.logger.warning(f"Queue full during history replay for {session_id}")
                    break
        
        self.session_queues[session_id].add(queue)
        self.logger.info(f"Client connected to {session_id} (total: {len(self.session_queues[session_id])})")
        return queue
    
    def remove_client(self, session_id: str, queue: asyncio.Queue):
        """Remove a client queue"""
        if session_id in self.session_queues:
            self.session_queues[session_id].discard(queue)
            remaining = len(self.session_queues[session_id])
            self.logger.info(f"Client disconnected from {session_id} (remaining: {remaining})")
            
            if not self.session_queues[session_id]:
                del self.session_queues[session_id]
                # Keep history for reconnections
                self.logger.info(f"No more clients for {session_id}, history preserved")
    
    def broadcast(self, session_id: str, message: str):
        """
        Synchronous broadcast method for sending messages from non-async contexts.
        This is used by the markdown streaming callback.
        Now also stores messages in history buffer.
        """
        # NEW: Store in history
        if session_id not in self.history:
            self.history[session_id] = deque(maxlen=self.max_history)
            self.history_timestamps[session_id] = datetime.now()
        
        self.history[session_id].append(message)
        self.history_timestamps[session_id] = datetime.now()
        
        # Broadcast to connected clients
        if session_id in self.session_queues:
            # Get list of queues to avoid modification during iteration
            queues = list(self.session_queues[session_id])
            dead_queues = []
            
            for queue in queues:
                try:
                    # Use put_nowait for synchronous non-blocking put
                    queue.put_nowait(message)
                except asyncio.QueueFull:
                    # If queue is full, mark for removal
                    self.logger.warning(f"Queue full for client in {session_id}")
                    dead_queues.append(queue)
            
            # Remove dead queues
            for queue in dead_queues:
                self.remove_client(session_id, queue)
    
    async def send_log(self, session_id: str, message: str):
        """Send log message to all clients subscribed to this session"""
        # NEW: Store in history
        if session_id not in self.history:
            self.history[session_id] = deque(maxlen=self.max_history)
            self.history_timestamps[session_id] = datetime.now()
        
        self.history[session_id].append(message)
        self.history_timestamps[session_id] = datetime.now()
        
        # Broadcast to connected clients
        if session_id in self.session_queues:
            # Create list of queues to avoid modification during iteration
            queues = list(self.session_queues[session_id])
            dead_queues = []
            
            for queue in queues:
                try:
                    await asyncio.wait_for(queue.put(message), timeout=1.0)
                except (asyncio.QueueFull, asyncio.TimeoutError):
                    # If queue is full or timeout, mark for removal
                    self.logger.warning(f"Queue timeout/full for client in {session_id}")
                    dead_queues.append(queue)
            
            # Remove dead queues
            for queue in dead_queues:
                self.remove_client(session_id, queue)
    
    # NEW METHODS
    
    def clear_history(self, session_id: str):
        """Clear history for a specific session"""
        if session_id in self.history:
            del self.history[session_id]
        if session_id in self.history_timestamps:
            del self.history_timestamps[session_id]
        self.logger.info(f"Cleared history for session {session_id}")
    
    def cleanup_old_history(self):
        """
        Clean up history for old sessions.
        Should be called periodically (e.g., every hour).
        """
        now = datetime.now()
        sessions_to_remove = []
        
        for session_id, timestamp in self.history_timestamps.items():
            # Only clean if no active clients AND history is old
            if (session_id not in self.session_queues and 
                now - timestamp > self.history_ttl):
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            self.clear_history(session_id)
        
        if sessions_to_remove:
            self.logger.info(f"Auto-cleaned {len(sessions_to_remove)} old session histories")
        
        return len(sessions_to_remove)
    
    def get_stats(self) -> dict:
        """Get statistics about current sessions and history"""
        return {
            "active_sessions": len(self.session_queues),
            "total_clients": sum(len(queues) for queues in self.session_queues.values()),
            "sessions_with_history": len(self.history),
            "total_history_messages": sum(len(hist) for hist in self.history.values()),
            "history_size_per_session": {
                session_id: len(hist) 
                for session_id, hist in self.history.items()
            }
        }

# Global instance
log_streamer = LogStreamer(max_history=200, history_ttl_hours=24)


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