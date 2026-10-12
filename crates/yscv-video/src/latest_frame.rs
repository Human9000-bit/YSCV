//! Bounded, latest-only frame delivery to independent consumers.
//!
//! Publishing replaces one retained frame; slow readers skip intermediate frames.
//! Pixel storage is shared with `Arc`, with no per-publication allocation here.
//! Dropping the last sender wakes readers; each may drain the final frame once.

use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

/// Shared pixels and metadata from a single publication.
#[derive(Debug)]
pub struct FrameSnapshot<T> {
    /// Caller-owned pixel storage; its layout is chosen by the producer.
    pub data: Arc<T>,
    /// Width in pixels, supplied by the producer.
    pub width: usize,
    /// Height in pixels, supplied by the producer.
    pub height: usize,
    /// Monotonic capture time supplied by the producer, not the receive time.
    pub captured_at: Instant,
    /// Wrapping publication counter. Compare for equality, not ordering.
    pub sequence: u64,
}

impl<T> Clone for FrameSnapshot<T> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            ..*self
        }
    }
}

/// A frame channel is closed, or a receive deadline expired.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum FrameChannelError {
    /// No more frames can arrive.
    #[error("frame channel closed")]
    Closed,
    /// No unread frame arrived within the requested duration.
    #[error("frame receive timed out")]
    Timeout,
}

struct Slot<T> {
    latest: Option<FrameSnapshot<T>>,
    sequence: u64,
    closed: bool,
}

struct Shared<T> {
    slot: Mutex<Slot<T>>,
    ready: Condvar,
}

impl<T> Shared<T> {
    fn lock(&self) -> MutexGuard<'_, Slot<T>> {
        // No user code runs under this lock; every mutation leaves a valid slot.
        self.slot.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn close(&self) {
        self.lock().closed = true;
        self.ready.notify_all();
    }
}

struct Publisher<T>(Arc<Shared<T>>);

impl<T> Drop for Publisher<T> {
    fn drop(&mut self) {
        self.0.close();
    }
}

/// Cloneable producer for a one-slot frame broadcast.
pub struct FrameSender<T> {
    publisher: Arc<Publisher<T>>,
}

impl<T> Clone for FrameSender<T> {
    fn clone(&self) -> Self {
        Self {
            publisher: self.publisher.clone(),
        }
    }
}

impl<T> Default for FrameSender<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> FrameSender<T> {
    /// Create an empty channel. The last sender's drop closes it.
    pub fn new() -> Self {
        Self {
            publisher: Arc::new(Publisher(Arc::new(Shared {
                slot: Mutex::new(Slot {
                    latest: None,
                    sequence: 0,
                    closed: false,
                }),
                ready: Condvar::new(),
            }))),
        }
    }

    /// Subscribe independently, including the currently retained frame, if any.
    pub fn subscribe(&self) -> FrameReceiver<T> {
        FrameReceiver {
            shared: self.publisher.0.clone(),
            seen: None,
        }
    }

    /// Replace the retained frame and wake all readers.
    ///
    /// Dimensions describe the payload; this channel does not validate its layout.
    /// Retired payloads are dropped outside the mutex, so their destructors cannot
    /// block readers or deadlock by re-entering the channel.
    pub fn publish(
        &self,
        data: Arc<T>,
        width: usize,
        height: usize,
        captured_at: Instant,
    ) -> Result<(), FrameChannelError> {
        let shared = &self.publisher.0;
        let mut slot = shared.lock();
        if slot.closed {
            return Err(FrameChannelError::Closed);
        }
        slot.sequence = slot.sequence.wrapping_add(1);
        let sequence = slot.sequence;
        let retired = slot.latest.replace(FrameSnapshot {
            data,
            width,
            height,
            captured_at,
            sequence,
        });
        drop(slot);
        shared.ready.notify_all();
        drop(retired);
        Ok(())
    }

    /// Close for all producers and readers. Unread final frames remain available.
    pub fn close(&self) {
        self.publisher.0.close();
    }
}

/// Independent cursor over a frame broadcast; receiving never consumes another
/// reader's frame. Holding a snapshot keeps its pixel buffer alive.
pub struct FrameReceiver<T> {
    shared: Arc<Shared<T>>,
    seen: Option<u64>,
}

impl<T> FrameReceiver<T> {
    /// Wait for an unread frame or channel closure.
    pub fn recv(&mut self) -> Result<FrameSnapshot<T>, FrameChannelError> {
        self.receive(None)
    }

    /// Wait at most `timeout`, including all spurious wakeups. A zero duration
    /// polls once. An available unread frame takes precedence over closure.
    pub fn recv_timeout(
        &mut self,
        timeout: Duration,
    ) -> Result<FrameSnapshot<T>, FrameChannelError> {
        self.receive(Some(timeout))
    }

    fn receive(
        &mut self,
        timeout: Option<Duration>,
    ) -> Result<FrameSnapshot<T>, FrameChannelError> {
        let start = Instant::now();
        let mut slot = self.shared.lock();
        loop {
            if let Some(frame) = &slot.latest
                && self.seen != Some(frame.sequence)
            {
                self.seen = Some(frame.sequence);
                return Ok(frame.clone());
            }
            if slot.closed {
                return Err(FrameChannelError::Closed);
            }
            slot = if let Some(timeout) = timeout {
                let remaining = timeout.saturating_sub(start.elapsed());
                if remaining.is_zero() {
                    return Err(FrameChannelError::Timeout);
                }
                self.shared
                    .ready
                    .wait_timeout(slot, remaining)
                    .unwrap_or_else(PoisonError::into_inner)
                    .0
            } else {
                self.shared
                    .ready
                    .wait(slot)
                    .unwrap_or_else(PoisonError::into_inner)
            };
        }
    }
}

#[cfg(test)]
#[path = "tests/latest_frame.rs"]
mod tests;
