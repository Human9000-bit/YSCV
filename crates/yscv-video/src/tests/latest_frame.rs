use super::*;
use std::sync::mpsc;
use std::thread;

type TestResult = Result<(), Box<dyn std::error::Error>>;

#[test]
fn readers_skip_independently_and_share_storage() -> TestResult {
    let sender = FrameSender::new();
    let mut fast = sender.subscribe();
    let mut slow = sender.subscribe();
    let pixels = Arc::new(vec![7; 12]);
    let captured_at = Instant::now();
    sender.publish(pixels.clone(), 2, 2, captured_at)?;
    let first = fast.recv()?;
    assert!(Arc::ptr_eq(&first.data, &pixels));
    assert_eq!(
        (first.width, first.height, first.captured_at),
        (2, 2, captured_at)
    );
    sender.publish(Arc::new(vec![8; 3]), 1, 1, captured_at)?;
    let second = slow.recv()?;
    assert_eq!(*second.data, vec![8; 3]);
    assert_eq!(fast.recv()?.sequence, second.sequence);
    assert!(matches!(
        fast.recv_timeout(Duration::ZERO),
        Err(FrameChannelError::Timeout)
    ));
    assert_eq!(sender.subscribe().recv()?.sequence, second.sequence);
    assert_eq!(*first.data, vec![7; 12]);
    Ok(())
}

#[test]
fn final_frame_drains_then_last_sender_drop_closes() -> TestResult {
    let sender = FrameSender::new();
    let mut reader = sender.subscribe();
    let clone = sender.clone();
    drop(sender);
    assert!(matches!(
        reader.recv_timeout(Duration::ZERO),
        Err(FrameChannelError::Timeout)
    ));
    clone.publish(Arc::new(42), 1, 1, Instant::now())?;
    drop(clone);
    assert_eq!(*reader.recv()?.data, 42);
    assert!(matches!(reader.recv(), Err(FrameChannelError::Closed)));
    Ok(())
}

#[test]
fn close_wakes_all_waiters_and_rejects_publish() -> TestResult {
    let sender = FrameSender::<u8>::new();
    let (done, results) = mpsc::channel();
    thread::scope(|scope| -> TestResult {
        for _ in 0..4 {
            let mut reader = sender.subscribe();
            let done = done.clone();
            scope.spawn(move || {
                let _ = done.send(reader.recv());
            });
        }
        sender.close();
        for _ in 0..4 {
            assert!(matches!(
                results.recv_timeout(Duration::from_secs(2))?,
                Err(FrameChannelError::Closed)
            ));
        }
        assert_eq!(
            sender.publish(Arc::new(1), 1, 1, Instant::now()),
            Err(FrameChannelError::Closed)
        );
        Ok(())
    })
}

#[test]
fn publication_wakes_independent_waiters() -> TestResult {
    let sender = FrameSender::new();
    let (done, results) = mpsc::channel();
    thread::scope(|scope| -> TestResult {
        for _ in 0..4 {
            let mut reader = sender.subscribe();
            let done = done.clone();
            scope.spawn(move || {
                let _ = done.send(reader.recv_timeout(Duration::from_secs(2)));
            });
        }
        sender.publish(Arc::new(7), 1, 1, Instant::now())?;
        for _ in 0..4 {
            assert_eq!(*results.recv_timeout(Duration::from_secs(3))??.data, 7);
        }
        Ok(())
    })
}

#[test]
fn spurious_wakeups_do_not_restart_timeout() -> TestResult {
    let sender = FrameSender::<u8>::new();
    let mut reader = sender.subscribe();
    let (stop, stopped) = mpsc::channel();
    thread::scope(|scope| -> TestResult {
        scope.spawn(move || {
            let start = Instant::now();
            while start.elapsed() < Duration::from_secs(1) {
                sender.publisher.0.ready.notify_all();
                if stopped.recv_timeout(Duration::from_millis(2)).is_ok() {
                    break;
                }
            }
        });
        let start = Instant::now();
        let result = reader.recv_timeout(Duration::from_millis(30));
        let elapsed = start.elapsed();
        let _ = stop.send(());
        assert!(matches!(result, Err(FrameChannelError::Timeout)));
        assert!(elapsed >= Duration::from_millis(30));
        assert!(
            elapsed < Duration::from_millis(750),
            "timeout restarted: {elapsed:?}"
        );
        Ok(())
    })
}

#[test]
fn sequence_wrap_is_a_new_frame() -> TestResult {
    let sender = FrameSender::new();
    sender.publisher.0.lock().sequence = u64::MAX - 1;
    let mut reader = sender.subscribe();
    sender.publish(Arc::new(1), 1, 1, Instant::now())?;
    assert_eq!(reader.recv()?.sequence, u64::MAX);
    sender.publish(Arc::new(2), 1, 1, Instant::now())?;
    assert_eq!(reader.recv()?.sequence, 0);
    Ok(())
}

#[test]
fn payload_destructor_runs_without_holding_the_slot() -> TestResult {
    struct Payload(Option<(mpsc::Sender<()>, mpsc::Receiver<()>)>);
    impl Drop for Payload {
        fn drop(&mut self) {
            if let Some((started, finish)) = &self.0 {
                let _ = started.send(());
                let _ = finish.recv_timeout(Duration::from_secs(2));
            }
        }
    }
    // The receiver in this test payload needs a Mutex to be Sync.
    let sender = FrameSender::new();
    let mut reader = sender.subscribe();
    let (started, dropping) = mpsc::channel();
    let (finish, finished) = mpsc::channel();
    sender.publish(
        Arc::new(Mutex::new(Payload(Some((started, finished))))),
        1,
        1,
        Instant::now(),
    )?;
    thread::scope(|scope| -> TestResult {
        scope.spawn(|| sender.publish(Arc::new(Mutex::new(Payload(None))), 2, 1, Instant::now()));
        dropping.recv_timeout(Duration::from_secs(2))?;
        let (done, result) = mpsc::channel();
        scope.spawn(move || {
            let _ = done.send(reader.recv());
        });
        let frame = result.recv_timeout(Duration::from_millis(500));
        let _ = finish.send(());
        assert_eq!(frame??.width, 2);
        Ok(())
    })
}
