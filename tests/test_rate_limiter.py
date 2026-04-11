import time

import pytest


def test_token_bucket_rate():
    from brainlayer.pipeline.rate_limiter import TokenBucket

    bucket = TokenBucket(rate_per_sec=10.0, burst=1)

    start = time.monotonic()
    for _ in range(20):
        bucket.acquire()
    elapsed = time.monotonic() - start

    assert elapsed >= 1.9


def test_token_bucket_burst():
    from brainlayer.pipeline.rate_limiter import TokenBucket

    bucket = TokenBucket(rate_per_sec=20.0, burst=5)

    start = time.monotonic()
    for _ in range(5):
        bucket.acquire()
    initial_elapsed = time.monotonic() - start

    sixth_start = time.monotonic()
    bucket.acquire()
    sixth_elapsed = time.monotonic() - sixth_start

    assert initial_elapsed < 0.02
    assert sixth_elapsed >= 0.045


def test_token_bucket_rejects_non_positive_requests():
    from brainlayer.pipeline.rate_limiter import TokenBucket

    bucket = TokenBucket(rate_per_sec=20.0, burst=5)

    with pytest.raises(ValueError, match="positive"):
        bucket.acquire(0)


def test_token_bucket_rejects_requests_above_burst():
    from brainlayer.pipeline.rate_limiter import TokenBucket

    bucket = TokenBucket(rate_per_sec=20.0, burst=5)

    with pytest.raises(ValueError, match="burst capacity"):
        bucket.acquire(6)
