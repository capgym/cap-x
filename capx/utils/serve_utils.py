import time

import requests


def post_with_retries(
    url: str,
    payload: dict,
    timeout_seconds: float = 120.0,
    retry_interval: float = 1.0,
    max_retries: int = 5,
):
    """
    Retry POST requests with exponential backoff for up to `timeout_seconds` of wall clock time.

    Args:
        url: The URL to POST to.
        payload: JSON payload to send.
        timeout_seconds: Maximum wall clock time before giving up.
        retry_interval: Initial interval between retries (doubles each retry).
        max_retries: Maximum number of retry attempts.

    Raises RuntimeError if the time limit or retry count is exceeded.
    """
    deadline = time.time() + timeout_seconds
    current_interval = retry_interval

    last_err = None
    attempts = 0
    while time.time() < deadline and attempts < max_retries:
        try:
            resp = requests.post(url, json=payload, timeout=timeout_seconds)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            last_err = e
            attempts += 1
            time.sleep(min(current_interval, max(0, deadline - time.time())))
            current_interval = min(current_interval * 2, 8.0)

    raise RuntimeError(
        f"Request to {url} failed after {attempts} retries / "
        f"{timeout_seconds:.2f}s. Last error: {last_err}"
    )


def post_with_queue_tolerance(
    url: str,
    payload: dict,
    timeout_seconds: float = 120.0,
    retry_interval: float = 1.0,
    max_retries: int = 5,
):
    """
    POST with tolerance for queued servers (handles 503 gracefully).

    Like `post_with_retries`, but treats HTTP 503 (Service Unavailable) as a
    transient condition (server is busy with other requests) and retries with
    exponential backoff instead of raising immediately.

    Args:
        url: The URL to POST to.
        payload: JSON payload to send.
        timeout_seconds: Maximum wall clock time before giving up.
        retry_interval: Initial interval between retries (doubles each retry).
        max_retries: Maximum number of retry attempts.

    Raises RuntimeError if the time limit or retry count is exceeded.
    """
    deadline = time.time() + timeout_seconds
    current_interval = retry_interval

    last_err = None
    attempts = 0
    while time.time() < deadline and attempts < max_retries:
        try:
            resp = requests.post(url, json=payload, timeout=timeout_seconds)
            if resp.status_code == 503:
                # Server is busy / model not ready -- treat as transient
                last_err = requests.HTTPError(
                    f"503 Service Unavailable: {resp.text}", response=resp
                )
                attempts += 1
                time.sleep(min(current_interval, max(0, deadline - time.time())))
                current_interval = min(current_interval * 2, 8.0)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            last_err = e
            attempts += 1
            time.sleep(min(current_interval, max(0, deadline - time.time())))
            current_interval = min(current_interval * 2, 8.0)

    raise RuntimeError(
        f"Request to {url} failed after {attempts} retries / "
        f"{timeout_seconds:.2f}s. Last error: {last_err}"
    )
