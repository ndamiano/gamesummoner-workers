"""Client for the image contract: /health, /metadata, /job, /shutdown on the box's inference
image. Nothing here knows which engine or provider sits behind it."""

from enum import Enum

import requests


class Health(Enum):
    BOOTING = "booting"
    HEALTHY = "healthy"
    LEAVING = "leaving"
    BROKEN = "broken"
    BUSY = "busy"


class JobTimeout(Exception):
    pass


class ImageClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def health(self) -> tuple:
        try:
            r = self.session.get(f"{self.base_url}/health", timeout=5)
        except requests.RequestException as e:
            return Health.BOOTING, f"no answer ({e.__class__.__name__})"
        reason = _reason(r)
        if r.status_code == 503:
            return Health.BOOTING, reason
        if r.status_code == 410:
            return Health.LEAVING, reason
        if r.status_code == 429:
            return Health.BUSY, reason
        if 200 <= r.status_code < 300:
            return Health.HEALTHY, reason
        return Health.BROKEN, f"status {r.status_code}: {reason}"

    def metadata(self) -> dict:
        r = self.session.get(f"{self.base_url}/metadata", timeout=10)
        r.raise_for_status()
        return r.json()

    def job(self, job_id: str, job_type: str, payload: dict, timeout: float) -> tuple:
        """(result, error), exactly one of them None."""
        try:
            r = self.session.post(f"{self.base_url}/job", timeout=timeout,
                                  json={"jobId": job_id, "type": job_type, "payload": payload})
        except requests.Timeout as e:
            raise JobTimeout(f"image did not answer job {job_id} within {timeout:.0f}s") from e
        try:
            body = r.json()
        except ValueError:
            body = {}
        if 200 <= r.status_code < 300:
            return body.get("result"), None
        return None, body.get("error") or f"Status {r.status_code}: {r.text[:2000]}"

    def shutdown(self) -> None:
        try:
            self.session.post(f"{self.base_url}/shutdown", timeout=10)
        except requests.RequestException:
            pass


def _reason(r) -> str:
    try:
        return (r.json() or {}).get("reason") or ""
    except ValueError:
        return r.text[:200]
