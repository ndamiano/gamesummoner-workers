"""Worker agent for GameSummoner"""

import argparse
import logging
import queue as queue_mod
import signal
import threading
import time

import requests

from gamesummoner_worker import envdata
from gamesummoner_worker.image import Health, ImageClient, JobTimeout

logger = logging.getLogger("worker")

HEARTBEAT_INTERVAL = 45.0
BOOT_POLL_INTERVAL = 2.0
POLL_INTERVAL = 1.0
CLAIM_WAIT_SECONDS = 2.0
DEFAULT_IMAGE_URL = "http://127.0.0.1:8700"


class Agent:
    def __init__(self, server: str, image: ImageClient, queue: str, token: str, worker_id: str,
                 idle_exit_seconds: float, slots: int, job_timeout_seconds: float):
        self.server = server.rstrip("/")
        self.image = image
        self.queue = queue
        self.worker_id = worker_id
        self.idle_exit_seconds = idle_exit_seconds
        self.slots = slots
        self.job_timeout_seconds = job_timeout_seconds
        self._running: set[str] = set()
        self._running_lock = threading.Condition()
        self._idle_since = time.monotonic()
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {token}"
        self.stopping = False
        self._uploads = queue_mod.Queue()

    def _post(self, path: str, body: dict, timeout: float) -> dict:
        r = self.session.post(f"{self.server}{path}", json=body, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def claim(self) -> dict | None:
        body = {
            "worker_id": self.worker_id,
            "queue": self.queue,
            "wait_seconds": min(CLAIM_WAIT_SECONDS, self.idle_exit_seconds)
        }
        return self._post("/worker/claim", body, timeout=body["wait_seconds"] + 10).get("job")

    def deregister(self) -> None:
        try:
            self._post("/worker/deregister", {"worker_id": self.worker_id}, timeout=10)
        except requests.RequestException as e:
            logger.warning("deregister failed: %s", e)

    def complete(self, body: dict) -> None:
        if not self._post("/worker/complete", body, timeout=120).get("ok"):
            logger.warning("job %s: lease lapsed before completion — result dropped",
                           body["job_id"])

    def _upload_loop(self) -> None:
        for body in iter(self._uploads.get, None):
            for attempt in (1, 2):
                try:
                    self.complete(body)
                    break
                except requests.RequestException as e:
                    logger.warning("job %s: complete failed (attempt %d/2): %s",
                                   body["job_id"], attempt, e)
                    time.sleep(2)
            else:
                logger.error("job %s: result dropped", body["job_id"])

    def _heartbeat_loop(self) -> None:
        while not self.stopping:
            time.sleep(HEARTBEAT_INTERVAL)
            with self._running_lock:
                running = list(self._running)
            for job_id in running:
                try:
                    ok = self._post("/worker/heartbeat",
                                    {"job_id": job_id, "worker_id": self.worker_id},
                                    timeout=10).get("ok")
                except requests.RequestException as e:
                    logger.warning("job %s: heartbeat failed: %s", job_id, e)
                    continue
                if not ok:
                    logger.warning("job %s: heartbeat rejected (lease lost)", job_id)

    def execute(self, job: dict) -> None:
        job_id, payload = job["id"], job["payload"]
        try:
            job_type = payload["kind"]
            t0 = time.perf_counter()
            try:
                result, error = self.image.job(job_id, job_type, payload,
                                               self.job_timeout_seconds)
            except JobTimeout as e:
                result, error = None, str(e)
                logger.error("%s — declaring the box unhealthy", e)
                self.stopping = True
            except requests.RequestException as e:
                result, error = None, f"Connection error: {e}"
            elapsed = time.perf_counter() - t0
            self._uploads.put({
                "worker_id": self.worker_id,
                "job_id": job_id,
                "result": result,
                "error": error,
                "exec_seconds": elapsed,
            })
            if error:
                logger.warning("job %s (%s) failed after %.1fs: %s", job_id, job_type, elapsed,
                               error)
            else:
                logger.info("job %s (%s) done in %.1fs", job_id, job_type, elapsed)
        finally:
            with self._running_lock:
                self._running.discard(job_id)
                if not self._running:
                    self._idle_since = time.monotonic()
                self._running_lock.notify_all()

    def _claim_loop(self) -> None:
        while not self.stopping:
            state, reason = self.image.health()
            if state in (Health.LEAVING, Health.BROKEN, Health.BOOTING):
                logger.error("image turned %s (%s) — draining", state.value, reason)
                return
            if state is Health.BUSY:
                logger.warning("image is not available (%s) — pausing claims", reason)
                time.sleep(POLL_INTERVAL)
                continue
            with self._running_lock:
                if not self._running_lock.wait_for(lambda: len(self._running) < self.slots,
                                                   timeout=POLL_INTERVAL):
                    continue

            try:
                job = self.claim()
            except requests.RequestException as e:
                if e.response is not None and e.response.status_code == 403:
                    logger.error("worker %s refused: %s — exiting", self.worker_id,
                                 e.response.text)
                    return
                logger.warning("claim failed (%s) — retrying in 5s", e)
                time.sleep(5)
                continue

            if job is not None:
                with self._running_lock:
                    self._running.add(job["id"])
                threading.Thread(target=self.execute, args=(job,), daemon=True).start()
                continue
            if len(self._running) == 0:
                idle_for = time.monotonic() - self._idle_since
                if idle_for >= self.idle_exit_seconds:
                    logger.info("worker %s idle for %.0fs — exiting", self.worker_id, idle_for)
                    return

    def wait_for_image(self) -> bool:
        """Block until the image is healthy. False when it will never serve (leaving or broken)."""
        last = None
        while not self.stopping:
            state, reason = self.image.health()
            if state is Health.HEALTHY:
                return True
            if state in (Health.LEAVING, Health.BROKEN):
                logger.error("image is %s (%s) — not claiming", state.value, reason)
                return False
            if reason != last:
                logger.info("image booting: %s", reason)
                last = reason
            time.sleep(BOOT_POLL_INTERVAL)
        return False

    def _serve(self) -> None:
        if not self.wait_for_image():
            return
        metadata = self.image.metadata()
        if self.queue not in metadata["capabilities"]:
            logger.error("image serves %s, not queue %s", metadata["capabilities"], self.queue)
            return
        logger.info("worker %s pulling queue=%s from %s (%d slots), image %s",
                    self.worker_id, self.queue, self.server, self.slots, metadata)
        self._claim_loop()

    def run(self) -> None:
        uploader = threading.Thread(target=self._upload_loop, daemon=True)
        uploader.start()
        threading.Thread(target=self._heartbeat_loop, daemon=True).start()
        try:
            self._serve()
        finally:
            self.stopping = True
            with self._running_lock:
                self._running_lock.wait_for(lambda: not self._running)
            self._uploads.put(None)
            uploader.join()
            self.deregister()
            logger.info("worker %s stopped — asking the image to decommission the box",
                        self.worker_id)
            self.image.shutdown()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="GameSummoner inference worker")
    parser.add_argument("--env-file", default=str(envdata.ENV_FILE))
    parser.add_argument("--image", default=DEFAULT_IMAGE_URL)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
    env = envdata.load(args.env_file)
    agent = Agent(
        server=env["CP_URL"],
        image=ImageClient(args.image),
        queue=env["WORKER_QUEUE"],
        token=env["WORKER_TOKEN"],
        worker_id=env["WORKER_ID"],
        idle_exit_seconds=float(env.get("IDLE_EXIT_SECONDS", 25)),
        slots=int(env.get("WORKER_SLOTS", 1)),
        job_timeout_seconds=float(env.get("JOB_TIMEOUT_SECONDS", 1800)),
    )

    def _stop(signum, frame):
        logger.info("signal %s — finishing current jobs then exiting", signum)
        agent.stopping = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    agent.run()
    return 0
