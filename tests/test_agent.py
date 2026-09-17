"""Worker agent: waits on the image, claims, hands payloads to the image's /job, lands results
with measured exec time, and decommissions the box when it is done."""

import json
import threading
import time
from unittest.mock import MagicMock

import pytest
import requests

import gamesummoner_worker.agent as worker_agent
from gamesummoner_worker.agent import Agent
from gamesummoner_worker.image import Health, ImageClient, JobTimeout


class FakeResponse:
    def __init__(self, status_code=200, body=None, text=""):
        self.status_code = status_code
        self._body = body
        self.text = text

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)


class FakeImage:
    def __init__(self, health=None, capabilities=("llm",), job=None, metadata_error=None):
        self.health_states = list(health or [(Health.HEALTHY, "")])
        self.capabilities = list(capabilities)
        self.metadata_error = metadata_error
        self.job_fn = job or (lambda job_id, job_type, payload: ({"choices": []}, None))
        self.jobs = []
        self.shutdowns = 0

    def health(self):
        if len(self.health_states) > 1:
            return self.health_states.pop(0)
        return self.health_states[0]

    def metadata(self):
        if self.metadata_error:
            raise self.metadata_error
        return {"imageId": "img-1", "cliParams": "", "capabilities": self.capabilities}

    def job(self, job_id, job_type, payload, timeout):
        self.jobs.append((job_id, job_type, payload, timeout))
        return self.job_fn(job_id, job_type, payload)

    def shutdown(self):
        self.shutdowns += 1


def make_agent(image=None, **kw):
    settings = {"idle_exit_seconds": 3600, "slots": 1, "job_timeout_seconds": 1800} | kw
    a = Agent("http://server", image or FakeImage(), "llm", "wsecret", "w1", **settings)
    a.session = MagicMock()
    return a


def land_uploads(a):
    a._uploads.put(None)
    a._upload_loop()


def posts_to(a, path):
    return sum(1 for c in a.session.post.call_args_list if c.args[0].endswith(path))


def cp_calls(a):
    return [c.args[0].rsplit("/", 1)[-1] for c in a.session.post.call_args_list]


@pytest.fixture(autouse=True)
def fast_polls(monkeypatch):
    monkeypatch.setattr(worker_agent, "BOOT_POLL_INTERVAL", 0)
    monkeypatch.setattr(worker_agent, "POLL_INTERVAL", 0)


def test_execute_hands_the_payload_to_the_image_and_lands_its_result():
    image = FakeImage(job=lambda *_: ({"choices": [{"message": {"content": "hi"}}]}, None))
    a = make_agent(image, job_timeout_seconds=600)
    a.session.post.return_value = FakeResponse(200, {"ok": True})
    payload = {"kind": "llm", "body": {"model": "m", "messages": []}}

    a.execute({"id": "j1", "payload": payload})
    land_uploads(a)

    assert image.jobs == [("j1", "llm", payload, 600)]
    body = a.session.post.call_args.kwargs["json"]
    assert a.session.post.call_args.args[0] == "http://server/worker/complete"
    assert body["job_id"] == "j1"
    assert body["result"] == {"choices": [{"message": {"content": "hi"}}]}
    assert body["error"] is None and body["exec_seconds"] > 0


def test_image_failures_and_timeouts_land_as_job_errors():
    """A failed job is reported and the worker carries on; a job the image never answers is
    reported and the worker stops claiming, because the box is no longer trusted."""
    outcomes = iter([(None, "Status 500: engine died"), JobTimeout("no answer in 600s")])

    def job(*_):
        o = next(outcomes)
        if isinstance(o, Exception):
            raise o
        return o

    a = make_agent(FakeImage(job=job))
    a.session.post.return_value = FakeResponse(200, {"ok": True})

    a.execute({"id": "j1", "payload": {"kind": "llm"}})
    assert a.stopping is False
    a.execute({"id": "j2", "payload": {"kind": "llm"}})
    assert a.stopping is True
    land_uploads(a)

    errors = [c.kwargs["json"]["error"] for c in a.session.post.call_args_list]
    assert errors == ["Status 500: engine died", "no answer in 600s"]


def test_a_job_that_blows_up_still_frees_its_slot():
    a = make_agent()
    with a._running_lock:
        a._running.add("j1")
    with pytest.raises(KeyError):
        a.execute({"id": "j1", "payload": {}})
    assert a._running == set()


def test_a_slot_claims_again_while_the_last_result_is_still_uploading():
    a = make_agent()
    claimed_again = threading.Event()
    claims = {"n": 0}
    uploads_saw_the_next_claim = []

    def post(url, **kw):
        if url.endswith("/worker/claim"):
            claims["n"] += 1
            if claims["n"] == 1:
                return FakeResponse(200, {"job": {"id": "j1", "payload": {"kind": "llm"}}})
            a.stopping = True
            claimed_again.set()
            return FakeResponse(200, {"job": None})
        if url.endswith("/worker/complete"):
            uploads_saw_the_next_claim.append(claimed_again.wait(5))
        return FakeResponse(200, {"ok": True})

    a.session.post.side_effect = post
    a.run()
    assert uploads_saw_the_next_claim == [True]


def test_run_waits_for_health_then_uploads_deregisters_and_decommissions_in_order():
    image = FakeImage(health=[(Health.BOOTING, "no answer"), (Health.BOOTING, "warming"),
                              (Health.HEALTHY, "")])
    a = make_agent(image)

    def post(url, **kw):
        if url.endswith("/worker/claim"):
            assert image.health_states == [(Health.HEALTHY, "")]
            a.stopping = True
            return FakeResponse(200, {"job": {"id": "j1", "payload": {"kind": "llm"}}})
        return FakeResponse(200, {"ok": True})

    a.session.post.side_effect = post
    a.run()
    assert cp_calls(a) == ["claim", "complete", "deregister"]
    assert image.shutdowns == 1


def test_an_unexpected_error_still_deregisters_and_decommissions():
    image = FakeImage(metadata_error=requests.ConnectionError("image died"))
    a = make_agent(image)
    a.session.post.return_value = FakeResponse(200, {"ok": True})
    with pytest.raises(requests.ConnectionError):
        a.run()
    assert cp_calls(a) == ["deregister"]
    assert image.shutdowns == 1


@pytest.mark.parametrize("image", [
    FakeImage(health=[(Health.LEAVING, "spot notice")]),
    FakeImage(health=[(Health.BROKEN, "status 500")]),
    FakeImage(capabilities=["image"]),
], ids=["leaving", "broken", "wrong-capability"])
def test_an_image_that_cannot_serve_the_queue_is_never_claimed_from(image):
    a = make_agent(image)
    a.session.post.return_value = FakeResponse(200, {"ok": True})
    a.run()
    assert cp_calls(a) == ["deregister"]
    assert image.shutdowns == 1


def test_an_image_turning_unhealthy_mid_run_stops_the_claims():
    image = FakeImage(health=[(Health.HEALTHY, ""), (Health.LEAVING, "spot notice")])
    a = make_agent(image)
    a.session.post.return_value = FakeResponse(200, {"job": None})
    a.run()
    assert posts_to(a, "/worker/deregister") == 1
    assert image.shutdowns == 1


def test_an_idle_window_of_zero_exits_on_the_first_empty_claim():
    image = FakeImage()
    a = make_agent(image, idle_exit_seconds=0)
    a.session.post.return_value = FakeResponse(200, {"job": None})
    a.run()
    assert posts_to(a, "/worker/claim") == 1
    assert a.session.post.call_args_list[0].kwargs["json"]["wait_seconds"] == 0
    assert image.shutdowns == 1


def test_claims_stop_while_every_slot_is_full_and_resume_when_one_frees():
    started = threading.Semaphore(0)
    release = threading.Event()

    def job(*_):
        started.release()
        release.wait(5)
        return {"choices": []}, None

    a = make_agent(FakeImage(job=job), slots=2)
    handed = ["j1", "j2"]
    claims = {"n": 0}
    claims_while_full = {"n": 0}
    lock = threading.Lock()

    def post(url, **kw):
        if url.endswith("/worker/claim"):
            with lock:
                claims["n"] += 1
                if handed:
                    return FakeResponse(200, {"job": {"id": handed.pop(0),
                                                      "payload": {"kind": "llm"}}})
                if not release.is_set():
                    claims_while_full["n"] += 1
                a.stopping = True
            return FakeResponse(200, {"job": None})
        return FakeResponse(200, {"ok": True})

    a.session.post.side_effect = post
    runner = threading.Thread(target=a.run)
    runner.start()
    assert started.acquire(timeout=5) and started.acquire(timeout=5)
    time.sleep(0.05)
    assert claims["n"] == 2 and claims_while_full["n"] == 0
    release.set()
    runner.join(5)
    assert posts_to(a, "/worker/complete") == 2


def test_claims_pause_while_the_image_takes_no_work():
    image = FakeImage(health=[(Health.HEALTHY, ""), (Health.BUSY, "queue full"),
                              (Health.HEALTHY, "")])
    a = make_agent(image, idle_exit_seconds=0)
    a.session.post.return_value = FakeResponse(200, {"job": None})
    a.run()
    assert posts_to(a, "/worker/claim") == 1
    assert image.shutdowns == 1


def _clock(monkeypatch, start=1000.0):
    t = {"now": start}
    monkeypatch.setattr(worker_agent.time, "monotonic", lambda: t["now"])
    return t


def test_null_claims_exit_only_once_idle_for_the_whole_window(monkeypatch):
    clock = _clock(monkeypatch)
    image = FakeImage()
    a = make_agent(image, idle_exit_seconds=300)
    nulls = {"n": 0}

    def post(url, **kw):
        if url.endswith("/worker/claim"):
            nulls["n"] += 1
            clock["now"] += 25
            return FakeResponse(200, {"job": None})
        return FakeResponse(200, {"ok": True})

    a.session.post.side_effect = post
    a.run()
    assert nulls["n"] == 12
    assert posts_to(a, "/worker/deregister") == 1
    assert image.shutdowns == 1


def test_the_idle_clock_restarts_when_a_job_ends(monkeypatch):
    clock = _clock(monkeypatch)

    def long_job(*_):
        clock["now"] += 500
        return {"choices": []}, None

    a = make_agent(FakeImage(job=long_job), idle_exit_seconds=60)
    claims = {"n": 0}

    def post(url, **kw):
        if url.endswith("/worker/claim"):
            claims["n"] += 1
            if claims["n"] == 1:
                return FakeResponse(200, {"job": {"id": "j1", "payload": {"kind": "llm"}}})
            clock["now"] += 25
            return FakeResponse(200, {"job": None})
        return FakeResponse(200, {"ok": True})

    a.session.post.side_effect = post
    a.run()
    assert claims["n"] == 1 + 3


def test_a_refused_worker_exits_and_deregister_failures_are_swallowed():
    a = make_agent()

    def post(url, **kw):
        if url.endswith("/worker/claim"):
            return FakeResponse(403, text="worker w1 was never created")
        raise requests.ConnectionError("cp down")

    a.session.post.side_effect = post
    a.run()
    assert posts_to(a, "/worker/claim") == 1
    assert posts_to(a, "/worker/deregister") == 1


def test_slots_run_jobs_side_by_side():
    both_in = threading.Barrier(2, timeout=5)

    def job(*_):
        both_in.wait()
        return {"choices": []}, None

    a = make_agent(FakeImage(job=job), slots=2)
    handed = ["j1", "j2"]
    lock = threading.Lock()

    def post(url, **kw):
        if url.endswith("/worker/claim"):
            with lock:
                if handed:
                    return FakeResponse(200, {"job": {"id": handed.pop(0),
                                                      "payload": {"kind": "llm"}}})
            a.stopping = True
            return FakeResponse(200, {"job": None})
        return FakeResponse(200, {"ok": True})

    a.session.post.side_effect = post
    a.run()
    assert both_in.broken is False
    assert posts_to(a, "/worker/complete") == 2
    assert posts_to(a, "/worker/deregister") == 1


def test_an_idle_slot_waits_for_a_busy_sibling():
    release = threading.Event()

    def job(*_):
        release.wait(5)
        return {"choices": []}, None

    a = make_agent(FakeImage(job=job), idle_exit_seconds=0.5, slots=2)
    claims = {"n": 0}
    null_while_busy = {"n": 0}
    lock = threading.Lock()

    def post(url, **kw):
        if url.endswith("/worker/claim"):
            with lock:
                claims["n"] += 1
                first = claims["n"] == 1
            if first:
                return FakeResponse(200, {"job": {"id": "j1", "payload": {"kind": "llm"}}})
            with a._running_lock:
                busy = bool(a._running)
            if busy:
                null_while_busy["n"] += 1
                if null_while_busy["n"] == 2:
                    release.set()
            return FakeResponse(200, {"job": None})
        return FakeResponse(200, {"ok": True})

    a.session.post.side_effect = post
    a.run()
    assert null_while_busy["n"] >= 2
    assert posts_to(a, "/worker/complete") == 1
    assert posts_to(a, "/worker/deregister") == 1


def test_one_heartbeat_thread_keeps_every_running_job_leased(monkeypatch):
    monkeypatch.setattr(worker_agent, "HEARTBEAT_INTERVAL", 0.01)
    beaten = threading.Event()

    def job(*_):
        assert beaten.wait(5)
        return {"choices": []}, None

    a = make_agent(FakeImage(job=job), slots=2)
    handed = ["j1", "j2"]
    beats = []
    lock = threading.Lock()

    def post(url, **kw):
        if url.endswith("/worker/claim"):
            with lock:
                if handed:
                    return FakeResponse(200, {"job": {"id": handed.pop(0),
                                                      "payload": {"kind": "llm"}}})
            a.stopping = True
            return FakeResponse(200, {"job": None})
        if url.endswith("/worker/heartbeat"):
            with lock:
                beats.append(kw["json"])
                if {b["job_id"] for b in beats} == {"j1", "j2"}:
                    beaten.set()
        return FakeResponse(200, {"ok": True})

    a.session.post.side_effect = post
    a.run()
    assert beaten.is_set()
    assert {b["worker_id"] for b in beats} == {"w1"}
    assert posts_to(a, "/worker/complete") == 2


def test_image_client_maps_the_contract_onto_health_results_and_errors():
    client = ImageClient("http://img/")
    client.session = MagicMock()
    health_answers = [
        requests.ConnectionError("refused"),
        FakeResponse(503, {"reason": "Warming the model"}),
        FakeResponse(200, {"reason": "ready"}),
        FakeResponse(410, {"reason": "spot notice"}),
        FakeResponse(429, {"reason": "queue full"}),
        FakeResponse(500, text="boom"),
    ]
    client.session.get.side_effect = health_answers
    states = [client.health()[0] for _ in health_answers]
    assert states == [Health.BOOTING, Health.BOOTING, Health.HEALTHY, Health.LEAVING,
                      Health.BUSY, Health.BROKEN]

    client.session.post.side_effect = [
        FakeResponse(200, {"status": "succeeded", "result": {"choices": []}}),
        FakeResponse(502, {"status": "failed", "error": "Status 400: context too long"}),
        FakeResponse(500, text="Internal Server Error"),
        requests.ReadTimeout("slow"),
    ]
    assert client.job("j1", "llm", {"kind": "llm"}, 60) == ({"choices": []}, None)
    assert client.job("j2", "llm", {}, 60) == (None, "Status 400: context too long")
    assert client.job("j3", "llm", {}, 60) == (None, "Status 500: Internal Server Error")
    with pytest.raises(JobTimeout):
        client.job("j4", "llm", {}, 60)

    sent = client.session.post.call_args_list[0]
    assert sent.args[0] == "http://img/job"
    assert sent.kwargs["json"] == {"jobId": "j1", "type": "llm", "payload": {"kind": "llm"}}
    assert "Authorization" not in client.session.headers


def test_main_builds_the_agent_from_the_env_file_ignoring_unknown_keys(tmp_path, monkeypatch):
    env_file = tmp_path / "env.json"
    env_file.write_text(json.dumps({
        "CP_URL": "http://cp:8000", "WORKER_TOKEN": "t", "WORKER_QUEUE": "llm",
        "WORKER_ID": "w-9", "WORKER_SLOTS": "2", "IDLE_EXIT_SECONDS": "30",
        "JOB_TIMEOUT_SECONDS": "900", "LLM_MODEL": "some-model",
    }))
    built = {}
    monkeypatch.setattr(Agent, "run", lambda self: built.update(agent=self))

    assert worker_agent.main(["--env-file", str(env_file), "--image", "http://img:1"]) == 0

    a = built["agent"]
    assert (a.server, a.queue, a.worker_id, a.slots) == ("http://cp:8000", "llm", "w-9", 2)
    assert (a.idle_exit_seconds, a.job_timeout_seconds) == (30.0, 900.0)
    assert a.image.base_url == "http://img:1"
    assert a.session.headers["Authorization"] == "Bearer t"
