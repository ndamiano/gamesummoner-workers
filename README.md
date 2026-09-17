# gamesummoner-workers

The worker is the communication layer between the control plane and the actual inference.

The worker:

- waits for the image's `/health` to report healthy, then reads `/metadata`
- claims jobs from the control plane's `/worker/claim`
- hands each job to the image's `/job` and measures the execution seconds that billing uses
- returns results to the control plane on `worker/complete`
- exits after `IDLE_EXIT_SECONDS` when the queue on control plane is empty

Additionally, the worker listens to the image for shutdown requests (IE spot check deaths), and
deregisters from the control plane as needed.

## Image contract

### /health

This endpoint tells the worker whether the inference is ready to pickup work. A worker is still
booting if it returns a 503 or nothing. A 410 means the box is going away, and the worker should
clean up and deregister. A 429 means the image is up but taking no work, so the worker waits and
keeps the box. A 2xx response is healthy. Any other response means kill the box.

```json
{
  "reason": "Warming the model",
}
```

### /metadata

This endpoint will return the metadata to the worker, so they can tag anything needed with it.

```json
{
  "imageId": "UUID of the build",
  "cliParams": "string of params",
  "capabilities": ["image", "mesh"],
}
```

### /shutdown

This endpoint kills the pod. It may take some amount of time to attempt to flush logs, but the pod
will be dead shortly. If the pod fails to shutdown, it will be reaped by the scaler in the near term
future.

### /job

The endpoint the worker calls to do the work. Synchronous post endpoint. A worker hitting the
timeout (`JOB_TIMEOUT_SECONDS`, per queue, 2x that queue's longest job) may declare the box
unhealthy and request a shutdown.

Payload:
```json
{
  "jobId": "job id from the control plane",
  "type": "llm",
  "payload": {
    // Exactly as the control plane included
  }
}
```

Result:
```json
{
  "status": "succeeded",
  "result": {result} // The type of the result depends on the type of the image
}
```

Status codes: 2xx for success, 5xx for failures

## Env data

The image's entrypoint writes one JSON object to `/etc/gamesummoner/env.json`. Unknown keys are
ignored.

| key | meaning |
|---|---|
| `CP_URL` | control plane base URL |
| `WORKER_TOKEN` | the control plane's workqueue token |
| `WORKER_QUEUE` | the queue to claim from |
| `WORKER_ID` | the id the control plane created this worker under |
| `WORKER_SLOTS` | jobs the image runs at once (default 1) |
| `IDLE_EXIT_SECONDS` | seconds without work before the box exits |
| `JOB_TIMEOUT_SECONDS` | per-job ceiling (default 1800) |

## Run

```
python -m gamesummoner_worker --env-file /etc/gamesummoner/env.json --image http://127.0.0.1:8700
```

Only dependency is `requests`. Tests: `python -m pytest`.
