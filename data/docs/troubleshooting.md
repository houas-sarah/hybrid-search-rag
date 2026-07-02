# Troubleshooting

## Jobs stuck in "pending"

If jobs never leave the `pending` state, the worker pool is usually not running
or cannot reach Redis. Verify `ACME_REDIS_URL` is correct and that at least one
worker process is started with `make worker`.

## Database connection errors

`could not connect to server` on startup means the control plane cannot reach
PostgreSQL. Confirm `ACME_DATABASE_URL` points at a reachable PostgreSQL 15
instance and that the credentials are valid.

## High memory usage

Workers holding memory after large jobs is expected because result buffers are
cached. Lower `ACME_WORKER_CONCURRENCY` to reduce peak memory, or set
`ACME_JOB_TIMEOUT` lower to release long-running jobs sooner.
