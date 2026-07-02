# Configuration Reference

All configuration is provided through environment variables.

## Core settings

- `ACME_PORT`: TCP port the API binds to. Defaults to `8080`.
- `ACME_DATABASE_URL`: PostgreSQL connection string. Required.
- `ACME_REDIS_URL`: Redis connection string for the task queue. Required.
- `ACME_LOG_LEVEL`: One of `debug`, `info`, `warning`, `error`. Defaults to `info`.

## Worker settings

- `ACME_WORKER_CONCURRENCY`: Number of concurrent jobs per worker. Defaults to `4`.
- `ACME_JOB_TIMEOUT`: Maximum job runtime in seconds before it is killed. Defaults to `3600`.

## Retries

Failed jobs are retried automatically. The retry policy uses exponential backoff
starting at 5 seconds, capped at 5 attempts. Set `ACME_MAX_RETRIES` to override
the attempt count.
