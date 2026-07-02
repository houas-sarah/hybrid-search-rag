# Acme Platform — Getting Started

Acme Platform is an internal service for orchestrating batch data jobs.

## Requirements

- Python 3.11 or newer is required to run the control plane.
- PostgreSQL 15 is used as the metadata store.
- Redis 7 is required for the task queue.

## Installation

Clone the repository and install dependencies with `pip install -r requirements.txt`.
Then copy `.env.example` to `.env` and set your database credentials.

## Running

Start the API with `make run`. The service listens on port 8080 by default. You
can override the port with the `ACME_PORT` environment variable.
