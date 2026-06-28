# Deploying bestee on a DigitalOcean droplet

Templates for running the API and the Celery worker as `systemd` services with
persistent, rotated logs. Adjust users, paths, and ports to your droplet.

## Logging model

Each service configures logging at startup via
`bestee_compute.logging_config.configure_logging(service)`, which installs:

- a **stdout** handler — captured by `systemd`/`journald` (`journalctl -u ...`), and
- a **file** handler — `WatchedFileHandler` writing `${LOG_DIR}/{service}.log`.

In-process rotation is intentionally disabled; the system `logrotate`
(`deploy/logrotate-bestee`) rotates the files, and the handler reopens on the
inode change. This is safe across the worker's forked processes.

The Celery worker additionally writes a **per-task audit trail** to
`${LOG_DIR}/audit.jsonl` — one JSON line per task (timestamp, task id, name,
state, result id, requester email, duration), covering successes and failures.

## Environment variables

| Variable     | Default  | Purpose                                        |
| ------------ | -------- | ---------------------------------------------- |
| `LOG_DIR`    | `logs`   | Directory for `{service}.log` + `audit.jsonl`. |
| `LOG_LEVEL`  | `INFO`   | Root logger level.                             |
| `LOG_FORMAT` | `json`   | `json` (one object per line) or `text`.        |

Secrets (`MASSIVE_API_KEY`, `DO_REDIS_CONNECTION`, `CLERK_*`, `RESEND_API_KEY`,
`RESULTS_DIR`, ...) live in the repo-root `.env`; the apps load it with
`python-dotenv`, which walks up from each service's working directory.

## One-time setup

```sh
# 1. Service user + dirs
sudo useradd --system --home /opt/bestee --shell /usr/sbin/nologin bestee
sudo mkdir -p /opt/bestee /var/log/bestee
sudo chown -R bestee:bestee /opt/bestee /var/log/bestee

# 2. Code + venvs (as the bestee user), e.g.
sudo -u bestee git clone <repo> /opt/bestee
sudo -u bestee bash -c 'cd /opt/bestee/compute && uv sync'
sudo -u bestee bash -c 'cd /opt/bestee/queue   && uv sync'
sudo -u bestee bash -c 'cd /opt/bestee/api     && uv sync'

# 3. Install the unit + logrotate files
sudo cp deploy/bestee-queue.service deploy/bestee-api.service /etc/systemd/system/
sudo cp deploy/logrotate-bestee /etc/logrotate.d/bestee

# 4. Start
sudo systemctl daemon-reload
sudo systemctl enable --now bestee-queue bestee-api
```

## Verify

```sh
journalctl -u bestee-queue -f          # live stdout
tail -f /var/log/bestee/queue.log      # persisted file logs (json)
tail -f /var/log/bestee/audit.jsonl    # per-task audit trail
sudo logrotate --debug /etc/logrotate.d/bestee   # dry-run rotation
```
