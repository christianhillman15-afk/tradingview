# Deploying the live paper account on your own server

This runs a **standing $50,000 paper account** 24/7 on a server you control, with
the dashboard reachable in your browser. Two paths: **Docker** (simplest) or
**systemd + virtualenv** (no Docker). Pick one.

> ⚠️ Paper trading only. The dashboard is **read-only** and has **no
> authentication** — do not expose it to the public internet unprotected. Use an
> SSH tunnel (recommended) or put it behind a reverse proxy with auth (below).

---

## Option A — Docker (recommended)

On the server:

```bash
git clone <your-repo-url> ai-futures-bot && cd ai-futures-bot
git checkout claude/ai-futures-trading-bot-cyop2o      # the bot branch
docker compose -f deploy/docker-compose.yml up -d --build
docker compose -f deploy/docker-compose.yml logs -f    # watch it tick
```

- The account + dashboard state persist in `./runtime` (a mounted volume), so
  restarts/redeploys resume the same account.
- By default the port is published to `127.0.0.1:8000` only — reach it with the
  SSH tunnel below. To trade a different market/strategy, edit the `command:` in
  `deploy/docker-compose.yml`.
- For the ML strategies, uncomment the `numpy scikit-learn` line in the
  `Dockerfile` before building.

Stop / reset:
```bash
docker compose -f deploy/docker-compose.yml down        # stop (account kept)
rm runtime/account.json                                 # start fresh next up
```

## Option B — systemd + virtualenv (no Docker)

```bash
sudo mkdir -p /opt/ai-futures-bot && cd /opt/ai-futures-bot
git clone <your-repo-url> . && git checkout claude/ai-futures-trading-bot-cyop2o
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt               # PyYAML (+ numpy/sklearn for ML)
sudo useradd -r -s /usr/sbin/nologin botuser && sudo chown -R botuser /opt/ai-futures-bot

sudo cp deploy/ai-futures-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ai-futures-bot
journalctl -u ai-futures-bot -f                         # follow logs
```

Edit `deploy/ai-futures-bot.service` to change the symbol/strategy or paths.

---

## Viewing the dashboard

### Best: SSH tunnel (no ports exposed)
From your laptop:
```bash
ssh -N -L 8000:127.0.0.1:8000 you@your-server
# now open http://localhost:8000 in your browser
```

### Alternative: expose + protect
1. Bind to all interfaces: set `--host 0.0.0.0` (compose: change ports to
   `"8000:8000"`).
2. **Put auth in front of it.** Example nginx with basic auth:
   ```nginx
   server {
     listen 443 ssl;
     server_name bot.example.com;
     # ssl_certificate ...;
     location / {
       auth_basic "AI Futures Bot";
       auth_basic_user_file /etc/nginx/.htpasswd;   # htpasswd -c ... youruser
       proxy_pass http://127.0.0.1:8000;
     }
   }
   ```
3. Restrict the port with a firewall (`ufw allow from <your-ip> to any port 8000`).

---

## Using real market data

The container/service uses a live **synthetic** feed by default (free real-time
endpoints rate-limit shared/cloud IPs). To trade real history instead:

```bash
python -m ai_futures_bot.cli fetch --symbol ES --range 2y --interval 1d --out data/es.csv
python -m ai_futures_bot.cli backtest --symbol ES --csv data/es.csv --timeframe 1
```

For a true real-time feed, wire `broker/ib.py` (Interactive Brokers) into the
live loop — `papertrade` is structured so the feed is the only piece to swap.

## Operations

| Task | Command |
|------|---------|
| Resume account | just restart — it loads `runtime/account.json` |
| Start fresh | delete `runtime/account.json` (or `--reset`) |
| Change strategy | edit the `command:` / `ExecStart` line |
| Update the bot | `git pull` then rebuild/restart |
| Back up the account | copy `runtime/account.json` somewhere safe |

## Resource notes
- The default `ensemble` strategy re-derives indicators each bar over a small
  rolling window — negligible CPU at 1 bar/sec. The ML strategies are heavier;
  pre-train a model (`train`) rather than retraining live.
- Memory footprint is tiny (a few tens of MB).

## Reminder
Education & research only — not financial advice. Even paper results don't
guarantee anything live. Validate with `walkforward` and real data before you'd
ever consider real capital.
