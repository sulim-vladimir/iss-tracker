#!/usr/bin/env bash
# From the laptop: copy the code to the Pi, restart the console service, check it came back.
#   ./deploy/deploy.sh            refuses while a session or a calibration runs
#   ./deploy/deploy.sh --force    restarts anyway
#   ./deploy/deploy.sh --copy     copies only, no restart
# PI=pi@host and PORT=8090 override the defaults. The password is asked once (one shared ssh).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PI="${PI:-pi@192.168.0.104}"
HOST="${PI#*@}"
PORT="${PORT:-8090}"
mode="${1:-}"
ssh_opts=(-o ControlMaster=auto -o "ControlPath=$HOME/.ssh/cm-%r@%h:%p" -o ControlPersist=120)

state() { curl -s --max-time 5 "http://$HOST:$PORT/api/state" || true; }
field() {   # field <python expression on s (the state)>: prints it, or nothing
    python3 -c 'import json, sys
try:
    s = json.loads(sys.stdin.read())
except Exception:
    sys.exit()
v = eval(sys.argv[1])
print("" if v is None else v)' "$1"
}

if [[ "$mode" != --force && "$mode" != --copy ]]; then
    s="$(state)"
    if [[ "$(field 's["mount"].get("mode")' <<<"$s")" == track ]]; then
        echo "a session is running on the Pi - stop it first, or: $0 --force" >&2
        exit 1
    fi
    if [[ "$(field 's["mount"].get("busy")' <<<"$s")" == True ]]; then
        echo "the Pi is busy (calibration, goto...) - wait, or: $0 --force" >&2
        exit 1
    fi
fi

rsync -a --exclude __pycache__ -e "ssh ${ssh_opts[*]}" \
    "$here/issctl" "$here/deploy" "$here/config.example.toml" "$here/issctl.sh" "$PI:iss-tracker/"
echo "copied to $PI:iss-tracker"
[[ "$mode" == --copy ]] && exit 0

ssh "${ssh_opts[@]}" "$PI" 'systemctl --user restart iss-console'
echo -n "restarting"
s=""
for _ in $(seq 60); do
    s="$(state)"
    [[ -n "$s" ]] && break
    echo -n "."
    sleep 2
done
echo
if [[ -z "$s" ]]; then
    echo "the console did not come back - on the Pi: journalctl --user -u iss-console -n 50" >&2
    exit 1
fi

# the cameras re-enumerate after a restart: give them a little while
for _ in $(seq 15); do
    s="$(state)"
    ok="$(field 'all((c.get("fps") or 0) > 0 for c in s["cams"].values())' <<<"$s")"
    [[ "$ok" == True ]] && break
    sleep 2
done
sleep 10                # the first fps figures after a start are not settled yet
s="$(state)"
field '"\n".join(["clock: %+.1f s against this laptop" % (s["mount"]["now"] - __import__("time").time())]
                 + ["%s: %.1f fps at %g ms" % (n, c.get("fps") or 0, c["exposure_ms"])
                    for n, c in s["cams"].items()])' <<<"$s"
