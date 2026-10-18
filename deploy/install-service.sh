#!/usr/bin/env bash
# On the Pi, once: install the web console as a user service that starts at boot.
#   ./deploy/install-service.sh
# Then: systemctl --user start|stop|restart|status iss-console,  journalctl --user -u iss-console -f
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
unit="$HOME/.config/systemd/user/iss-console.service"

mkdir -p "$(dirname "$unit")"
sed "s#@ROOT@#$here#g" "$here/deploy/iss-console.service" > "$unit"
systemctl --user daemon-reload
systemctl --user enable iss-console
echo "installed $unit (enabled at boot)"

# a user service runs at boot without anyone logged in only with "linger" - needs sudo, once
if [[ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" != yes ]]; then
    echo "starting at boot without a login needs linger - sudo asks for your password once:"
    sudo loginctl enable-linger "$USER"
fi

# the page's "set the Pi's clock" button (a night with no network): a root-owned script that
# takes one number, and a sudo rule for that script alone - not for date, whose other options
# (date -f FILE prints any file) a wildcard rule would let through
rule="/etc/sudoers.d/iss-console-clock"
tool="/usr/local/sbin/iss-set-clock"
# is the rule there? /etc/sudoers.d is root's only, so ask sudo instead: with the rule the tool
# itself runs, and answers a call with no time by its usage (exit 2); without it, sudo wants a
# password (exit 1)
rc=0
sudo -n "$tool" >/dev/null 2>&1 || rc=$?
if ! cmp -s "$here/deploy/iss-set-clock" "$tool" || [[ "$rc" -ne 2 ]]; then
    tmp="$(mktemp)"
    echo "$USER ALL=(root) NOPASSWD: $tool" > "$tmp"
    if visudo -cf "$tmp" >/dev/null; then
        echo "the clock button needs a small root tool and one sudo rule - sudo asks once:"
        sudo install -m 0755 -o root -g root "$here/deploy/iss-set-clock" "$tool"
        sudo install -m 0440 -o root -g root "$tmp" "$rule"
    fi
    rm -f "$tmp"
fi

if systemctl --user is-active --quiet iss-console; then
    echo "it is running: systemctl --user restart iss-console to pick up a new version"
else
    echo "not started yet. Stop a console you started by hand (Ctrl-C), then:"
    echo "    systemctl --user start iss-console"
fi
