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

if systemctl --user is-active --quiet iss-console; then
    echo "it is running: systemctl --user restart iss-console to pick up a new version"
else
    echo "not started yet. Stop a console you started by hand (Ctrl-C), then:"
    echo "    systemctl --user start iss-console"
fi
