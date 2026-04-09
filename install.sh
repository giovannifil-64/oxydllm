#!/usr/bin/env sh
# rLLM installer
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/giovannifil-64/rllm/main/install.sh | sh
#
# Nightly:
#   curl -fsSL https://raw.githubusercontent.com/giovannifil-64/rllm/main/install.sh | RLLM_CHANNEL=nightly sh
#
# Environment overrides:
#   RLLM_CHANNEL     - stable (default) or nightly
#   RLLM_VERSION     - install a specific version tag, e.g. v0.1.0 (ignored for nightly)
#   RLLM_NO_GPU      - set to 1 to force CPU binary on Linux
#   RLLM_INSTALL_DIR - destination directory (default: /usr/local/bin)

main() {
set -eu

REPO="giovannifil-64/rllm"
GITHUB_API="https://api.github.com/repos/${REPO}"
GITHUB_RELEASES="https://github.com/${REPO}/releases"

INSTALL_DIR="${RLLM_INSTALL_DIR:-/usr/local/bin}"
CHANNEL="${RLLM_CHANNEL:-stable}"
NO_GPU="${RLLM_NO_GPU:-0}"

say() {
    printf '>>> %s\n' "$*" >&2
}

warn() {
    printf 'WARNING: %s\n' "$*" >&2
}

err() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

available() {
    command -v "$1" >/dev/null 2>&1
}

need() {
    available "$1" || err "'$1' is required but was not found."
}

is_root() {
    [ "$(id -u)" -eq 0 ]
}

as_root() {
    if is_root; then
        "$@"
    else
        need sudo
        sudo "$@"
    fi
}

resolve_home_for_user() {
    _user="$1"
    if available getent; then
        getent passwd "${_user}" | awk -F: 'NR==1 {print $6}'
        return
    fi
    awk -F: -v u="${_user}" '$1==u {print $6; exit}' /etc/passwd 2>/dev/null || true
}

install_systemd_service() {
    if ! available systemctl; then
        warn "systemd not found. The server was installed but not configured as a service."
        warn "Start manually with: rllm start"
        return
    fi

    SERVICE_USER="${SUDO_USER:-$(id -un)}"
    SERVICE_GROUP="$(id -gn "${SERVICE_USER}" 2>/dev/null || echo "${SERVICE_USER}")"
    SERVICE_HOME="$(resolve_home_for_user "${SERVICE_USER}")"
    if [ -z "${SERVICE_HOME}" ]; then
        SERVICE_HOME="${HOME}"
    fi
    MODELS_DIR="${SERVICE_HOME}/.rllm/models"

    say "Configuring systemd service as user '${SERVICE_USER}'..."
    as_root mkdir -p "${MODELS_DIR}"
    as_root chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "${SERVICE_HOME}/.rllm"

    if [ ! -f /etc/default/rllm ]; then
        cat <<'ENVEOF' | as_root tee /etc/default/rllm >/dev/null
# rLLM server configuration
# RLLM_PORT=11313
# RLLM_MAX_CONTEXT_LEN=4096
# RLLM_KEEP_ALIVE=900
# RLLM_MEMORY_BUDGET=
# RLLM_KV_QUANT=off
# RLLM_DEVICES=
# RUST_LOG=warn
ENVEOF
    fi

    cat <<UNITEOF | as_root tee /etc/systemd/system/rllm.service >/dev/null
[Unit]
Description=rLLM inference server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${SERVICE_USER}
Group=${SERVICE_GROUP}
Environment=HOME=${SERVICE_HOME}
Environment=RUST_LOG=warn
EnvironmentFile=-/etc/default/rllm
ExecStart=${INSTALL_DIR}/rllm start --models-dir ${MODELS_DIR}
Restart=always
RestartSec=3
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
UNITEOF

    SYSTEMCTL_STATE="$(systemctl is-system-running 2>/dev/null || true)"
    case "${SYSTEMCTL_STATE}" in
        running|degraded)
            as_root systemctl daemon-reload
            as_root systemctl enable rllm >/dev/null 2>&1
            as_root systemctl restart rllm
            say "systemd service installed and started."
            ;;
        *)
            warn "systemd is not fully running (state: ${SYSTEMCTL_STATE:-unknown})."
            warn "Service file installed but not started automatically."
            warn "Start manually with: sudo systemctl start rllm"
            ;;
    esac
}

install_launchd_agent() {
    if [ "$(id -u)" -eq 0 ]; then
        warn "Running as root on macOS: skipping LaunchAgent installation."
        warn "Re-run installer as your login user to install autostart service."
        return
    fi

    MODELS_DIR="${HOME}/.rllm/models"
    LOG_DIR="${HOME}/.rllm/logs"
    PLIST_PATH="${HOME}/Library/LaunchAgents/com.rllm.rllmd.plist"
    LABEL="com.rllm.rllmd"
    GUI_DOMAIN="gui/$(id -u)"

    mkdir -p "${MODELS_DIR}" "${LOG_DIR}" "$(dirname "${PLIST_PATH}")"

    say "Installing launchd agent..."
    cat > "${PLIST_PATH}" <<PLISTEOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${LABEL}</string>

    <key>ProgramArguments</key>
    <array>
        <string>${INSTALL_DIR}/rllm</string>
        <string>start</string>
        <string>--models-dir</string>
        <string>${MODELS_DIR}</string>
    </array>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>

    <key>StandardOutPath</key>
    <string>${LOG_DIR}/rllm.log</string>
    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/rllm.log</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>RUST_LOG</key>
        <string>warn</string>
    </dict>

    <key>SoftResourceLimits</key>
    <dict>
        <key>NumberOfFiles</key>
        <integer>65536</integer>
    </dict>
</dict>
</plist>
PLISTEOF

    launchctl bootout "${GUI_DOMAIN}" "${PLIST_PATH}" 2>/dev/null || true
    launchctl bootstrap "${GUI_DOMAIN}" "${PLIST_PATH}"
    launchctl enable "${GUI_DOMAIN}/${LABEL}" 2>/dev/null || true

    say "launchd agent installed and started."
}

need curl
need tar

OS="$(uname -s)"
ARCH="$(uname -m)"

case "${OS}" in
    Darwin)
        case "${ARCH}" in
            arm64) PLATFORM="macos-arm64" ;;
            *) err "rLLM supports only Apple Silicon (arm64) on macOS." ;;
        esac
        ;;
    Linux)
        case "${ARCH}" in
            x86_64)
                if [ "${NO_GPU}" = "1" ]; then
                    PLATFORM="linux-x86_64-cpu"
                elif available nvidia-smi; then
                    DRIVER_MAJOR="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)"
                    if [ -n "${DRIVER_MAJOR}" ] && [ "${DRIVER_MAJOR}" -ge 525 ] 2>/dev/null; then
                        PLATFORM="linux-x86_64-cuda"
                    else
                        warn "NVIDIA driver found but < 525. Falling back to CPU build."
                        PLATFORM="linux-x86_64-cpu"
                    fi
                else
                    PLATFORM="linux-x86_64-cpu"
                fi
                ;;
            aarch64|arm64)
                err "Linux ARM is not supported yet."
                ;;
            *)
                err "Unsupported architecture: ${ARCH}"
                ;;
        esac
        ;;
    *)
        err "Unsupported OS: ${OS}"
        ;;
esac

case "${CHANNEL}" in
    nightly)
        RLLM_VERSION="nightly"
        say "Installing nightly (${PLATFORM})..."
        ;;
    stable)
        if [ -n "${RLLM_VERSION:-}" ]; then
            say "Installing ${RLLM_VERSION} (${PLATFORM})..."
        else
            say "Fetching latest stable release..."
            RLLM_VERSION="$(
                curl -fsSL "${GITHUB_API}/releases/latest" \
                | sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
                | head -1
            )"
            [ -n "${RLLM_VERSION}" ] || err "Could not determine latest release tag. Set RLLM_VERSION manually."
            say "Installing ${RLLM_VERSION} (${PLATFORM})..."
        fi
        ;;
    *)
        err "RLLM_CHANNEL must be 'stable' or 'nightly'."
        ;;
esac

TARBALL="rllm-${PLATFORM}.tar.gz"
CHECKSUM="${TARBALL}.sha256"
BASE_URL="${GITHUB_RELEASES}/download/${RLLM_VERSION}"

TMPDIR="$(mktemp -d)"
cleanup() {
    rm -rf "${TMPDIR}"
}
trap cleanup EXIT

say "Downloading ${TARBALL}..."
curl -fsSL --progress-bar "${BASE_URL}/${TARBALL}" -o "${TMPDIR}/${TARBALL}"
curl -fsSL "${BASE_URL}/${CHECKSUM}" -o "${TMPDIR}/${CHECKSUM}"

say "Verifying checksum..."
(
    cd "${TMPDIR}"
    if available sha256sum; then
        sha256sum -c "${CHECKSUM}" >/dev/null
    elif available shasum; then
        shasum -a 256 -c "${CHECKSUM}" >/dev/null
    else
        warn "No sha256 tool found; skipping checksum verification."
    fi
)

say "Extracting archive..."
tar -xzf "${TMPDIR}/${TARBALL}" -C "${TMPDIR}"
[ -f "${TMPDIR}/rllm" ] || err "Binary 'rllm' not found in archive."
chmod +x "${TMPDIR}/rllm"

if [ "${OS}" = "Linux" ] && available systemctl && systemctl list-unit-files 2>/dev/null | grep -q '^rllm\.service'; then
    if as_root systemctl is-active --quiet rllm 2>/dev/null; then
        say "Stopping running rllm service for upgrade..."
        as_root systemctl stop rllm || true
    fi
elif [ "${OS}" = "Darwin" ] && [ -f "${HOME}/Library/LaunchAgents/com.rllm.rllmd.plist" ]; then
    say "Stopping running rllm launchd agent for upgrade..."
    launchctl bootout "gui/$(id -u)" "${HOME}/Library/LaunchAgents/com.rllm.rllmd.plist" 2>/dev/null || true
fi

if [ ! -d "${INSTALL_DIR}" ]; then
    if [ -w "$(dirname "${INSTALL_DIR}")" ]; then
        mkdir -p "${INSTALL_DIR}"
    else
        as_root mkdir -p "${INSTALL_DIR}"
    fi
fi

DEST="${INSTALL_DIR}/rllm"
if [ -w "${INSTALL_DIR}" ]; then
    mv "${TMPDIR}/rllm" "${DEST}"
else
    say "Installing to ${INSTALL_DIR} (may ask for password)..."
    as_root mv "${TMPDIR}/rllm" "${DEST}"
fi
chmod +x "${DEST}" 2>/dev/null || as_root chmod +x "${DEST}"

if [ "${OS}" = "Linux" ]; then
    install_systemd_service
elif [ "${OS}" = "Darwin" ]; then
    install_launchd_agent
fi

INSTALLED_VER="$(${DEST} --version 2>/dev/null | head -1 || echo unknown)"

echo ""
say "rLLM installed successfully."
echo ""
echo "Binary  : ${DEST}"
echo "Version : ${INSTALLED_VER}"
echo "Channel : ${CHANNEL}"
echo "Backend : ${PLATFORM}"
echo ""

if [ "${OS}" = "Linux" ] && available systemctl; then
    echo "Service management:"
    echo "  sudo systemctl status rllm"
    echo "  sudo systemctl restart rllm"
    echo "  sudo systemctl stop rllm"
    echo "  sudo journalctl -u rllm -f"
    echo ""
elif [ "${OS}" = "Darwin" ]; then
    echo "launchd service management:"
    echo "  launchctl kickstart -k gui/$(id -u)/com.rllm.rllmd"
    echo "  launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.rllm.rllmd.plist"
    echo "  launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.rllm.rllmd.plist"
    echo "Logs: ~/.rllm/logs/rllm.log"
    echo ""
fi

echo "Quick start:"
echo "  rllm pull Qwen/Qwen3-0.6B"
echo "  rllm run Qwen3-0.6B"
echo ""

case ":${PATH}:" in
    *":${INSTALL_DIR}:"*) ;;
    *)
        warn "'${INSTALL_DIR}' is not in PATH. Add: export PATH=\"${INSTALL_DIR}:\$PATH\""
        ;;
esac
}

main "$@"
