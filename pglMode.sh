#!/bin/bash

PGL_DIR="$HOME/.pgl"
STATE_FILE="$PGL_DIR/pglModeState"
CAFFEINATE_PID_FILE="$PGL_DIR/caffeinatePID"

# Ensure .pgl directory exists
mkdir -p "$PGL_DIR"

showHelp() {
cat << EOF

pglMode.sh - Enable or restore PGL mode on macOS

DESCRIPTION
  PGL mode temporarily disables background services that may
  interfere with precise stimulus timing, including:

    - Time Machine backups
    - Spotlight indexing (main volume only)
    - WiFi
    - System and display sleep
    - Enables caffeinate

  The previous system state is saved and can be restored.

USAGE
  ./pglMode.sh enable       Enable PGL mode (checks for Full Disk Access)
  ./pglMode.sh enable force Enable PGL mode even if Full Disk Access is not granted
  ./pglMode.sh restore      Restore or disable PGL mode
  ./pglMode.sh disable      Same as restore
  ./pglMode.sh help         Show this help message
  ./pglMode.sh -h
  ./pglMode.sh --help

NOTES
  - State files are stored in: $PGL_DIR

EOF
}

checkFullDiskAccess() {
    TMP=$(mktemp)
    echo "Checking Full Disk Access (you may be prompted for your administrator password)..."
    sudo tmutil disable -n &> "$TMP"
    if grep -q "requires Full Disk Access" "$TMP"; then
        rm "$TMP"
        return 1
    fi
    rm "$TMP"
    return 0
}

enableMode() {
    FORCE_MODE="$1"

    if [ "$FORCE_MODE" != "force" ]; then
        if ! checkFullDiskAccess; then
            echo "ERROR: Full Disk Access is required for controlling Time Machine."
            echo "To enable it, go to:"
            echo "  System Settings → Privacy & Security → Full Disk Access"
            echo "  Add Terminal (or your shell app) to the list and restart it."
            echo
            echo "If you really want to bypass Full Disk Access, run:"
            echo "  ./pglMode.sh enable force"
            exit 1
        fi
    else
        echo "WARNING: Running PGL mode without Full Disk Access."
        echo "Time Machine control may fail, background disk activity could occur,"
        echo "and precise timing may be slightly less reliable."
    fi

    echo "Saving current system state for PGL mode..."

    # Trim numbers and quote strings for safe sourcing
    TM_ENABLED=$(tmutil destinationinfo 2>/dev/null | wc -l | tr -d '[:space:]')
    SPOTLIGHT_STATUS=$(mdutil -s / | grep "Indexing enabled")
    SPOTLIGHT_STATUS_ESCAPED=$(printf '%q' "$SPOTLIGHT_STATUS")
    WIFI_DEVICE=$(networksetup -listallhardwareports | awk '/Wi-Fi/{getline; print $2}')
    WIFI_STATUS=$(networksetup -getairportpower "$WIFI_DEVICE" | awk '{print $4}')
    SLEEP_SETTING=$(pmset -g | grep " sleep " | awk '{print $2}' | tr -d '[:space:]')
    DISPLAY_SLEEP_SETTING=$(pmset -g | grep " displaysleep " | awk '{print $2}' | tr -d '[:space:]')

    echo "TM_ENABLED=$TM_ENABLED" > "$STATE_FILE"
    echo "SPOTLIGHT_STATUS=$SPOTLIGHT_STATUS_ESCAPED" >> "$STATE_FILE"
    echo "WIFI_DEVICE=$(printf '%q' "$WIFI_DEVICE")" >> "$STATE_FILE"
    echo "WIFI_STATUS=$(printf '%q' "$WIFI_STATUS")" >> "$STATE_FILE"
    echo "SLEEP_SETTING=$SLEEP_SETTING" >> "$STATE_FILE"
    echo "DISPLAY_SLEEP_SETTING=$DISPLAY_SLEEP_SETTING" >> "$STATE_FILE"

    echo
    echo "Configuring PGL mode:"
    echo

    TM_SUCCESS="✓"
    SPOTLIGHT_SUCCESS="✓"
    
    if checkFullDiskAccess || [ "$FORCE_MODE" == "force" ]; then
        echo "  • Time Machine: disabling..."
        if sudo tmutil disable &>/dev/null; then
            TM_SUCCESS="✓"
        else
            TM_SUCCESS="✗"
        fi
    fi

    echo "  • Spotlight: disabling on main volume..."
    if sudo mdutil -i off / &>/dev/null; then
        SPOTLIGHT_SUCCESS="✓"
    else
        SPOTLIGHT_SUCCESS="✗"
    fi

    echo "  • WiFi: turning off..."
    networksetup -setairportpower "$WIFI_DEVICE" off

    echo "  • System sleep: disabling..."
    sudo pmset -a sleep 0
    sudo pmset -a displaysleep 0

    echo "  • Caffeinate: starting..."
    caffeinate -d -i -m -s &
    echo $! > "$CAFFEINATE_PID_FILE"

    echo
    echo "PGL mode enabled:"
    echo
    echo "  $TM_SUCCESS Time Machine: off"
    echo "  $SPOTLIGHT_SUCCESS Spotlight indexing: off"
    echo "  ✓ WiFi: off"
    echo "  ✓ System sleep: disabled"
    echo "  ✓ Display sleep: disabled"
    echo "  ✓ Caffeinate: running"
    echo
}

restoreMode() {
    if [ ! -f "$STATE_FILE" ]; then
        echo "No saved PGL mode state found."
        exit 1
    fi

    # Safely source the state
    source "$STATE_FILE"

    echo "Restoring state from PGL mode:"
    echo

    if [ "$TM_ENABLED" -gt 0 ]; then
        echo "  • Time Machine: enabling..."
        sudo tmutil enable &>/dev/null || echo "    Warning: Could not enable Time Machine"
    else
        echo "  • Time Machine: was disabled, leaving disabled"
    fi

    if [[ "$SPOTLIGHT_STATUS" == *"enabled"* ]]; then
        echo "  • Spotlight: enabling..."
        sudo mdutil -i on / &>/dev/null || echo "    Warning: Could not enable Spotlight on /"
    else
        echo "  • Spotlight: was disabled, leaving disabled"
    fi

    if [[ "$WIFI_STATUS" == "On" ]]; then
        echo "  • WiFi: turning on..."
        networksetup -setairportpower "$WIFI_DEVICE" on
    else
        echo "  • WiFi: was off, leaving off"
    fi

    echo "  • Sleep settings: restoring (sleep: ${SLEEP_SETTING}min, display: ${DISPLAY_SLEEP_SETTING}min)..."
    sudo pmset -a sleep "$SLEEP_SETTING"
    sudo pmset -a displaysleep "$DISPLAY_SLEEP_SETTING"

    if [ -f "$CAFFEINATE_PID_FILE" ]; then
        echo "  • Caffeinate: stopping..."
        kill "$(cat "$CAFFEINATE_PID_FILE")" 2>/dev/null
        rm "$CAFFEINATE_PID_FILE"
    fi

    rm "$STATE_FILE"

    echo
    echo "System restored from PGL mode."
}

# Main
case "$1" in
    enable)
        enableMode "$2"
        ;;
    restore|disable)
        restoreMode
        ;;
    help|-h|--help|"")
        showHelp
        ;;
    *)
        echo "Unknown command: $1"
        echo
        showHelp
        exit 1
        ;;
esac