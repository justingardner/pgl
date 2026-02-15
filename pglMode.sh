#!/bin/bash

PGL_DIR="$HOME/.pgl"
STATE_FILE="$PGL_DIR/pglModeState"
CAFFEINATE_PID_FILE="$PGL_DIR/caffeinatePID"

mkdir -p "$PGL_DIR"

showHelp() {
cat << EOF

pglMode.sh - Enable or restore PGL mode on macOS

DESCRIPTION
  PGL mode temporarily disables background services that may
  interfere with precise stimulus timing.

USAGE
  ./pglMode.sh enable [flags]   Enable PGL mode (default is all)
  ./pglMode.sh restore [flags]  Restore previous system state (default is all)

FLAGS
  -a   Adjust all settings (default)
  -d   Do Not Disturb only
  -w   WiFi only
  -b   Bluetooth only
  -t   Time Machine only
  -s   Spotlight only
  -p   Power Nap only
  -n   Network time sync only
  -h   Show help

EOF
}

enableDoNotDisturb() {
    if ! shortcuts run "Enable pglMode" 2>/dev/null; then
        echo "Error: 'Enable pglMode' shortcut not found" >&2
        return 1
    fi
}

disableDoNotDisturb() {
    if ! shortcuts run "Disable pglMode" 2>/dev/null; then
        echo "Error: 'Disable pglMode' shortcut not found" >&2
        return 1
    fi
}

getCurrentFocus() {
    val=$(defaults -currentHost read ~/Library/Preferences/ByHost/com.apple.notificationcenterui doNotDisturb 2>/dev/null || echo 0)
    if [[ "$val" == "1" ]]; then echo "On"; else echo "Off"; fi
}

# Flag defaults
ADJUST_ALL=true
ADJUST_DND=false
ADJUST_WIFI=false
ADJUST_BLUETOOTH=false
ADJUST_TIMEMACHINE=false
ADJUST_SPOTLIGHT=false
ADJUST_SLEEP=false
ADJUST_POWERNAP=false
ADJUST_NETWORKTIME=false

# Parse command
COMMAND="$1"
shift

# Parse flags
while getopts "adwbtspnh" opt; do
    case "$opt" in
        a) ADJUST_ALL=true ;;
        d) ADJUST_DND=true; ADJUST_ALL=false ;;
        w) ADJUST_WIFI=true; ADJUST_ALL=false ;;
        b) ADJUST_BLUETOOTH=true; ADJUST_ALL=false ;;
        t) ADJUST_TIMEMACHINE=true; ADJUST_ALL=false ;;
        s) ADJUST_SPOTLIGHT=true; ADJUST_ALL=false ;;
        p) ADJUST_POWERNAP=true; ADJUST_ALL=false ;;
        n) ADJUST_NETWORKTIME=true; ADJUST_ALL=false ;;
        h) showHelp; exit 0 ;;
        *) echo "Invalid option: -$OPTARG"; exit 1 ;;
    esac
done
shift $((OPTIND-1))

checkFullDiskAccess() {
    TMP=$(mktemp)
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

    if [ -f "$STATE_FILE" ]; then
        echo "ERROR: PGL mode is already enabled!"
        exit 1
    fi

    if [ "$FORCE_MODE" != "force" ] && ! checkFullDiskAccess; then
        echo "ERROR: Full Disk Access is required for controlling Time Machine."
        exit 1
    fi

    echo "Saving current system state..."
    TM_ENABLED=$(tmutil destinationinfo 2>/dev/null | wc -l | tr -d '[:space:]')
    SPOTLIGHT_STATUS=$(mdutil -s / | grep "Indexing enabled")
    WIFI_DEVICE=$(networksetup -listallhardwareports | awk '/Wi-Fi/{getline; print $2}')
    WIFI_STATUS=$(networksetup -getairportpower "$WIFI_DEVICE" 2>/dev/null | awk '{print $4}')
    SLEEP_SETTING=$(pmset -g | grep " sleep " | awk '{print $2}' | tr -d '[:space:]')
    DISPLAY_SLEEP_SETTING=$(pmset -g | grep " displaysleep " | awk '{print $2}' | tr -d '[:space:]')
    POWERNAP_SETTING=$(pmset -g | grep " powernap " | awk '{print $2}' | tr -d '[:space:]')
    NETWORK_TIME=$(systemsetup -getusingnetworktime 2>/dev/null | grep -o "On\|Off")
    BLUETOOTH_POWER=$(defaults read /Library/Preferences/com.apple.Bluetooth ControllerPowerState 2>/dev/null || echo "1")
    FOCUS_STATUS=$(getCurrentFocus)

    # Save state
    echo "TM_ENABLED=$TM_ENABLED" > "$STATE_FILE"
    echo "SPOTLIGHT_STATUS=\"$SPOTLIGHT_STATUS\"" >> "$STATE_FILE"
    echo "WIFI_DEVICE=\"$WIFI_DEVICE\"" >> "$STATE_FILE"
    echo "WIFI_STATUS=\"$WIFI_STATUS\"" >> "$STATE_FILE"
    echo "SLEEP_SETTING=$SLEEP_SETTING" >> "$STATE_FILE"
    echo "DISPLAY_SLEEP_SETTING=$DISPLAY_SLEEP_SETTING" >> "$STATE_FILE"
    echo "POWERNAP_SETTING=$POWERNAP_SETTING" >> "$STATE_FILE"
    echo "NETWORK_TIME=$NETWORK_TIME" >> "$STATE_FILE"
    echo "BLUETOOTH_POWER=$BLUETOOTH_POWER" >> "$STATE_FILE"
    echo "FOCUS_STATUS=$FOCUS_STATUS" >> "$STATE_FILE"

    echo "Applying PGL mode settings..."

    [[ "$ADJUST_ALL" == true || "$ADJUST_DND" == true ]] && [[ "$FOCUS_STATUS" == "Off" ]] && enableDoNotDisturb
    [[ "$ADJUST_ALL" == true || "$ADJUST_WIFI" == true ]] && [[ -n "$WIFI_DEVICE" ]] && networksetup -setairportpower "$WIFI_DEVICE" off
    [[ "$ADJUST_ALL" == true || "$ADJUST_BLUETOOTH" == true ]] && sudo defaults write /Library/Preferences/com.apple.Bluetooth ControllerPowerState -int 0 2>/dev/null && sudo killall -HUP blued 2>/dev/null
    [[ "$ADJUST_ALL" == true || "$ADJUST_TIMEMACHINE" == true ]] && [[ "$FORCE_MODE" == "force" ]] || sudo tmutil disable &>/dev/null
    [[ "$ADJUST_ALL" == true || "$ADJUST_SPOTLIGHT" == true ]] && sudo mdutil -i off / &>/dev/null
    [[ "$ADJUST_ALL" == true || "$ADJUST_POWERNAP" == true ]] && sudo pmset -a powernap 0 &>/dev/null
    [[ "$ADJUST_ALL" == true || "$ADJUST_NETWORKTIME" == true ]] && sudo systemsetup -setusingnetworktime off &>/dev/null
    [[ "$ADJUST_ALL" == true || "$ADJUST_SLEEP" == true ]] && sudo pmset -a sleep 0 displaysleep 0
    
    caffeinate -d -i -m -s &
    echo $! > "$CAFFEINATE_PID_FILE"

    echo "PGL mode enabled."
}

restoreMode() {
    RESTORE_ALL="$1"
    if [ ! -f "$STATE_FILE" ]; then echo "No saved PGL mode state found."; exit 1; fi
    source "$STATE_FILE"

    # Default to all if no flags
    if [[ "$ADJUST_ALL" != true && "$ADJUST_DND" != true && "$ADJUST_WIFI" != true && "$ADJUST_BLUETOOTH" != true && \
          "$ADJUST_TIMEMACHINE" != true && "$ADJUST_SPOTLIGHT" != true && "$ADJUST_POWERNAP" != true && \
          "$ADJUST_NETWORKTIME" != true && "$ADJUST_SLEEP" != true ]]; then
        ADJUST_ALL=true
    fi

    echo "Restoring system settings..."

    SUMMARY=""

    if [[ "$ADJUST_ALL" == true || "$ADJUST_DND" == true ]]; then
        disableDoNotDisturb
        SUMMARY+="Focus mode: Do Not Disturb → Off\n"
    fi
    if [[ "$ADJUST_ALL" == true || "$ADJUST_WIFI" == true ]]; then
        [[ -n "$WIFI_DEVICE" ]] && networksetup -setairportpower "$WIFI_DEVICE" on
        SUMMARY+="WiFi → On\n"
    fi
    if [[ "$ADJUST_ALL" == true || "$ADJUST_BLUETOOTH" == true ]]; then
        sudo defaults write /Library/Preferences/com.apple.Bluetooth ControllerPowerState -int 1 2>/dev/null
        sudo killall -HUP blued 2>/dev/null
        SUMMARY+="Bluetooth → On\n"
    fi
    if [[ "$ADJUST_ALL" == true || "$ADJUST_TIMEMACHINE" == true ]]; then
        sudo tmutil enable &>/dev/null
        SUMMARY+="Time Machine → Enabled\n"
    fi
    if [[ "$ADJUST_ALL" == true || "$ADJUST_SPOTLIGHT" == true ]]; then
        sudo mdutil -i on / &>/dev/null
        SUMMARY+="Spotlight → Enabled\n"
    fi
    if [[ "$ADJUST_ALL" == true || "$ADJUST_POWERNAP" == true ]]; then
        sudo pmset -a powernap "$POWERNAP_SETTING" &>/dev/null
        SUMMARY+="Power Nap → $POWERNAP_SETTING\n"
    fi
    if [[ "$ADJUST_ALL" == true || "$ADJUST_NETWORKTIME" == true ]]; then
        sudo systemsetup -setusingnetworktime on &>/dev/null
        SUMMARY+="Network time sync → On\n"
    fi
    if [[ "$ADJUST_ALL" == true || "$ADJUST_SLEEP" == true ]]; then
        sudo pmset -a sleep "$SLEEP_SETTING" displaysleep "$DISPLAY_SLEEP_SETTING"
        SUMMARY+="Sleep → $SLEEP_SETTING min, Display → $DISPLAY_SLEEP_SETTING min\n"
    fi

    if [ -f "$CAFFEINATE_PID_FILE" ]; then
        kill "$(cat "$CAFFEINATE_PID_FILE")" 2>/dev/null
        rm "$CAFFEINATE_PID_FILE"
        SUMMARY+="Caffeinate → Stopped\n"
    fi

    rm "$STATE_FILE"

    echo -e "\nSystem restored. Summary:"
    echo -e "$SUMMARY"
}

# Main
case "$COMMAND" in
    enable) enableMode "$1" ;;
    restore|disable) restoreMode "$1" ;;
    help|-h|--help|"") showHelp ;;
    *) echo "Unknown command: $COMMAND"; showHelp; exit 1 ;;
esac
