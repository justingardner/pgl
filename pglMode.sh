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
    - Bluetooth
    - System and display sleep
    - Power Nap
    - Network time synchronization
    - Focus mode (enables Do Not Disturb)
    - Enables caffeinate

  The previous system state is saved and can be restored.

USAGE
  ./pglMode.sh enable       Enable PGL mode (checks for Full Disk Access)
  ./pglMode.sh enable force Enable PGL mode even if Full Disk Access is not granted
  ./pglMode.sh restore      Restore previous system state
  ./pglMode.sh restore all  Restore and turn everything on (ignore saved state)
  ./pglMode.sh disable      Same as restore
  ./pglMode.sh help         Show this help message
  ./pglMode.sh -h
  ./pglMode.sh --help

NOTES
  - State files are stored in: $PGL_DIR
  - Cannot enable PGL mode twice - must restore first

EOF
}

showStatus() {
    if [ ! -f "$STATE_FILE" ]; then
        showHelp
        return
    fi

    # PGL mode is active
    source "$STATE_FILE"

    echo
    echo "⚠️  PGL mode is currently ENABLED"
    echo
    echo "To restore your system to its previous state, run:"
    echo "  ./pglMode.sh restore"
    echo
    echo "To restore and turn everything on (ignore previous state), run:"
    echo "  ./pglMode.sh restore all"
    echo
    echo "This will restore the following settings:"
    echo

    if [ "$TM_ENABLED" -gt 0 ]; then
        echo "  • Time Machine: will be enabled"
    else
        echo "  • Time Machine: will remain disabled (was off before PGL mode)"
    fi

    if [[ "$SPOTLIGHT_STATUS" == *"enabled"* ]]; then
        echo "  • Spotlight: will be enabled"
    else
        echo "  • Spotlight: will remain disabled (was off before PGL mode)"
    fi

    if [[ "$WIFI_STATUS" == "On" ]]; then
        echo "  • WiFi: will be turned on"
    else
        echo "  • WiFi: will remain off (was off before PGL mode)"
    fi

    if [[ "$BLUETOOTH_POWER" == "1" ]]; then
        echo "  • Bluetooth: will be turned on"
    else
        echo "  • Bluetooth: will remain off (was off before PGL mode)"
    fi

    echo "  • Sleep settings: sleep after $${SLEEP_SETTING} min, display sleep after $${DISPLAY_SLEEP_SETTING} min"
    
    echo "  • Power Nap: will be set to $POWERNAP_SETTING"

    if [[ "$NETWORK_TIME" == "On" ]]; then
        echo "  • Network time sync: will be enabled"
    else
        echo "  • Network time sync: will remain disabled (was off before PGL mode)"
    fi

    if [[ "$FOCUS_STATUS" == "Off" ]]; then
        echo "  • Focus mode: Do Not Disturb will be disabled"
    else
        echo "  • Focus mode: will remain enabled (was on before PGL mode)"
    fi

    echo "  • Caffeinate: will be stopped"
    echo
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

# Simpler, more reliable Focus mode handling
getCurrentFocus() {
    # Check if Focus is enabled by reading the current assertion
    local focus_status=$(plutil -extract dnd_prefs raw -o - ~/Library/Preferences/com.apple.ncprefs.plist 2>/dev/null | grep -o "userPref.*1" | head -1)
    if [[ -n "$focus_status" ]]; then
        echo "On"
    else
        echo "Off"
    fi
}

enableDoNotDisturb() {
    osascript -e 'tell application "System Events" to keystroke "D" using {command down, shift down, option down, control down}' 2>/dev/null
    return $?
}

disableDoNotDisturb() {
    osascript -e 'tell application "System Events" to keystroke "D" using {command down, shift down, option down, control down}' 2>/dev/null
    return $?
}

enableMode() {
    FORCE_MODE="$1"

    # Check if PGL mode is already enabled
    if [ -f "$STATE_FILE" ]; then
        echo "ERROR: PGL mode is already enabled!"
        echo
        echo "You must restore the previous state before enabling again."
        echo "This prevents overwriting your original system settings."
        echo
        showStatus
        exit 1
    fi

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
    WIFI_STATUS=$(networksetup -getairportpower "$WIFI_DEVICE" 2>/dev/null | awk '{print $4}')
    SLEEP_SETTING=$(pmset -g | grep " sleep " | awk '{print $2}' | tr -d '[:space:]')
    DISPLAY_SLEEP_SETTING=$(pmset -g | grep " displaysleep " | awk '{print $2}' | tr -d '[:space:]')
    POWERNAP_SETTING=$(pmset -g | grep " powernap " | awk '{print $2}' | tr -d '[:space:]')
    
    # Network time
    NETWORK_TIME=$(systemsetup -getusingnetworktime 2>/dev/null | grep -o "On\|Off")
    
    # Bluetooth
    BLUETOOTH_POWER=$(defaults read /Library/Preferences/com.apple.Bluetooth ControllerPowerState 2>/dev/null || echo "1")
    
    # Focus mode - simplified approach
    FOCUS_STATUS=$(getCurrentFocus)

    # Save state
    echo "TM_ENABLED=$TM_ENABLED" > "$STATE_FILE"
    echo "SPOTLIGHT_STATUS=$SPOTLIGHT_STATUS_ESCAPED" >> "$STATE_FILE"
    echo "WIFI_DEVICE=$(printf '%q' "$WIFI_DEVICE")" >> "$STATE_FILE"
    echo "WIFI_STATUS=$(printf '%q' "$WIFI_STATUS")" >> "$STATE_FILE"
    echo "SLEEP_SETTING=$SLEEP_SETTING" >> "$STATE_FILE"
    echo "DISPLAY_SLEEP_SETTING=$DISPLAY_SLEEP_SETTING" >> "$STATE_FILE"
    echo "POWERNAP_SETTING=$POWERNAP_SETTING" >> "$STATE_FILE"
    echo "NETWORK_TIME=$NETWORK_TIME" >> "$STATE_FILE"
    echo "BLUETOOTH_POWER=$BLUETOOTH_POWER" >> "$STATE_FILE"
    echo "FOCUS_STATUS=$FOCUS_STATUS" >> "$STATE_FILE"

    echo
    echo "Configuring PGL mode:"
    echo

    TM_SUCCESS="✓"
    SPOTLIGHT_SUCCESS="✓"
    POWERNAP_SUCCESS="✓"
    NETWORK_TIME_SUCCESS="✓"
    BLUETOOTH_SUCCESS="✓"
    FOCUS_SUCCESS="✓"
    
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

    if [[ -n "$WIFI_DEVICE" ]]; then
        echo "  • WiFi: turning off..."
        networksetup -setairportpower "$WIFI_DEVICE" off
    fi

    echo "  • Bluetooth: turning off..."
    if sudo defaults write /Library/Preferences/com.apple.Bluetooth ControllerPowerState -int 0 2>/dev/null && \
       sudo killall -HUP blued 2>/dev/null; then
        BLUETOOTH_SUCCESS="✓"
    else
        BLUETOOTH_SUCCESS="✗"
    fi

    echo "  • System sleep: disabling..."
    sudo pmset -a sleep 0
    sudo pmset -a displaysleep 0

    echo "  • Power Nap: disabling..."
    if sudo pmset -a powernap 0 &>/dev/null; then
        POWERNAP_SUCCESS="✓"
    else
        POWERNAP_SUCCESS="✗"
    fi

    echo "  • Network time sync: disabling..."
    if sudo systemsetup -setusingnetworktime off &>/dev/null; then
        NETWORK_TIME_SUCCESS="✓"
    else
        NETWORK_TIME_SUCCESS="✗"
    fi

    echo "  • Focus mode: enabling Do Not Disturb..."
    if [[ "$FOCUS_STATUS" == "Off" ]]; then
        if enableDoNotDisturb; then
            FOCUS_SUCCESS="✓"
        else
            FOCUS_SUCCESS="✗"
        fi
    else
        FOCUS_SUCCESS="✓ (already on)"
    fi

    echo "  • Caffeinate: starting..."
    caffeinate -d -i -m -s &
    echo $! > "$CAFFEINATE_PID_FILE"

    echo
    echo "PGL mode enabled:"
    echo
    echo "  $TM_SUCCESS Time Machine: off"
    echo "  $SPOTLIGHT_SUCCESS Spotlight indexing: off"
    echo "  ✓ WiFi: off"
    echo "  $BLUETOOTH_SUCCESS Bluetooth: off"
    echo "  ✓ System sleep: disabled"
    echo "  ✓ Display sleep: disabled"
    echo "  $POWERNAP_SUCCESS Power Nap: disabled"
    echo "  $NETWORK_TIME_SUCCESS Network time sync: disabled"
    echo "  $FOCUS_SUCCESS Focus: Do Not Disturb"
    echo "  ✓ Caffeinate: running"
    echo
}
restoreMode() {
    RESTORE_ALL="$1"

    if [ ! -f "$STATE_FILE" ]; then
        echo "No saved PGL mode state found."
        exit 1
    fi

    # Safely source the state
    source "$STATE_FILE"

    if [[ "$RESTORE_ALL" == "all" ]]; then
        echo "Restoring all settings to ON (ignoring previous state):"
    else
        echo "Restoring state from PGL mode:"
    fi
    echo

    if [[ "$RESTORE_ALL" == "all" ]] || [ "$TM_ENABLED" -gt 0 ]; then
        echo "  • Time Machine: enabling..."
        sudo tmutil enable &>/dev/null || echo "    Warning: Could not enable Time Machine"
    else
        echo "  • Time Machine: was disabled, leaving disabled"
    fi

    if [[ "$RESTORE_ALL" == "all" ]] || [[ "$SPOTLIGHT_STATUS" == *"enabled"* ]]; then
        echo "  • Spotlight: enabling..."
        sudo mdutil -i on / &>/dev/null || echo "    Warning: Could not enable Spotlight on /"
    else
        echo "  • Spotlight: was disabled, leaving disabled"
    fi

    if [[ "$RESTORE_ALL" == "all" ]] || [[ "$WIFI_STATUS" == "On" ]]; then
        echo "  • WiFi: turning on..."
        networksetup -setairportpower "$WIFI_DEVICE" on
    else
        echo "  • WiFi: was off, leaving off"
    fi

    if [[ "$RESTORE_ALL" == "all" ]] || [[ "$BLUETOOTH_POWER" == "1" ]]; then
        echo "  • Bluetooth: turning on..."
        sudo defaults write /Library/Preferences/com.apple.Bluetooth ControllerPowerState -int 1 2>/dev/null
        sudo killall -HUP blued 2>/dev/null
    else
        echo "  • Bluetooth: was off, leaving off"
    fi

    if [[ "$RESTORE_ALL" == "all" ]]; then
        echo "  • Sleep settings: restoring to defaults (sleep: 10min, display: 2min)..."
        sudo pmset -a sleep 10
        sudo pmset -a displaysleep 2
    else
        echo "  • Sleep settings: restoring (sleep: ${SLEEP_SETTING}min, display: ${DISPLAY_SLEEP_SETTING}min)..."
        sudo pmset -a sleep "$SLEEP_SETTING"
        sudo pmset -a displaysleep "$DISPLAY_SLEEP_SETTING"
    fi

    if [[ "$RESTORE_ALL" == "all" ]]; then
        echo "  • Power Nap: enabling..."
        sudo pmset -a powernap 1
    else
        echo "  • Power Nap: restoring to $POWERNAP_SETTING..."
        sudo pmset -a powernap "$POWERNAP_SETTING"
    fi

    if [[ "$RESTORE_ALL" == "all" ]] || [[ "$NETWORK_TIME" == "On" ]]; then
        echo "  • Network time sync: enabling..."
        sudo systemsetup -setusingnetworktime on &>/dev/null
    else
        echo "  • Network time sync: was off, leaving off"
    fi

    if [[ "$RESTORE_ALL" == "all" ]] || [[ "$FOCUS_STATUS" == "Off" ]]; then
        echo "  • Focus mode: disabling Do Not Disturb..."
        disableDoNotDisturb
    else
        echo "  • Focus mode: was on, leaving enabled"
    fi

    if [ -f "$CAFFEINATE_PID_FILE" ]; then
        echo "  • Caffeinate: stopping..."
        kill "$(cat "$CAFFEINATE_PID_FILE")" 2>/dev/null
        rm "$CAFFEINATE_PID_FILE"
    fi

    rm "$STATE_FILE"

    echo
    if [[ "$RESTORE_ALL" == "all" ]]; then
        echo "System restored from PGL mode with all settings enabled."
    else
        echo "System restored from PGL mode."
    fi
}
# Main
case "$1" in
    enable)
        enableMode "$2"
        ;;
    restore|disable)
        restoreMode "$2"
        ;;
    help|-h|--help)
        showHelp
        ;;
    "")
        showStatus
        ;;
    *)
        echo "Unknown command: $1"
        echo
        showHelp
        exit 1
        ;;
esac