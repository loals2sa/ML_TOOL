#!/bin/bash

# Fouad Tool Launcher Script
# Quick launcher for both terminal and GUI modes

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOL_PATH="$SCRIPT_DIR/redteam/fouad.py"

# Check if tool exists
if [ ! -f "$TOOL_PATH" ]; then
    echo "❌ Fouad Tool not found at: $TOOL_PATH"
    exit 1
fi

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    exit 1
fi

# Parse command line arguments
MODE="terminal"
for arg in "$@"; do
    case $arg in
        --gui|-g)
            MODE="gui"
            ;;
        --help|-h)
            echo "🔥 FOUAD TOOL LAUNCHER 🔥"
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --gui, -g    Launch in GUI mode"
            echo "  --help, -h   Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0           # Launch terminal mode"
            echo "  $0 --gui     # Launch GUI mode"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Launch the tool
echo "🔥 Launching Fouad Tool in $MODE mode..."
echo ""

if [ "$MODE" = "gui" ]; then
    python3 "$TOOL_PATH" --gui
else
    python3 "$TOOL_PATH"
fi
