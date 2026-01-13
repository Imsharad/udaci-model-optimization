#!/bin/bash

# =============================================================================
# GOLD STANDARD: run_auto_review.sh
# Master orchestration script for automated Udacity project review
# =============================================================================
# USAGE: ./run_auto_review.sh
# 
# This script:
# 1. Runs setup_next_student_optimized.sh to prepare workspace
# 2. Invokes gemini agent with notes.txt prompt
# 3. Verifies feedback generation
# =============================================================================

# Configuration - UPDATE THIS FOR EACH PROJECT
SETUP_SCRIPT="./setup_next_student_optimized.sh"
NOTES_FILE="notes.txt"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_step() { echo -e "${BLUE}[STEP]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# macOS Notification
notify_macos() {
    local title="$1"
    local message="$2"
    osascript -e "display notification \"$message\" with title \"$title\" sound name \"Submarine\""
}

# Check prerequisites
if [[ ! -f "$SETUP_SCRIPT" ]]; then
    print_error "Setup script not found: $SETUP_SCRIPT"
    exit 1
fi

if [[ ! -f "$NOTES_FILE" ]]; then
    print_error "Notes file not found: $NOTES_FILE"
    exit 1
fi

if ! command -v gemini &> /dev/null; then
    print_error "'gemini' CLI not found in path."
    exit 1
fi

# Step 1: Run setup script to prepare next student
print_step "Running setup script..."
OUTPUT=$($SETUP_SCRIPT "$@")
EXIT_CODE=$?

# Print output for user visibility
echo "$OUTPUT"

# Warn if setup failed, but continue anyway to give Gemini a chance
if [[ $EXIT_CODE -ne 0 ]]; then
    print_warning "Setup script reported errors, but continuing to run review..."
fi

# Extract the new student directory from the output
# Robustly parse the log message "Creating stu_XXX" to get the exact directory name
NEW_DIR_NAME=$(echo "$OUTPUT" | grep -o "Creating stu_[0-9]*" | head -n 1 | awk '{print $2}')

if [[ -n "$NEW_DIR_NAME" ]]; then
    NEW_DIR="./$NEW_DIR_NAME"
else
    # If we can't detect from output, try to find the latest stu_* directory
    print_warning "Could not detect directory from logs, searching for latest..."
    LATEST_STU=$(ls -dt stu_* 2>/dev/null | grep -v "stu_template" | head -1)
    if [[ -n "$LATEST_STU" ]]; then
        NEW_DIR="./$LATEST_STU"
        print_info "Found: $LATEST_STU"
    else
        print_error "Could not find any student directory to review."
        exit 1
    fi
fi

STUDENT_ID=$(basename "$NEW_DIR")
print_info "Targeting student directory: $STUDENT_ID"

# Step 2: Navigate and Run Automation
print_step "Starting Automated Review for $STUDENT_ID..."

cd "$NEW_DIR" || exit 1

# Start timer
START_TIME=$(date +%s)

# Run Gemini in headless/YOLO mode with notes.txt content
print_info "Invoking Gemini agent..."
cat "../$NOTES_FILE" | gemini --yolo

GEMINI_EXIT=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

if [[ $GEMINI_EXIT -eq 0 ]]; then
    print_step "Review generation completed in ${DURATION}s."
    
    # Verify outputs - check both naming conventions
    COUNT=$(find feedback -maxdepth 1 -name "*.md" 2>/dev/null | grep -v "summary.md" | wc -l)
    HAS_SUMMARY=$([[ -f "feedback/summary.md" ]] && echo "yes" || echo "no")
    
    if [[ $COUNT -ge 1 && "$HAS_SUMMARY" == "yes" ]]; then
        print_info "SUCCESS: Generated $COUNT criteria files and summary.md"
        
        # Step 3: Generate Feedback JSON
        print_step "Step 3: Generating all_feedback.json..."
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        JSON_GEN_SCRIPT="$SCRIPT_DIR/generate_feedback_json.py"
        
        if [[ -f "$JSON_GEN_SCRIPT" ]]; then
            python3 "$JSON_GEN_SCRIPT" "feedback"
            if [[ $? -eq 0 ]]; then
                print_info "Automation Complete! Feedback JSON is ready."
                notify_macos "✅ Review Complete" "Feedback for $STUDENT_ID is ready"
            else
                print_error "Failed to generate feedback JSON."
            fi
        else
            print_warning "generate_feedback_json.py not found at: $JSON_GEN_SCRIPT"
        fi
    else
        print_warning "Process finished but some files might be missing. Found $COUNT criteria files."
        notify_macos "⚠️ Review Warning" "Some files might be missing for $STUDENT_ID"
    fi
else
    print_error "Gemini agent failed with exit code $GEMINI_EXIT"
    notify_macos "❌ Review Failed" "Gemini agent failed for $STUDENT_ID"
    exit 1
fi
