#!/bin/bash
set -e
set -o pipefail

echo "=================================="
echo "🚀 Vast instance started"
echo "JOB_ID=$JOB_ID"
echo "=================================="

# --- Fetch run script ---
RUN_JOB_URL="https://raw.githubusercontent.com/mahibeulani-hash/ai-video-gpu-jobs/main/run_job.sh"

echo "📥 Downloading run_job.sh"
curl -fSL "$RUN_JOB_URL" -o /root/run_job.sh
chmod +x /root/run_job.sh

# --- Execute job ---
/root/run_job.sh

echo "🧹 onstart.sh completed"
