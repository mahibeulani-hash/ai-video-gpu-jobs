#!/bin/bash
set -e
set -o pipefail

JOB_URL="https://raw.githubusercontent.com/mahibeulani-hash/ai-video-gpu-jobs/main/jobs/${JOB_ID}.json"

echo "📥 Fetching job.json"
curl -fSL "$JOB_URL" -o job.json
cat job.json

echo "🐍 Running GPU job"
python3 generate_gpu_job.py --job job.json

echo "✅ Job finished — shutting down"
shutdown -h now
