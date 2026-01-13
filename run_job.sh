#!/bin/bash
set -e
set -o pipefail

echo "=============================="
echo "🚀 Vast GPU job started"
echo "JOB_ID=$JOB_ID"
echo "=============================="

# Fetch job
JOB_URL="https://raw.githubusercontent.com/mahibeulani-hash/ai-video-gpu-jobs/main/jobs/${JOB_ID}.json"
echo "📥 Fetching job from $JOB_URL"
curl -fSL "$JOB_URL" -o job.json

echo "✅ job.json fetched"
cat job.json

# Run GPU job
python3 generate_gpu_job.py --job job.json

echo "🧹 Job finished, shutting down"
shutdown -h now
