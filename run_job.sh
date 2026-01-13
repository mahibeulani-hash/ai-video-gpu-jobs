#!/bin/bash

curl -X PUT \
  -H "Accept: application/vnd.github+json" \
  -d "$(jq -n \
    --arg msg "Job $JOB_ID started" \
    --arg content "$(echo started | base64 -w0)" \
    '{message:$msg, content:$content}')" \
  "https://api.github.com/repos/mahibeulani-hash/ai-video-gpu-jobs/contents/heartbeats/${JOB_ID}.txt"


set -e
set -o pipefail

JOB_URL="https://raw.githubusercontent.com/mahibeulani-hash/ai-video-gpu-jobs/main/jobs/${JOB_ID}.json"

echo "📥 Fetching job.json"
curl -fSL "$JOB_URL" -o job.json
cat job.json

echo "🐍 Running GPU job"
mkdir -p /root/ai-video-gpu-jobs
cd /root/ai-video-gpu-jobs

curl -fSL \
  https://raw.githubusercontent.com/mahibeulani-hash/ai-video-gpu-jobs/main/generate_gpu_job.py \
  -o generate_gpu_job.py

python3 generate_gpu_job.py --job job.json

echo "✅ Job finished — shutting down"
shutdown -h now
