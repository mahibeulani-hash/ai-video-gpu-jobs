#!/bin/bash
set -euxo pipefail

############################################
# BASIC ENV
############################################
: "${JOB_ID:?JOB_ID not set}"
cd /root

echo "🚀 Starting job: $JOB_ID"

############################################
# FETCH JOB JSON
############################################
echo "📥 Fetching job.json"

JOB_JSON="/root/job.json"
JOB_URL="https://raw.githubusercontent.com/mahibeulani-hash/ai-video-gpu-jobs/main/jobs/${JOB_ID}.json"

curl -4 -fSL \
  --connect-timeout 10 \
  --max-time 30 \
  "$JOB_URL" \
  -o "$JOB_JSON"

cat "$JOB_JSON"

############################################
# WAIT FOR APT LOCK (NO UPDATE)
############################################
echo "⏳ Waiting for dpkg lock to be released..."

while fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do
  sleep 5
done

############################################
# SYSTEM DEPENDENCY (FFMPEG ONLY)
############################################
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "📦 Installing ffmpeg"
  apt-get install -y ffmpeg
else
  echo "✅ ffmpeg already installed"
fi

############################################
# DOWNLOAD SOURCE ZIP (SINGLE REQUEST)
############################################
echo "⬇️ Downloading source archive"

ZIP_NAME="ai-video-gpu-jobs-main.zip"
ZIP_URL="https://github.com/mahibeulani-hash/ai-video-gpu-jobs/archive/refs/heads/main.zip"

curl -4 -L \
  --connect-timeout 10 \
  --max-time 120 \
  "$ZIP_URL" \
  -o "$ZIP_NAME"

############################################
# EXTRACT SOURCE
############################################
echo "📦 Extracting source"

if ! command -v unzip >/dev/null 2>&1; then
  apt-get install -y unzip
fi

rm -rf /root/ai-video-gpu-jobs
unzip -o "$ZIP_NAME"
mv ai-video-gpu-jobs-main ai-video-gpu-jobs

cd /root/ai-video-gpu-jobs
ls -l

############################################
# PYTHON DEPENDENCIES
############################################
echo "🐍 Installing Python dependencies"

pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt

############################################
# RUN JOB
############################################
echo "🎬 Running GPU job"

python3 generate_gpu_job.py --job "$JOB_JSON"

############################################
# SHUTDOWN
############################################
echo "✅ Job finished successfully — shutting down"
#shutdown -h now  //for testing commented
