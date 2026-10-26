FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY regression/ regression/
COPY prompts/ prompts/
COPY data/ data/

# Configuration is injected at runtime:
#   GEMINI_API_KEY, SLACK_WEBHOOK_URL, REPORT_URL
#   WARN_THRESHOLD, CRITICAL_THRESHOLD, DRIFT_WINDOW, DRIFT_FLOOR
# Mount a volume at /app/results to keep run history between container runs.
VOLUME /app/results
ENTRYPOINT ["python", "-m", "regression.cli"]
CMD ["--prompt", "prompts/v1.yaml"]
