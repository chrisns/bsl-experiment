"""
Process Amazon Transcribe result.
Downloads the transcription output JSON from S3 and extracts the transcript text.
"""
import json
import os
import logging

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client("s3")
DATA_BUCKET = os.environ["DATA_BUCKET"]


def lambda_handler(event, context):
    """Extract transcript text from Transcribe result."""
    # The Step Functions state machine passes the transcription result
    transcription_result = event.get("transcriptionResult", event)

    # Get the transcript file URI from the Transcribe result
    transcript_uri = (
        transcription_result
        .get("TranscriptionJob", {})
        .get("Transcript", {})
        .get("TranscriptFileUri", "")
    )

    if not transcript_uri:
        # Try alternative path from Step Functions SDK integration
        transcript_uri = event.get("Transcript", {}).get("TranscriptFileUri", "")

    if not transcript_uri:
        logger.warning("No transcript URI found in event")
        return {"text": "", "error": "No transcript URI found"}

    # Parse S3 URI or HTTPS URL to get bucket/key
    if transcript_uri.startswith("s3://"):
        parts = transcript_uri[5:].split("/", 1)
        bucket = parts[0]
        key = parts[1]
    elif "s3.amazonaws.com" in transcript_uri:
        # HTTPS format: https://s3.region.amazonaws.com/bucket/key
        from urllib.parse import urlparse
        parsed = urlparse(transcript_uri)
        path_parts = parsed.path.lstrip("/").split("/", 1)
        bucket = path_parts[0]
        key = path_parts[1]
    else:
        # The output bucket is our data bucket
        key = transcript_uri.split("/")[-1]
        bucket = DATA_BUCKET

    # Download and parse the transcript
    response = s3.get_object(Bucket=bucket, Key=key)
    transcript_data = json.loads(response["Body"].read().decode())

    # Extract full transcript text
    results = transcript_data.get("results", {})
    transcripts = results.get("transcripts", [])

    if transcripts:
        text = transcripts[0].get("transcript", "")
    else:
        text = ""

    logger.info(f"Transcribed text: '{text[:100]}...'")

    return {"text": text}
