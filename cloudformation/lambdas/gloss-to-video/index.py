"""
Gloss-to-video Lambda.
Looks up BSL sign videos from DynamoDB multi-source lexicon,
stitches clips together using FFmpeg with crossfade transitions,
uploads result to S3 and returns pre-signed URL.
"""
import json
import os
import subprocess
import uuid
import logging

import boto3
from boto3.dynamodb.conditions import Key

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")

DATA_BUCKET = os.environ["DATA_BUCKET"]
LEXICON_TABLE = os.environ["LEXICON_TABLE"]
CROSSFADE_DURATION = 0.2  # seconds


def lambda_handler(event, context):
    """Translate BSL gloss words to stitched video."""
    gloss_words = event.get("glossWords", [])
    if not gloss_words:
        return {"videoUrl": "", "glossBreakdown": [], "error": "No gloss words provided"}

    table = dynamodb.Table(LEXICON_TABLE)
    clips = []
    breakdown = []

    for word in gloss_words:
        clip_info = resolve_gloss_word(table, word)
        clips.append(clip_info)
        breakdown.append({
            "word": word,
            "source": clip_info["source"],
            "signId": clip_info.get("sign_id", ""),
            "type": clip_info["type"],
        })

    # Download all clips to /tmp
    local_clips = []
    for i, clip in enumerate(clips):
        if clip["type"] == "fingerspell":
            # Fingerspelling: multiple letter clips
            for j, letter_key in enumerate(clip["video_keys"]):
                local_path = f"/tmp/clip_{i}_{j}.mp4"
                try:
                    s3.download_file(DATA_BUCKET, letter_key, local_path)
                    local_clips.append(local_path)
                except Exception as e:
                    logger.warning(f"Missing fingerspelling clip {letter_key}: {e}")
        else:
            local_path = f"/tmp/clip_{i}.mp4"
            try:
                s3.download_file(DATA_BUCKET, clip["video_key"], local_path)
                local_clips.append(local_path)
            except Exception as e:
                logger.warning(f"Missing sign clip {clip['video_key']}: {e}")

    if not local_clips:
        return {"videoUrl": "", "glossBreakdown": breakdown, "error": "No video clips found"}

    # Stitch clips with FFmpeg crossfade
    output_path = f"/tmp/output_{uuid.uuid4().hex[:8]}.mp4"

    if len(local_clips) == 1:
        # Single clip — just copy
        output_path = local_clips[0]
    else:
        stitch_with_crossfade(local_clips, output_path)

    # Upload to S3
    output_key = f"output/{uuid.uuid4().hex}.mp4"
    s3.upload_file(output_path, DATA_BUCKET, output_key, ExtraArgs={"ContentType": "video/mp4"})

    # Generate pre-signed URL (1 hour expiry)
    video_url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": DATA_BUCKET, "Key": output_key},
        ExpiresIn=3600,
    )

    # Clean up /tmp
    for clip_path in local_clips:
        try:
            os.remove(clip_path)
        except OSError:
            pass
    try:
        os.remove(output_path)
    except OSError:
        pass

    return {"videoUrl": video_url, "glossBreakdown": breakdown}


def resolve_gloss_word(table, word):
    """Look up a gloss word in DynamoDB. Falls back to fingerspelling."""
    word_upper = word.upper().strip()

    # Skip pronouns and index signs (no video needed, rendered as text)
    if word_upper.startswith("IX-"):
        return {
            "source": "index",
            "type": "index",
            "sign_id": word_upper,
            "video_key": "",
        }

    # Query DynamoDB for this gloss (sorted by quality rank)
    response = table.query(
        KeyConditionExpression=Key("Gloss").eq(word_upper),
        ScanIndexForward=True,  # Ascending = best quality first
    )

    items = response.get("Items", [])
    if items:
        best = items[0]
        return {
            "source": best["Source"],
            "type": "sign",
            "sign_id": best["SignId"],
            "video_key": best["VideoKey"],
        }

    # Fallback: fingerspell the word
    logger.info(f"No sign found for '{word_upper}', falling back to fingerspelling")
    letters = [c for c in word_upper if c.isalpha()]
    return {
        "source": "fingerspelling",
        "type": "fingerspell",
        "sign_id": f"fs-{word_upper.lower()}",
        "video_keys": [f"lexicon/fingerspelling/{letter}.mp4" for letter in letters],
    }


def stitch_with_crossfade(clip_paths, output_path):
    """Stitch video clips with FFmpeg crossfade transitions.

    For 2 clips: simple xfade filter.
    For 3+ clips: chain xfade filters progressively.
    """
    if len(clip_paths) < 2:
        return

    # Get durations of all clips
    durations = []
    for path in clip_paths:
        dur = get_video_duration(path)
        durations.append(dur if dur > 0 else 1.0)

    if len(clip_paths) == 2:
        # Simple case: two clips
        offset = max(0, durations[0] - CROSSFADE_DURATION)
        cmd = [
            "ffmpeg", "-y",
            "-i", clip_paths[0],
            "-i", clip_paths[1],
            "-filter_complex",
            f"[0][1]xfade=transition=fade:duration={CROSSFADE_DURATION}:offset={offset}",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-an",  # No audio
            output_path,
        ]
    else:
        # Multi-clip: chain xfade filters
        inputs = []
        for path in clip_paths:
            inputs.extend(["-i", path])

        filter_parts = []
        cumulative_offset = 0
        prev_label = "[0]"

        for i in range(1, len(clip_paths)):
            offset = cumulative_offset + durations[i - 1] - CROSSFADE_DURATION
            if i == 1:
                in_label = "[0][1]"
            else:
                in_label = f"[v{i-1}][{i}]"

            out_label = f"[v{i}]" if i < len(clip_paths) - 1 else ""
            filter_parts.append(
                f"{in_label}xfade=transition=fade:duration={CROSSFADE_DURATION}:offset={max(0, offset)}{out_label}"
            )
            cumulative_offset = max(0, offset)

        filter_complex = ";".join(filter_parts)
        cmd = [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", filter_complex,
            "-c:v", "libx264", "-preset", "ultrafast",
            "-an",
            output_path,
        ]

    logger.info(f"FFmpeg command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=90,
        )
        if result.returncode != 0:
            logger.error(f"FFmpeg stderr: {result.stderr}")
            # Fallback: simple concat without crossfade
            concat_fallback(clip_paths, output_path)
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg timed out, using concat fallback")
        concat_fallback(clip_paths, output_path)


def concat_fallback(clip_paths, output_path):
    """Simple concat demuxer fallback (no crossfade)."""
    list_path = "/tmp/concat_list.txt"
    with open(list_path, "w") as f:
        for path in clip_paths:
            f.write(f"file '{path}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", list_path,
        "-c:v", "libx264", "-preset", "ultrafast",
        "-an",
        output_path,
    ]

    subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    os.remove(list_path)


def get_video_duration(path):
    """Get video duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True, text=True, timeout=10,
        )
        return float(result.stdout.strip())
    except (ValueError, subprocess.TimeoutExpired):
        return 1.0
