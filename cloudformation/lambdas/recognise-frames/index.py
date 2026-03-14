"""
BSL sign recognition Lambda (Function URL endpoint).
Receives 64-frame batch, invokes SageMaker endpoint, optionally validates
low-confidence predictions with Bedrock Nova Lite.
"""
import json
import os
import base64
import logging

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

sagemaker_runtime = boto3.client("sagemaker-runtime")
bedrock_runtime = boto3.client("bedrock-runtime")

SAGEMAKER_ENDPOINT = os.environ["SAGEMAKER_ENDPOINT"]
CONFIDENCE_THRESHOLD = 0.70  # Below this, invoke Nova Lite for validation


def lambda_handler(event, context):
    """Handle HTTP POST with 64-frame batch from Lambda Function URL."""
    try:
        body = json.loads(event.get("body", "{}"))
        frames = body.get("frames", [])
        batch_id = body.get("batchId", "unknown")

        if not frames:
            return response(400, {"error": "No frames provided"})

        # Invoke SageMaker endpoint
        sagemaker_response = sagemaker_runtime.invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT,
            ContentType="application/json",
            Body=json.dumps({"frames": frames, "batchId": batch_id}),
        )

        result = json.loads(sagemaker_response["Body"].read().decode())
        predictions = result.get("predictions", [])

        # Check if top prediction needs Nova Lite validation
        nova_description = None
        if predictions and predictions[0]["confidence"] < CONFIDENCE_THRESHOLD:
            nova_description = validate_with_nova(frames, predictions[0])

        return response(200, {
            "type": "prediction",
            "batchId": batch_id,
            "signs": predictions,
            "novaDescription": nova_description,
        })

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return response(500, {"error": str(e)})


def validate_with_nova(frames, top_prediction):
    """Use Bedrock Nova Lite to describe the signing gesture as a secondary signal."""
    try:
        # Use the middle frame for Nova Lite analysis
        middle_idx = len(frames) // 2
        frame_b64 = frames[middle_idx]

        nova_body = json.dumps({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "image": {
                                "format": "jpeg",
                                "source": {"bytes": frame_b64},
                            }
                        },
                        {
                            "text": (
                                "Describe what this person is doing with their hands in detail. "
                                "Focus on hand shape, position, and movement. "
                                "This may be a sign language gesture."
                            ),
                        },
                    ],
                }
            ],
            "inferenceConfig": {"maxTokens": 200},
        })

        nova_response = bedrock_runtime.invoke_model(
            modelId="amazon.nova-lite-v1:0",
            contentType="application/json",
            accept="application/json",
            body=nova_body,
        )

        nova_result = json.loads(nova_response["body"].read().decode())
        description = nova_result.get("output", {}).get("message", {}).get("content", [{}])[0].get("text", "")

        logger.info(
            f"Nova Lite validation for '{top_prediction['sign']}' "
            f"(conf={top_prediction['confidence']:.2f}): {description[:100]}"
        )
        return description

    except Exception as e:
        logger.warning(f"Nova Lite validation failed: {e}")
        return None


def response(status_code, body):
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(body),
    }
