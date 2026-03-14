# BSL Sign Language Translation

## Overview

Bidirectional British Sign Language (BSL) translation system — the first real-time BSL translation system for UK local government, deployed as an ISB scenario.

**Sign → Text**: Camera captures BSL signing → TensorFlow.js MediaPipe pose overlay in browser → 64-frame batches sent to BSL-1K I3D model on SageMaker (ml.g5.xlarge GPU, ~100ms inference) → text output with optional Polly speech.

**Text → Sign**: Text/speech input → Bedrock Claude translates English to BSL gloss → DynamoDB multi-source lexicon lookup → FFmpeg Lambda stitches sign video clips → video output.

## AWS Services Used

- Amazon SageMaker (real-time GPU inference endpoint)
- Amazon Bedrock (Claude for glossing, Nova Lite for validation)
- Amazon Polly (text-to-speech)
- Amazon Transcribe (speech-to-text)
- AWS Lambda + Function URLs (API endpoints)
- AWS Step Functions (text-to-sign orchestration)
- AWS CodeBuild (self-deploying SageMaker container)
- Amazon S3 + CloudFront (website hosting, data storage)
- Amazon DynamoDB (BSL sign lexicon)
- Amazon Cognito (authentication)

## Deployment

```bash
aws cloudformation deploy \
  --template-file template.yaml \
  --stack-name ndx-try-bsl-sign-language \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM \
  --profile NDX/SandboxAdmin \
  --region us-east-1
```

Stack creation takes ~15-20 minutes (seed data download + CodeBuild container build + SageMaker endpoint creation).

## Cost

- SageMaker ml.g5.xlarge: ~$1.41/hr (always-warm endpoint)
- Bedrock Nova Lite validation: ~$0.03-0.09/min of active signing
- Lambda/S3/DynamoDB: negligible for demo use

## Prerequisites

- ISB sandbox account with GPU quotas (ml.g5.xlarge=2, pre-provisioned)
- Bedrock model access for Claude Haiku and Nova Lite in us-east-1
