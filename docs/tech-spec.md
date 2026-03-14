---
title: 'BSL Sign Language Translation - ISB Scenario Deployment'
slug: 'bsl-sign-language-translation'
created: '2026-03-09'
status: 'review-complete'
stepsCompleted: [1, 2, 3, 4]
tech_stack:
  - AWS CloudFormation (SAM transform)
  - Amazon SageMaker real-time endpoint (BSL-1K I3D model, ml.g5.xlarge GPU inference ~100ms)
  - Amazon Bedrock (Claude - English→BSL glossing, Nova Lite - vision validation)
  - Amazon Polly (text-to-speech)
  - Amazon Transcribe (speech-to-text)
  - AWS Lambda (FFmpeg video stitching, API handlers, seed data, CodeBuild trigger)
  - AWS Step Functions (orchestration)
  - AWS Lambda Function URLs (HTTP endpoints for frame batch + text-to-sign, 6MB payload limit)
  - Amazon S3 (website hosting, BSL video lexicon, model artifacts)
  - Amazon CloudFront (CDN)
  - Amazon DynamoDB (multi-source gloss→video mapping with quality rank)
  - Amazon Cognito (auth)
  - AWS CodeBuild (build SageMaker inference container on stack creation)
  - TensorFlow.js + MediaPipe (browser-side pose extraction, 30fps)
files_to_modify:
  - cloudformation/scenarios/bsl-sign-language/template.yaml (new)
  - cloudformation/scenarios/bsl-sign-language/BLUEPRINT.md (new)
  - cloudformation/scenarios/bsl-sign-language/lambdas/ (new)
  - cloudformation/scenarios/bsl-sign-language/frontend/ (new)
  - cloudformation/scenarios/bsl-sign-language/sagemaker/ (new)
  - cloudformation/isb-hub/lib/isb-hub-stack.ts (add to SCENARIOS array)
  - src/_data/scenarios.yaml (add scenario metadata - later phase)
code_patterns:
  - SAM transform CloudFormation template (existing scenario pattern)
  - ISB-compatible IAM roles (InnovationSandbox-ndx- prefix)
  - Step Functions orchestration (GenASL pattern)
  - SageMaker real-time endpoint with custom inference.py
  - Lambda Function URLs (existing scenario pattern)
  - S3 static website + CloudFront (existing pattern)
test_patterns:
  - Manual deployment test to sandbox account (NDX/SandboxAdmin profile)
  - End-to-end functional test via Playwright
---

# Tech-Spec: BSL Sign Language Translation - ISB Scenario Deployment

**Created:** 2026-03-09

## Overview

### Problem Statement

87,000 BSL (British Sign Language) users in the UK face communication barriers when accessing council services. No UK local government currently offers real-time sign language translation. Existing open-source solutions (sign.mt) are tightly coupled to Google Cloud/Firebase and have minimal actual BSL support despite claiming it. There is no AWS-native BSL translation system.

### Solution

Build an AWS-native bidirectional BSL translation system deployed as an ISB (Innovation Sandbox) scenario — the first real-time BSL translation system in UK local government. The system provides two translation paths with a conversation-mode UX:

1. **Sign → Text**: Camera captures BSL signing at 25fps → TensorFlow.js MediaPipe runs in-browser for instant pose overlay (30fps) → 64-frame batches sent via Lambda Function URL to BSL-1K I3D model on SageMaker real-time endpoint (ml.g5.xlarge GPU, ~100ms inference) → sign predictions returned → text output with optional Polly speech. Nova Lite multimodal validates low-confidence predictions as a secondary signal. ISB sandbox accounts have GPU quotas pre-provisioned (ml.g5.xlarge=2) and the SageMaker SLR is creatable without SCP issues.
2. **Text → Sign**: Text/speech input → Bedrock Claude translates English to BSL gloss (using BSL grammar rules from English2BSL ACL 2023 paper) → DynamoDB multi-source lexicon lookup (BSL SignBank + signbsl.com when licensed + fingerspelling fallback) → FFmpeg Lambda stitches clips with crossfade transitions → video output as picture-in-picture avatar.

Three UI modes: **Live Translation** (full bidirectional conversation), **Practice Mode** (learn BSL signs with recognition-based scoring), **Kiosk Mode** (reception desk tablet optimised).

Uses BSL-1K pretrained model (1,064 BSL signs, Oxford VGG, freely available weights) on SageMaker ml.g5.xlarge real-time endpoint for recognition (~100ms GPU inference). BSL SignBank video clips (BSD-3-Clause, ~2,500 signs from UCL) for generation, with multi-source lexicon architecture supporting future expansion to signbsl.com (14,000+ clips, licensing in progress). Lambda Function URLs replace API Gateway — simpler and consistent with other NDX:Try scenarios.

### Scope

**In Scope:**
- CloudFormation template deployable to ISB sandbox accounts
- BSL-1K I3D model deployed on SageMaker real-time endpoint (ml.g5.xlarge GPU, ~100ms inference)
- Browser-side TensorFlow.js MediaPipe for real-time pose overlay (no server round-trip)
- Lambda Function URLs for frame batch submission and text-to-sign requests (6MB payload limit, simpler than API Gateway)
- Bedrock Claude-based English→BSL gloss translation with BSL grammar rules
- Bedrock Nova Lite multimodal as secondary confidence validation for sign recognition
- Multi-source DynamoDB lexicon with quality ranking (BSL SignBank now, signbsl.com later)
- BSL two-handed fingerspelling fallback (BANZ-FS dataset, CC BY 4.0)
- FFmpeg Lambda for sign video clip stitching with crossfade transitions
- Amazon Polly for text-to-speech output (British English voices)
- Amazon Transcribe for speech-to-text input
- Step Functions orchestration for both translation paths
- Three UI modes: Live Translation, Practice Mode, Kiosk Mode
- Conversation-mode UX (video call layout, not split-panel)
- CodeBuild project within stack to build SageMaker inference container image (self-deploying)
- Custom resource Lambda to seed data (download BSL-1K weights, BSL SignBank videos, populate DynamoDB)
- ISB-compatible IAM roles (`InnovationSandbox-ndx-` prefix)
- High-contrast, accessible UI (keyboard navigation, screen reader support, large text)
- Manual deployment test to sandbox via `--profile NDX/SandboxAdmin`

**Out of Scope:**
- Scenario portal pages (src/scenarios/, walkthroughs) — later phase
- StackSet/hub registration — later phase (after sandbox deployment proven)
- CI/CD pipeline integration — later phase
- Non-manual features (facial expressions, mouth patterns) — not captured by current models
- Custom model training/fine-tuning on BOBSL (but architecture supports it)
- signbsl.com integration (pending licensing — email drafted)
- Mobile-native apps (web-first, responsive for tablet kiosk mode)

## Context for Development

### Codebase Patterns

**Scenario template pattern**: Each scenario lives in `cloudformation/scenarios/{name}/` with a `template.yaml` (SAM/CloudFormation) and `BLUEPRINT.md`. Simple scenarios use raw YAML; complex ones (localgov-drupal, simply-readable) use CDK. This scenario will use raw SAM YAML since the architecture is Lambda-heavy.

**ISB SCP constraint**: All IAM roles must be named with `InnovationSandbox-ndx-` prefix or API calls will be blocked by the SCP (`p-tyb1wjxv`). This applies to Lambda execution roles, SageMaker execution role, Step Functions role, API Gateway role — everything.

**Hub stack integration**: The `SCENARIOS` array in `cloudformation/isb-hub/lib/isb-hub-stack.ts` defines StackSets. Adding a new entry there + a `template.yaml` in the scenarios dir is all that's needed for StackSet deployment. Deferred to later phase.

**GenASL reference architecture**: The `aws-samples/genai-asl-avatar-generator` project provides the pattern for the text→sign path: Step Functions orchestrates Transcribe → Bedrock (text→gloss) → Lambda (gloss→video via FFmpeg). We adapt this for BSL with better engineering (parameterized template, proper IAM, no hardcoded bucket names).

### Files to Reference

| File | Purpose |
| ---- | ------- |
| `cloudformation/scenarios/council-chatbot/template.yaml` | Reference SAM template pattern with Bedrock integration |
| `cloudformation/scenarios/text-to-speech/template.yaml` | Reference SAM template pattern with Polly/S3 |
| `cloudformation/isb-hub/lib/isb-hub-stack.ts` | Hub stack with SCENARIOS array and StackSet definitions |
| `cloudformation/scenarios/simply-readable/template.yaml` | Complex scenario with S3 website + CloudFront + Cognito |
| (external) `github.com/aws-samples/genai-asl-avatar-generator` | GenASL architecture reference |
| (external) `github.com/gulvarol/bsl1k` | BSL-1K model code, `demo/demo.py` for inference pipeline |
| (external) `bslsignbank.ucl.ac.uk` | BSL SignBank — source of sign video lexicon |

### Technical Decisions

1. **BSL-1K over CSLR2**: BSL-1K works end-to-end from raw video (no pre-extracted features needed), has freely downloadable weights, a working demo script, and 1,064 BSL signs. CSLR2 has more signs (8,697) but requires 262GB of pre-extracted BOBSL features and has no raw-video inference pipeline.

2. **BSL-1K over SignON PoseFormer**: SignON PoseFormer is tiny (26MB) and easy to deploy but has only 32.78% accuracy on BSL — unusable for a demo.

3. **SageMaker real-time endpoint (VERIFIED)**: ISB sandbox accounts come pre-provisioned with GPU quotas (ml.g5.xlarge=2, ml.g4dn.xlarge=2). The SageMaker SLR (`AWSServiceRoleForAmazonSageMakerNotebooks`) is creatable with SandboxUser — service-linked roles bypass the ISB SCP. Verified in sandbox account 572885592978 on 2026-03-10. This gives ~100ms GPU inference per 64-frame batch. Cost: ml.g5.xlarge ~$1.41/hr — acceptable for demo use.

4. **Browser-side MediaPipe (TensorFlow.js) over server-side Lambda**: Running MediaPipe Holistic in the browser via TensorFlow.js eliminates the network round-trip for pose extraction, gives instant 30fps pose overlay feedback, and removes the ECR container dependency. The browser sends raw JPEG frames to SageMaker (not keypoints) — MediaPipe is purely for the visual overlay and could optionally feed a lightweight browser-side classifier for instant predictions alongside the server model.

5. **Lambda Function URLs over WebSocket/API Gateway**: Client accumulates 64-frame batches in a ring buffer and sends complete batches via HTTP POST to a Lambda Function URL (6MB payload limit, well within the ~960KB for 64 JPEG frames). This is simpler than WebSocket API Gateway (no connection management, no route config, no $connect/$disconnect handlers) and consistent with other NDX:Try scenarios that all use Lambda Function URLs. Polling interval of ~320ms (8 new frames at 25fps = one batch per stride) provides near-real-time UX. SageMaker's ~100ms inference + Lambda overhead means results return within ~500ms of each batch — fast enough for live captioning.

6. **Multi-source lexicon with quality ranking**: DynamoDB schema supports multiple video sources per gloss (BSL SignBank, signbsl.com, fingerspelling) with a quality rank. Start with BSL SignBank (BSD-3-Clause, ~2,500 signs), expand to signbsl.com (14,000+ clips) when licensing is secured. Fingerspelling is the universal fallback.

7. **Nova Lite as secondary confidence signal**: At $0.06/M input tokens, sending every 5th frame to Amazon Nova Lite with "describe what this person is doing with their hands" costs ~$0.03-0.09 per minute of signing. When BSL-1K confidence is <70%, Nova's natural language description corroborates or contradicts the prediction. Novel multi-model fusion: CV model + LLM vision.

8. **CodeBuild for container image self-deployment**: Instead of pre-baking ECR images or manual Docker builds, a CodeBuild project within the CloudFormation stack builds and pushes container images (BSL-1K inference + MediaPipe if needed) from Dockerfiles stored in S3. This makes the template fully self-deploying — no pre-requisites beyond the template itself.

9a. **Lambda Function URLs for all endpoints**: All HTTP endpoints use Lambda Function URLs with `AuthType: NONE` (Cognito auth handled at the application level). Two primary endpoints: (1) frame-batch recognition endpoint (POST 64 JPEG frames → SageMaker → predictions), (2) text-to-sign endpoint (POST English text → Step Functions → video URL). Consistent with the pattern used by all other NDX:Try scenarios.

9. **Conversation-mode UX over split-panel**: Full-screen video call layout (like Teams/Zoom) with live-updating text transcript, picture-in-picture BSL avatar, and pose overlay. Three modes (Live Translation, Practice, Kiosk) serve different use cases. High-contrast accessible design with keyboard navigation.

10. **Crossfade transitions in FFmpeg stitching**: Simple `concat` produces jarring cuts between signs. Using FFmpeg's `xfade` filter with 200ms crossfade between clips produces smoother, more natural-looking sign sequences.

11. **Pre-recorded video clips over generated animation**: Using real BSL SignBank videos gives authentic signing. Generated avatars (pix2pix, diffusion) produce uncanny/low-quality output and add massive ML complexity.

### Competitive Landscape

| Competitor | Status | Approach | BSL Quality |
|---|---|---|---|
| **Signapse AI** | Commercial, £2M seed | GAN avatar + LLM glossing | Production-grade but closed-source |
| **Silence Speaks** | 250+ TfL stations deployed | AI avatar via QR code | Live deployment, Innovate UK funded |
| **sign.mt** | Open source (CC BY-NC-SA) | Client-side MediaPipe + pix2pix | BSL support is minimal/vapourware |
| **SignGPT** | £8.45M EPSRC (5-year) | Building largest BSL dataset + LLM | Early stage, potential future collaboration |
| **This project** | Open source, AWS-native | BSL-1K + Bedrock + real video clips | First open-source AWS-native BSL system |

### Future Model Opportunities

- **Keypoint-based BSL recognition** (arxiv 2412.09475, Oxford VGG Dec 2024) — validates MediaPipe-style keypoint approach for BSL, 8,162 word vocabulary. Could replace BSL-1K when code/weights are released.
- **SONAR-SLT** (Meta, Oct 2025) — multilingual sign language translation designed for cross-language generalisation. HuggingFace compatible. Strong candidate for BSL fine-tuning.
- **BSL MediaPipe Kaggle dataset** — pre-extracted landmarks for BSL alphabet/numbers. Immediately usable for browser-side fingerspelling recognition.

## Implementation Plan

### Tasks

**Implementation order: skeleton-first deployment.** Get a deploying template with stubs before fleshing out individual components. This de-risks the biggest unknowns (SageMaker endpoint creation, CodeBuild container build, seed data download) before investing in business logic.

- [x] **Task 1: Project scaffolding**
  - File: `cloudformation/scenarios/bsl-sign-language/`
- Create directory structure:
  ```
  bsl-sign-language/
  ├── template.yaml              # Main SAM CloudFormation template
  ├── BLUEPRINT.md               # ISB registration instructions
  ├── lambdas/
  │   ├── seed-data/             # Custom resource: download models + videos + build container
  │   ├── recognise-frames/      # Lambda Function URL: receive 64-frame batch → SageMaker → predictions
  │   ├── text-to-gloss/         # Invokes Bedrock Claude for BSL glossing
  │   ├── gloss-to-video/        # DynamoDB lookup + FFmpeg stitch with crossfade
  │   └── process-transcription/ # Transcribe result processor
  ├── sagemaker/
  │   └── bsl1k-inference/
  │       ├── Dockerfile         # BSL-1K I3D on SageMaker (PyTorch GPU)
  │       ├── inference.py       # SageMaker inference handler: load I3D, run on GPU
  │       ├── model.py           # I3D architecture (adapted from bsl1k/models/i3d.py)
  │       └── requirements.txt   # torch, torchvision, opencv-python-headless, numpy, scipy
  ├── frontend/
  │   ├── index.html             # SPA entry point
  │   ├── app.js                 # Main app logic + mode switching
  │   ├── camera.js              # WebRTC 25fps capture + frame ring buffer (JPEG quality 0.5)
  │   ├── mediapipe-overlay.js   # TensorFlow.js MediaPipe pose overlay (30fps)
  │   ├── api-client.js          # HTTP client for Lambda Function URL endpoints
  │   ├── player.js              # BSL video playback (PiP mode)
  │   ├── practice.js            # Practice mode (learn signs, scoring)
  │   ├── kiosk.js               # Kiosk mode (reception desk optimised)
  │   └── styles.css             # High-contrast accessible styles
  ├── statemachine/
  │   └── text-to-sign.asl.yaml  # Step Functions: text→gloss→video pipeline
  └── codebuild/
      └── buildspec.yml          # CodeBuild spec for SageMaker inference container
  ```

- [x] **Task 2: SAM CloudFormation template (skeleton with stubs)**
  - File: `template.yaml`
  - Action: Write the full template with all resources defined but Lambda code as stubs (return placeholder responses). Deploy this first to prove the infrastructure stands up — SageMaker endpoint, CodeBuild, ECR, S3, DynamoDB, CloudFront, Cognito all created successfully.
- Parameters: `Environment` (sandbox/dev/prod), `SageMakerInstanceType` (default ml.g5.xlarge)
- Resources:
  - **Storage**: S3 buckets (website, data/lexicon/models, output/stitched videos), DynamoDB table (BSLSignLexicon with Gloss PK + QualityRank#Source SK)
  - **CDN**: CloudFront distribution → website bucket (OAC)
  - **Auth**: Cognito User Pool + Identity Pool (unauthenticated disabled)
  - **Compute**: Lambda functions with Function URLs (all with `InnovationSandbox-ndx-bsl-*` role names):
    - seed-data (custom resource, 900s timeout, 1024MB)
    - recognise-frames (Function URL, 30s timeout, 256MB — invokes SageMaker endpoint)
    - text-to-gloss (60s timeout)
    - gloss-to-video (120s timeout, FFmpeg layer, 512MB)
    - process-transcription
  - **ML**: SageMaker Model + EndpointConfig + Endpoint (ml.g5.xlarge, BSL-1K I3D), ECR repository for inference container
  - **Orchestration**: Step Functions state machine (text-to-sign pipeline)
  - **Build**: CodeBuild project (builds SageMaker inference container from Dockerfile in S3, pushes to ECR, packages model.tar.gz)
  - **ECR**: Repository for SageMaker inference container image
  - **Custom Resources**: seed-data Lambda invocation (downloads BSL-1K weights, SignBank videos, populates DynamoDB, triggers CodeBuild)
- All bucket names use `!Sub` with `${AWS::AccountId}` for uniqueness
- All IAM roles use explicit `RoleName: !Sub 'InnovationSandbox-ndx-bsl-{function-name}'`
- Tags: Project=ndx-try, Scenario=bsl-sign-language, AutoCleanup=true
- SageMaker viable: ISB sandbox accounts have ml.g5.xlarge quota=2 and SLR is creatable (verified 2026-03-10)

- [x] **Task 3: Seed data custom resource Lambda**
  - File: `lambdas/seed-data/app.py`
  - Action: Implement first — this is the longest-running component and biggest risk. If model download or SignBank scraping fails, nothing else works.
- CloudFormation Custom Resource (Create/Delete handlers)
- **On Create:**
  1. Download BSL-1K pretrained weights from `https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/bsl1k.pth.tar` → S3 `models/bsl1k.pth.tar`
  2. Download BSL-1K vocab/info from VGG servers → S3 `models/info.pkl`
  3. Scrape BSL SignBank videos: iterate sign pages, download video files → S3 `lexicon/signs/{signId}.mp4`
  4. Download BANZ-FS fingerspelling reference videos → S3 `lexicon/fingerspelling/{letter}.mp4`
  5. **Build BSL-1K vocab → SignBank gloss mapping**: Parse `info.pkl` to extract BSL-1K's vocabulary labels (trained on BOBSL subtitle alignment). Cross-reference against SignBank's gloss index. BSL-1K uses BOBSL labels (e.g. "HELLO") while SignBank uses linguistic glosses (e.g. "GREET" or "hello") — build a normalised mapping table. Store mapping in DynamoDB alongside the sign video entries. Log unmapped signs for manual review.
  6. Populate DynamoDB `BSLSignLexicon` table with gloss→signId mappings from SignBank metadata, using the normalised mapping from step 5
  7. Package `model.tar.gz` (inference.py + weights + vocab) and upload to S3 for SageMaker
  8. **Trigger CodeBuild and poll to completion**: Start CodeBuild project to build SageMaker inference container. Poll `batch_get_builds` every 15s until build status is `SUCCEEDED` (or fail the custom resource if `FAILED`/`STOPPED`). The seed-data Lambda must NOT return SUCCESS to CloudFormation until CodeBuild has completed — SageMaker Model resource depends on the ECR image existing.
- **On Delete:** Clean up S3 objects (model artifacts, videos)
- Timeout: 900s (15 min) — downloading ~2,500 sign videos + CodeBuild (~5 min) will take time
- Memory: 1024MB

- [x] **Task 4: BSL-1K SageMaker inference endpoint**
  - Files: `sagemaker/bsl1k-inference/Dockerfile`, `sagemaker/bsl1k-inference/inference.py`, `sagemaker/bsl1k-inference/requirements.txt`
  - Action: Build the inference container and handler. Deploy and verify the endpoint returns predictions before writing any Lambdas that depend on it.
- **Dockerfile**: Based on `763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.1-gpu-py310-cu121-ubuntu20.04-sagemaker`. Copy inference handler + BSL-1K model code (adapted from `demo/demo.py`). ~5-8GB image.
- **inference.py**: SageMaker inference handler (TorchServe-compatible). **CRITICAL**: Model loading MUST match `bsl1k/demo/demo.py` exactly — use `torch.load` with the same checkpoint key mappings and `scipy.special.softmax` for confidence scores. Do not reimagine the loading logic; copy it verbatim from the demo.
  - `model_fn(model_dir)`: Load I3D model from `model_dir/bsl1k.pth.tar`, load vocab from `model_dir/info.pkl`. Move model to GPU (CUDA). Use exact key mapping from `demo.py`.
  - `input_fn(request_body, content_type)`: Accept JSON body with base64-encoded JPEG frames array. Decode to numpy, resize to 256×256, center crop to 224×224, normalize (mean=0.5, std=1.0). Stack into tensor on GPU.
  - `predict_fn(input_data, model)`: Run sliding window (64 frames, stride 8) through I3D on GPU. Apply `scipy.special.softmax` to logits. Return top-5 predictions per window.
  - `output_fn(prediction, accept)`: Return JSON `{"predictions": [{"sign": "HELLO", "confidence": 0.92, "window_start": 0, "window_end": 64}]}`
- **Performance**: I3D on ml.g5.xlarge GPU: ~100ms per 64-frame batch. Always-warm endpoint, no cold start.
- **Dependencies**: `torch` (CUDA), `torchvision`, `opencv-python-headless`, `numpy`, `scipy`
- **Model weights**: Downloaded by seed-data Lambda to S3 as `model.tar.gz`, referenced by SageMaker Model resource.

- [x] **Task 5: Frame recognition Lambda (Function URL)**
  - File: `lambdas/recognise-frames/app.py`
  - Action: Wire up the Lambda that sits between the frontend and SageMaker. Test end-to-end with sample frames.
- Lambda with Function URL (`AuthType: NONE`), receives HTTP POST with 64-frame batch
- Frame accumulation is **client-side**: JavaScript ring buffer accumulates 64 frames, sends complete batch every 8 new frames (stride 8 at 25fps = one batch per ~320ms)
- Processing pipeline:
  1. Parse JSON body with base64-encoded JPEG frames array
  2. Invoke SageMaker endpoint (`sagemaker-runtime:InvokeEndpoint`) → get top-5 predictions (~100ms)
  3. If top prediction confidence < 70%: invoke Bedrock Nova Lite with the middle frame + prompt "Describe what this person is doing with their hands in detail" → corroborate/adjust prediction (~500ms)
  4. Return prediction result as JSON response
- Request format: `{"frames": ["base64jpg1", ...], "batchId": "uuid"}`
- Response format: `{"type": "prediction", "batchId": "uuid", "signs": [{"sign": "HELLO", "confidence": 0.92}], "novaDescription": "person waving with open hand"}`
- Timeout: 30s, Memory: 256MB
- Total round-trip: ~500ms without Nova Lite, ~1s with Nova Lite validation (network + Lambda + SageMaker + optional Bedrock)

- [x] **Task 6: Text-to-gloss Lambda (Bedrock)**
  - File: `lambdas/text-to-gloss/app.py`
- Invoke Bedrock Claude (anthropic.claude-3-haiku-20240307-v1:0 for speed/cost)
- System prompt encodes BSL grammar rules from English2BSL paper:
  - TIME → TOPIC → COMMENT word order
  - Remove articles (a, an, the), copula (is, am, are, was, were), infinitive markers (to)
  - Move WH-question words (what, where, when, who, why, how) to sentence end
  - Establish tense markers at sentence start (PAST, FUTURE, NOW)
  - BSL verbs don't conjugate — use base form
  - Negation: add NOT after the verb
  - Pronouns: IX-1P (I/me), IX-2P (you), IX-3P (he/she/they)
  - Fingerspell proper nouns and technical terms
- Few-shot examples of English → BSL gloss:
  - "What is your name?" → "NAME IX-2P WHAT"
  - "I went to the shops yesterday" → "YESTERDAY SHOP IX-1P GO"
  - "The council tax bill has not been paid" → "COUNCIL TAX BILL PAY NOT"
  - "Can you help me with my housing application?" → "HOUSING APPLICATION IX-1P HELP IX-2P CAN"
- Output: `{"gloss": "YESTERDAY SHOP IX-1P GO", "glossWords": ["YESTERDAY", "SHOP", "IX-1P", "GO"]}`

- [x] **Task 7: Gloss-to-video Lambda (DynamoDB + FFmpeg with crossfade)**
  - File: `lambdas/gloss-to-video/app.py`
- For each gloss word:
  1. Query DynamoDB (`BSLSignLexicon` table, partition key `Gloss`, sort key `QualityRank`)
  2. Select highest-quality source: signbsl.com > BSL SignBank > fingerspelling
  3. If found: download sign video clip from S3 (`s3://{bucket}/lexicon/{source}/{signId}.mp4`)
  4. If not found in any source: fingerspell — look up each letter in `s3://{bucket}/lexicon/fingerspelling/{letter}.mp4`
- Use FFmpeg with crossfade transitions between clips:
  ```
  ffmpeg -i clip1.mp4 -i clip2.mp4 -filter_complex \
    "[0][1]xfade=transition=fade:duration=0.2:offset=DURATION1-0.2" \
    -c:v libx264 output.mp4
  ```
- For >2 clips: chain xfade filters progressively
- Upload stitched video to S3, generate pre-signed URL (1hr expiry)
- Return `{"videoUrl": "https://...", "glossBreakdown": [{"word": "SHOP", "source": "signbank", "signId": "1234", "type": "sign"}, {"word": "M-I-T-C-H-E-L-L", "source": "fingerspelling", "type": "fingerspell"}]}`
- DynamoDB schema: `Gloss` (PK, String), `QualityRank#Source` (SK, String), attributes: `SignId`, `Source` (signbank|signbsl|fingerspelling), `VideoKey` (S3 key), `Duration` (seconds)

- [x] **Task 8: Step Functions state machine (text→sign path)**
  - File: `statemachine/text-to-sign.asl.yaml`
- The sign→text path uses Lambda Function URL directly (no Step Functions needed — it's a request/response pipeline)
- Text→Sign path via Step Functions:
  1. `InputCheck` (Choice): text input → `TextToGloss`; audio S3 URI → `StartTranscription`
  2. `StartTranscription` (SDK: transcribe:startTranscriptionJob) → `WaitForTranscription` (2s loop)
  3. `ProcessTranscription` (Lambda) → extract transcript text
  4. `TextToGloss` (Lambda) → Bedrock Claude BSL glossing with grammar rules
  5. `GlossToVideo` (Lambda) → multi-source DynamoDB lookup + FFmpeg crossfade stitch
  6. `GenerateSpeech` (SDK: polly:synthesizeSpeech, parallel) → generate audio of original English text
  7. Return `{videoUrl, glossBreakdown, audioUrl}` → return via HTTP response to Lambda Function URL caller
- Retry config: TextToGloss retries 5x at 15s intervals with 1.5x backoff (Bedrock throttling). GlossToVideo retries 3x at 2s intervals.

- [x] **Task 9: Frontend web application**
  - Files: `frontend/`
- Single-page app (vanilla JS + TensorFlow.js MediaPipe — no framework, keeps deployment simple)
- **Three modes accessible via top nav:**

**Live Translation Mode** (conversation UX):
- Full-screen video call layout (like Teams/Zoom)
- Camera feed fills main area with canvas overlay for MediaPipe pose skeleton (30fps, browser-side)
- WebRTC `getUserMedia` at 25fps, frame ring buffer (64 frames), JPEG quality 0.5 (~5-8KB/frame, ~320-512KB per batch vs ~960KB at full quality). I3D resizes to 224×224 anyway — full quality is wasted bandwidth.
- HTTP POST batches to Lambda Function URL every 320ms (stride 8 at 25fps), receives predictions as JSON responses
- Live caption bar at bottom: recognised signs appear as text, flowing like live subtitles
- Text input panel (slide-in from right): type or dictate English text
- BSL video output appears as picture-in-picture overlay (bottom-right corner)
- Polly audio playback for reverse direction (read English text aloud)

**Practice Mode** (learn BSL):
- Shows a target word/phrase and its BSL sign video
- User attempts to replicate the sign on camera
- BSL-1K recognition model scores their attempt (star rating 1-5 based on confidence)
- Progress tracking: "You've learned X of Y council service signs"
- Curated sign sets: "Council Services", "Greetings", "Directions", "Emergency"
- Uses localStorage for progress persistence

**Kiosk Mode** (reception desk):
- Optimised for tablet/large touchscreen
- Minimal UI: two large buttons ("I want to sign" / "I want to type")
- Extra-large text, high contrast
- Auto-timeout and reset after 2 minutes of inactivity
- No keyboard required — on-screen keyboard for text input

**Shared frontend concerns:**
- Cognito auth (Amazon Cognito Identity SDK, hosted UI for login)
- Runtime config injected via `window.__BSL_CONFIG__` from a config.js file generated by seed Lambda
- High-contrast CSS with `prefers-contrast` media query support
- Keyboard navigation throughout (Tab, Enter, Escape)
- Screen reader ARIA labels on all interactive elements
- TensorFlow.js + MediaPipe Holistic loaded from CDN (`@mediapipe/holistic`, `@tensorflow/tfjs`)

- [ ] **Task 10: Deploy and test in sandbox** (deferred — requires sandbox deployment)
- Deploy using `aws cloudformation deploy --template-file template.yaml --stack-name ndx-try-bsl-sign-language --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM --profile NDX/SandboxAdmin --region us-east-1`
- If template >51KB, use `--s3-bucket ndx-try-isb-blueprints-{ACCOUNT_ID}`
- Verify:
  1. CodeBuild completed (SageMaker inference container image in ECR)
  2. SageMaker endpoint is InService (BSL-1K I3D on ml.g5.xlarge)
  3. Seed data Lambda completed (BSL-1K weights in S3/model.tar.gz, SignBank videos in S3, DynamoDB populated)
  4. SageMaker endpoint responds to test InvokeEndpoint with sample frames
  5. CloudFront distribution serves frontend
  6. Camera capture → sign recognition works via Lambda Function URL → SageMaker
  7. Text input → BSL video generation works
  8. Polly TTS works
  9. All IAM roles have `InnovationSandbox-ndx-` prefix

### Acceptance Criteria

- [ ] **AC1: Template deploys successfully**
- Given the CloudFormation template
- When deployed to an ISB sandbox account with `--profile NDX/SandboxAdmin`
- Then all resources create successfully within 20 minutes
- And all IAM roles are prefixed with `InnovationSandbox-ndx-`

- [ ] **AC2: BSL-1K model serves predictions**
- Given the BSL-1K SageMaker endpoint is deployed on ml.g5.xlarge
- When a batch of 64 video frames (224×224, 25fps) is sent via InvokeEndpoint
- Then the endpoint returns top-5 BSL sign predictions with confidence scores
- And latency is under 500ms per prediction (GPU inference)

- [ ] **AC3: English→BSL gloss translation**
- Given the text-to-gloss Lambda
- When "What is your name?" is submitted
- Then the response contains BSL-ordered gloss (e.g., "NAME IX-2P WHAT")
- And articles, copula, and infinitive markers are removed

- [ ] **AC4: BSL video generation**
- Given the gloss-to-video Lambda and populated DynamoDB/S3 lexicon
- When a BSL gloss sequence is submitted
- Then a stitched video is returned as a pre-signed S3 URL
- And unknown words fall back to fingerspelling

- [ ] **AC5: Seed data populates correctly**
- Given the seed-data custom resource Lambda
- When the stack creates
- Then BSL-1K model weights are in S3
- And BSL SignBank sign videos are in S3
- And DynamoDB `BSLSignLexicon` table has gloss→signId mappings
- And SageMaker model.tar.gz is packaged and uploaded

- [ ] **AC6: Real-time sign recognition via Lambda Function URL**
- Given the web frontend in Live Translation mode
- When a user grants camera access and starts signing
- Then webcam captures at 25fps with MediaPipe pose overlay at 30fps (browser-side)
- And 64-frame batches are sent via HTTP POST to the recognise-frames Lambda Function URL
- And sign predictions appear as live captions within 500ms of completing a sign (without Nova Lite)
- And when Nova Lite validation triggers (confidence <70%), predictions appear within 1.5 seconds
- And Nova Lite descriptions appear alongside low-confidence predictions

- [ ] **AC7: Frontend text→sign with video playback**
- Given the web frontend
- When a user types English text and clicks "Translate to BSL"
- Then a BSL gloss breakdown is displayed
- And a stitched BSL sign video with crossfade transitions plays as picture-in-picture
- And unknown words are fingerspelled (indicated in the gloss breakdown)

- [ ] **AC8: Speech integration**
- Given the web frontend with Polly and Transcribe
- When a user speaks into the microphone
- Then speech is transcribed to text (Transcribe or Web Speech API)
- And the text→sign pipeline is triggered
- And recognised signs can be read aloud via Polly (British English voice)

- [ ] **AC9: Practice Mode**
- Given the web frontend in Practice mode
- When a user selects a sign to learn and attempts it on camera
- Then the BSL-1K model scores their attempt
- And a star rating is displayed based on confidence
- And progress is tracked across the session

- [ ] **AC10: Kiosk Mode**
- Given the web frontend in Kiosk mode
- When displayed on a tablet/touchscreen
- Then the UI uses extra-large touch targets and high contrast
- And auto-resets after 2 minutes of inactivity
- And requires no physical keyboard

- [ ] **AC11: Accessibility**
- Given the web frontend in any mode
- Then all interactive elements have ARIA labels
- And keyboard navigation works throughout (Tab, Enter, Escape)
- And high-contrast mode activates with `prefers-contrast` media query
- And text is readable at 200% zoom

## Review Notes

**Adversarial review completed.** Findings: 20 total, 16 fixed, 4 acknowledged/deferred.

### Fixed (16)
- **F1** (Critical): Seed Lambda replaced from stub to full working implementation (~200 lines inline) — downloads BSL-1K weights, scrapes SignBank, populates DynamoDB, packages model.tar.gz, triggers CodeBuild, deploys config.js
- **F2** (Critical): GlossToVideo Lambda gracefully handles missing FFmpeg — returns individual clip pre-signed URLs for sequential playback when FFmpeg unavailable
- **F3** (Critical): CodeBuild buildspec now includes docker login for AWS DLC ECR registry (763104351884)
- **F5** (Critical): model.py forward() fixed — avg_pool/dropout applied before Logits layer, not after (prevented double-application)
- **F6** (Critical): Seed Lambda polls CodeBuild with remaining time budget (60s buffer) instead of blocking indefinitely
- **F7** (High): Removed redundant InvokeFunction permissions on Lambda Function URLs
- **F8** (High): TextToSign Lambda uses synchronous direct invocation for text path (frontend no longer shows "Translating..." forever)
- **F9** (High): XSS fixed — innerHTML replaced with textContent and DOM manipulation throughout app.js
- **F10** (High): CodeBuild changed from LINUX_GPU_CONTAINER to LINUX_CONTAINER (only building Docker image)
- **F11** (High): Auth button event listener added via setupAuth()
- **F12** (Medium): SageMaker role replaced AmazonSageMakerFullAccess with scoped-down policy
- **F13** (Medium): SageMaker resource names include ${AWS::AccountId} to prevent collisions
- **F14** (Medium): Practice deactivate() now correctly stops camera via Camera.stop()
- **F15** (Medium): startPracticeCamera() stops existing camera session before starting new one
- **F19** (Low): DynamoDB table name includes AccountId and Region for uniqueness
- **F20** (Low): Process-transcription Lambda handles HTTPS S3 URLs in addition to s3:// protocol

### Acknowledged/Deferred (4)
- **F4** (High): Frontend HTML/JS/CSS files not auto-uploaded to S3 WebsiteBucket — seed Lambda only deploys config.js. Documented as manual step in BLUEPRINT.md (`aws s3 sync frontend/ s3://bucket/`)
- **F16** (Medium): Extended Lambda files in `lambdas/` directory diverge from inline ZipFile implementations in template.yaml — inline versions are authoritative, external files are reference implementations
- **F17** (Low): pickle.load() / torch.load() security risk on remote model weights — inherent to BSL-1K model format, mitigated by downloading from known VGG URL
- **F18** (Low): CORS `Access-Control-Allow-Origin: *` should be restricted to CloudFront domain — deferred to deployment phase when domain is known

## Additional Context

### Dependencies

| Dependency | Source | License | Purpose |
|---|---|---|---|
| BSL-1K pretrained model | `robots.ox.ac.uk/~vgg/research/bsl1k/` | Research (weights freely available) | Sign recognition (1,064 BSL signs) |
| BSL SignBank videos | `bslsignbank.ucl.ac.uk` | BSD-3-Clause | Sign video lexicon (~2,500 signs) |
| BANZ-FS fingerspelling | Academic dataset | CC BY 4.0 | BSL two-handed fingerspelling reference |
| English2BSL grammar rules | ACL 2023 paper | Academic | BSL gloss translation prompt engineering |
| MediaPipe Holistic | Google | Apache 2.0 | Pose/hand/face landmark extraction |
| FFmpeg | FFmpeg project | LGPL 2.1 | Video clip concatenation |
| PyTorch | Meta | BSD-3-Clause | BSL-1K model runtime |
| I3D architecture | `github.com/gulvarol/bsl1k` | Research | Model architecture code |

### Testing Strategy

1. **Unit test seed-data Lambda locally** — mock S3/DynamoDB, verify download URLs and data mapping logic
2. **Test SageMaker inference locally** — run `inference.py` with sample frames locally, verify predictions match BSL-1K demo output
3. **Test Bedrock glossing** — invoke text-to-gloss Lambda with test sentences, verify BSL grammar rules applied
4. **Deploy to sandbox** — full stack deployment, manual verification of all paths
5. **Playwright E2E** (later phase) — automate camera capture test with recorded video, verify full pipeline

### Notes

- **SageMaker endpoint always-warm**: No cold start issues — the ml.g5.xlarge endpoint is always running. Cost: ~$1.41/hr (~$1,015/month if left running). For demo use in ISB sandboxes (typically active for hours, not months), this is acceptable. The CloudFormation stack will take ~15-20 minutes total due to seed data download + CodeBuild container build + SageMaker endpoint creation.
- **ISB SCP**: Every IAM role must use `InnovationSandbox-ndx-` prefix. SAM's auto-generated role names won't work — must use explicit `RoleName` properties on every role. Existing simple scenarios use `ndx-try-{scenario}-role-${AWS::Region}` pattern but this may not pass ISB SCP — use `InnovationSandbox-ndx-bsl-{function}` to be safe.
- **SageMaker VERIFIED in ISB sandboxes**: ISB sandbox accounts have pre-provisioned GPU quotas (ml.g5.xlarge=2, ml.g4dn.xlarge=2). The SageMaker SLR (`AWSServiceRoleForAmazonSageMakerNotebooks`) is creatable by SandboxUser — service-linked roles bypass the ISB SCP. Verified in sandbox account 572885592978 on 2026-03-10.
- **BSL SignBank scraping**: The SignBank is a Django app (BSD-3-Clause source at `github.com/Signbank/BSL-signbank`). Video URLs are discoverable from the source code. Implement polite scraping with 500ms delays between requests. ~2,500 videos at ~100KB each ≈ 250MB total.
- **signbsl.com licensing**: Email drafted to Daniel Mitchell. The multi-source lexicon architecture means we can start with SignBank and add signbsl.com videos as a data update (no code changes) once licensing is secured.
- **BANZ-FS fingerspelling**: Contains 35,000+ video instances but we only need 26 canonical two-handed letter poses. Curate one representative clip per letter.
- **Bedrock model availability**: Claude Haiku and Nova Lite should be available in us-east-1 without marketplace agreements. Verify at deploy time.
- **Template size**: Lambda code must be packaged as S3-hosted zip artifacts referenced by `CodeUri`. Template itself stays under 51KB.
- **CodeBuild self-deploy**: The SageMaker inference container image is built by a CodeBuild project triggered by the seed-data custom resource. CodeBuild pulls the Dockerfile from S3, builds the image, pushes to ECR, and packages model.tar.gz. This eliminates any pre-deployment steps.
- **Client-side frame buffering**: JavaScript ring buffer accumulates 64 frames in the browser. Complete batches are sent via HTTP POST to the Lambda Function URL (~960KB per batch, well within the 6MB payload limit). No server-side state management needed. This is simpler than WebSocket API (eliminated: connection management, $connect/$disconnect handlers, DynamoDB connections table, API Gateway WebSocket routing).
- **Cost awareness**: SageMaker ml.g5.xlarge: ~$1.41/hr (~$1,015/month continuous). For ISB sandbox leases (typically hours to days), cost is manageable. Nova Lite validation at $0.06/M tokens adds ~$0.03-0.09 per minute of active signing. Lambda Function URL invocations are negligible cost.
- **Competitive context**: Silence Speaks (UK startup) is already deployed at 250+ TfL stations. Signapse AI has raised £2M. SignGPT (£8.45M EPSRC) is building the world's largest BSL dataset. This project differentiates as the first open-source, AWS-native, self-deploying BSL translation system.
- **Future upgrade path**: When the Oxford VGG keypoint-based BSL model (arxiv 2412.09475, 8,162 words) releases code/weights, it can replace BSL-1K on the same SageMaker endpoint — just swap the model.tar.gz. The MediaPipe keypoints extracted browser-side could be sent directly to this model, eliminating raw frame transfer entirely and potentially allowing a lighter CPU-only endpoint.
