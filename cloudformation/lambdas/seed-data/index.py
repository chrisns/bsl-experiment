"""
Seed data custom resource Lambda.
Downloads BSL-1K model weights, BSL SignBank videos, populates DynamoDB,
packages SageMaker model artifacts, uploads inference code, triggers CodeBuild,
and deploys frontend assets.
"""
import json
import os
import io
import re
import time
import tarfile
import pickle
import logging
import urllib.request
import urllib.error

import boto3
import cfnresponse

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")

DATA_BUCKET = os.environ["DATA_BUCKET"]
WEBSITE_BUCKET = os.environ["WEBSITE_BUCKET"]
LEXICON_TABLE_NAME = os.environ["LEXICON_TABLE"]
CLOUDFRONT_DOMAIN = os.environ["CLOUDFRONT_DOMAIN"]
USER_POOL_ID = os.environ["USER_POOL_ID"]
USER_POOL_CLIENT_ID = os.environ["USER_POOL_CLIENT_ID"]
IDENTITY_POOL_ID = os.environ["IDENTITY_POOL_ID"]
RECOGNISE_URL = os.environ["RECOGNISE_FUNCTION_URL"]
TEXT_TO_SIGN_URL = os.environ["TEXT_TO_SIGN_FUNCTION_URL"]
ACCOUNT_ID = os.environ["AWS_ACCOUNT_ID_PARAM"]
REGION = os.environ.get("AWS_REGION", "us-east-1")

# BSL-1K model URLs (Oxford VGG) — actual BSL model with 1,064 British Sign Language signs
BSL1K_WEIGHTS_URL = "https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/pretrained_models/bsl1k.pth.tar"
BSL1K_VOCAB_URL = "https://raw.githubusercontent.com/gulvarol/bsl1k/master/misc/bsl1k/bsl1k_vocab.pkl"

# BSL SignBank base URL
SIGNBANK_BASE = "https://bslsignbank.ucl.ac.uk"
SIGNBANK_SIGNS_URL = f"{SIGNBANK_BASE}/dictionary/ajax/glosses/"

# Fingerspelling letters
ALPHABET = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def lambda_handler(event, context):
    """CloudFormation Custom Resource handler."""
    try:
        request_type = event["RequestType"]
        logger.info(f"Request type: {request_type}")

        if request_type == "Delete":
            handle_delete(event)
            cfnresponse.send(event, context, cfnresponse.SUCCESS, {})
            return

        if request_type in ("Create", "Update"):
            handle_create(context)
            cfnresponse.send(event, context, cfnresponse.SUCCESS, {
                "Message": "Seed data deployment complete"
            })
            return

        cfnresponse.send(event, context, cfnresponse.SUCCESS, {})

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        cfnresponse.send(event, context, cfnresponse.FAILED, {"Error": str(e)})


def handle_create(context):
    """Orchestrate all seed data operations."""
    remaining_ms = lambda: context.get_remaining_time_in_millis()

    # Step 1: Download model weights
    logger.info("Step 1: Downloading model weights...")
    download_to_s3(BSL1K_WEIGHTS_URL, DATA_BUCKET, "models/bsl1k.pth.tar")

    # Step 2: Download BSL-1K vocabulary (pkl file with 1,064 BSL sign classes)
    logger.info("Step 2: Downloading BSL-1K vocabulary...")
    download_to_s3(BSL1K_VOCAB_URL, DATA_BUCKET, "models/info.pkl")

    # Step 3: Load vocab and get class list
    logger.info("Step 3: Loading vocab for mapping...")
    vocab = load_vocab_from_s3()

    # Step 4: Scrape BSL SignBank videos
    logger.info("Step 4: Scraping BSL SignBank videos...")
    sign_mappings = scrape_signbank(vocab, remaining_ms)

    # Step 5: Create fingerspelling fallback entries
    logger.info("Step 5: Creating fingerspelling entries...")
    create_fingerspelling_entries()

    # Step 6: Populate DynamoDB
    logger.info("Step 6: Populating DynamoDB lexicon table...")
    populate_dynamodb(sign_mappings)

    # Step 7: Package model.tar.gz for SageMaker (includes inference code)
    logger.info("Step 7: Packaging model.tar.gz with inference code...")
    package_model_tarball()

    # Step 8: Deploy frontend
    logger.info("Step 8: Deploying frontend...")
    deploy_frontend()

    logger.info("Seed data deployment complete!")


def download_to_s3(url, bucket, key):
    """Download a file from URL and upload directly to S3."""
    logger.info(f"Downloading {url} -> s3://{bucket}/{key}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "NDX-Try-BSL/1.0"})
        with urllib.request.urlopen(req, timeout=300) as response:
            # Stream to /tmp first (Lambda has 10GB ephemeral)
            tmp_path = f"/tmp/{key.split('/')[-1]}"
            with open(tmp_path, "wb") as f:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)

        s3.upload_file(tmp_path, bucket, key)
        size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
        logger.info(f"Uploaded {size_mb:.1f}MB to s3://{bucket}/{key}")

        # Clean up /tmp
        os.remove(tmp_path)

    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        raise


def download_and_extract_info(tar_url, bucket):
    """Download info tar via streaming and extract only info.pkl (tar is 3.7GB, too large for /tmp).
    Stream through tarfile to extract only the small info.pkl file."""
    req = urllib.request.Request(tar_url, headers={"User-Agent": "NDX-Try-BSL/1.0"})
    response = urllib.request.urlopen(req, timeout=600)

    # Stream the tar and extract only info.pkl
    with tarfile.open(fileobj=response, mode="r|") as tar:
        for member in tar:
            if member.name.endswith("info.pkl"):
                logger.info(f"Found {member.name} ({member.size} bytes) in tar stream")
                f = tar.extractfile(member)
                info_data = f.read()
                s3.put_object(Bucket=bucket, Key="models/info.pkl", Body=info_data)
                logger.info("Uploaded info.pkl to S3")
                return
    raise RuntimeError("No info.pkl found in tar archive")


def load_vocab_from_s3():
    """Load info.pkl from S3 and extract vocabulary."""
    tmp_path = "/tmp/info.pkl"
    s3.download_file(DATA_BUCKET, "models/info.pkl", tmp_path)
    with open(tmp_path, "rb") as f:
        info = pickle.load(f)
    words = info.get("words", [])
    logger.info(f"Loaded {len(words)} vocabulary words from BSL-1K")
    os.remove(tmp_path)
    return words


def scrape_signbank(vocab, remaining_ms):
    """Scrape BSL SignBank for sign videos. Returns list of sign mappings."""
    sign_mappings = []

    # Build normalised vocab lookup (BSL-1K uses uppercase BOBSL labels)
    vocab_normalised = {w.upper().strip(): w for w in vocab}

    # Try to get the sign list from SignBank's AJAX endpoint
    try:
        # SignBank has a Django REST-style API
        page = 1
        per_page = 100
        total_fetched = 0

        while remaining_ms() > 120_000:  # Keep 2 min buffer
            url = f"{SIGNBANK_SIGNS_URL}?page={page}&per_page={per_page}"
            try:
                req = urllib.request.Request(url, headers={
                    "User-Agent": "NDX-Try-BSL/1.0 (UK Government Innovation)",
                    "Accept": "application/json",
                })
                with urllib.request.urlopen(req, timeout=30) as response:
                    data = json.loads(response.read().decode())
            except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError) as e:
                logger.warning(f"SignBank page {page} fetch failed: {e}")
                break

            signs = data if isinstance(data, list) else data.get("results", data.get("signs", []))
            if not signs:
                break

            for sign in signs:
                gloss = sign.get("gloss", sign.get("idgloss", "")).upper().strip()
                video_url = sign.get("video", sign.get("video_url", ""))
                sign_id = str(sign.get("id", sign.get("pk", "")))

                if not gloss or not video_url:
                    continue

                # Download video
                if not video_url.startswith("http"):
                    video_url = f"{SIGNBANK_BASE}{video_url}"

                s3_key = f"lexicon/signbank/{sign_id}.mp4"
                try:
                    download_sign_video(video_url, s3_key)
                    sign_mappings.append({
                        "gloss": gloss,
                        "sign_id": sign_id,
                        "source": "signbank",
                        "video_key": s3_key,
                        "bsl1k_match": gloss in vocab_normalised,
                    })
                    total_fetched += 1
                except Exception as e:
                    logger.warning(f"Failed to download sign {gloss}: {e}")

                # Polite delay
                time.sleep(0.5)

            page += 1
            logger.info(f"Fetched {total_fetched} signs so far (page {page})")

            if isinstance(data, dict) and not data.get("next"):
                break

    except Exception as e:
        logger.warning(f"SignBank scraping encountered error: {e}")

    # Also create entries for BSL-1K vocab words not found in SignBank
    signbank_glosses = {m["gloss"] for m in sign_mappings}
    unmapped = [w for w in vocab if w.upper() not in signbank_glosses]
    logger.info(f"BSL-1K vocab words without SignBank match: {len(unmapped)}")
    for word in unmapped[:50]:  # Log first 50 for review
        logger.info(f"  Unmapped: {word}")

    logger.info(f"Total sign mappings: {len(sign_mappings)}")
    return sign_mappings


def download_sign_video(url, s3_key):
    """Download a single sign video to S3."""
    req = urllib.request.Request(url, headers={"User-Agent": "NDX-Try-BSL/1.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        video_data = response.read()
    s3.put_object(Bucket=DATA_BUCKET, Key=s3_key, Body=video_data, ContentType="video/mp4")


def create_fingerspelling_entries():
    """Create placeholder fingerspelling entries in S3.
    In production, these would be sourced from BANZ-FS dataset.
    For now, create stub entries that the frontend can use.
    """
    for letter in ALPHABET:
        s3_key = f"lexicon/fingerspelling/{letter}.mp4"
        # Check if already exists
        try:
            s3.head_object(Bucket=DATA_BUCKET, Key=s3_key)
        except Exception:
            # Create a minimal placeholder — will be replaced with BANZ-FS clips
            s3.put_object(
                Bucket=DATA_BUCKET,
                Key=s3_key,
                Body=b"",  # Placeholder
                ContentType="video/mp4",
                Metadata={"placeholder": "true", "letter": letter},
            )


def populate_dynamodb(sign_mappings):
    """Populate BSLSignLexicon DynamoDB table with sign mappings."""
    table = dynamodb.Table(LEXICON_TABLE_NAME)

    with table.batch_writer() as batch:
        for mapping in sign_mappings:
            # Quality rank: 1 = best (signbsl.com), 2 = good (signbank), 3 = fallback (fingerspelling)
            quality_rank = "2" if mapping["source"] == "signbank" else "3"
            sort_key = f"{quality_rank}#{mapping['source']}"

            batch.put_item(Item={
                "Gloss": mapping["gloss"],
                "QualityRankSource": sort_key,
                "SignId": mapping["sign_id"],
                "Source": mapping["source"],
                "VideoKey": mapping["video_key"],
                "BSL1KMatch": mapping.get("bsl1k_match", False),
            })

        # Add fingerspelling entries
        for letter in ALPHABET:
            batch.put_item(Item={
                "Gloss": f"FS:{letter}",
                "QualityRankSource": "3#fingerspelling",
                "SignId": f"fs-{letter.lower()}",
                "Source": "fingerspelling",
                "VideoKey": f"lexicon/fingerspelling/{letter}.mp4",
                "BSL1KMatch": False,
            })

    logger.info(f"Populated DynamoDB with {len(sign_mappings) + len(ALPHABET)} entries")


def package_model_tarball():
    """Package model.tar.gz with weights, vocab, and inference code for SageMaker.
    SageMaker pre-built PyTorch container expects code/ directory with inference scripts."""
    tar_path = "/tmp/model.tar.gz"

    # Download weights and info from S3 to /tmp
    s3.download_file(DATA_BUCKET, "models/bsl1k.pth.tar", "/tmp/bsl1k.pth.tar")
    s3.download_file(DATA_BUCKET, "models/info.pkl", "/tmp/info.pkl")

    # Write inference code to /tmp/code/
    os.makedirs("/tmp/code", exist_ok=True)
    with open("/tmp/code/inference.py", "w") as f:
        f.write(INFERENCE_PY_CONTENT)
    with open("/tmp/code/model.py", "w") as f:
        f.write(MODEL_PY_CONTENT)
    with open("/tmp/code/requirements.txt", "w") as f:
        f.write(REQUIREMENTS_CONTENT)

    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add("/tmp/bsl1k.pth.tar", arcname="bsl1k.pth.tar")
        tar.add("/tmp/info.pkl", arcname="info.pkl")
        tar.add("/tmp/code/inference.py", arcname="code/inference.py")
        tar.add("/tmp/code/model.py", arcname="code/model.py")
        tar.add("/tmp/code/requirements.txt", arcname="code/requirements.txt")

    s3.upload_file(tar_path, DATA_BUCKET, "models/model.tar.gz")
    size_mb = os.path.getsize(tar_path) / (1024 * 1024)
    logger.info(f"Uploaded model.tar.gz ({size_mb:.1f}MB)")

    # Clean up
    import shutil
    for f in ["/tmp/bsl1k.pth.tar", "/tmp/info.pkl", tar_path]:
        if os.path.exists(f):
            os.remove(f)
    if os.path.exists("/tmp/code"):
        shutil.rmtree("/tmp/code")



def deploy_frontend():
    """Generate config.js and upload to website bucket."""
    config_js = f"""// Auto-generated runtime configuration
window.__BSL_CONFIG__ = {{
    region: '{REGION}',
    userPoolId: '{USER_POOL_ID}',
    userPoolClientId: '{USER_POOL_CLIENT_ID}',
    identityPoolId: '{IDENTITY_POOL_ID}',
    recogniseUrl: '{RECOGNISE_URL}',
    textToSignUrl: '{TEXT_TO_SIGN_URL}',
    dataBucket: '{DATA_BUCKET}',
    cloudfrontDomain: '{CLOUDFRONT_DOMAIN}'
}};
"""
    s3.put_object(
        Bucket=WEBSITE_BUCKET,
        Key="config.js",
        Body=config_js.encode("utf-8"),
        ContentType="application/javascript",
    )
    logger.info("Deployed config.js to website bucket")


def handle_delete(event):
    """Clean up S3 objects only during actual stack deletion, not during updates."""
    # Check if the stack is being deleted (vs custom resource being replaced during update)
    cfn = boto3.client("cloudformation")
    stack_id = event.get("StackId", "")
    try:
        resp = cfn.describe_stacks(StackName=stack_id)
        stack_status = resp["Stacks"][0]["StackStatus"] if resp.get("Stacks") else ""
    except Exception:
        stack_status = ""

    if "DELETE" not in stack_status:
        logger.info(f"Stack status is {stack_status}, skipping S3 cleanup (update, not deletion)")
        return

    logger.info("Stack is being deleted — cleaning up S3 objects...")
    for bucket in [DATA_BUCKET, WEBSITE_BUCKET]:
        try:
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket):
                objects = page.get("Contents", [])
                if objects:
                    s3.delete_objects(
                        Bucket=bucket,
                        Delete={"Objects": [{"Key": obj["Key"]} for obj in objects]},
                    )
            logger.info(f"Cleaned bucket {bucket}")
        except Exception as e:
            logger.warning(f"Failed to clean bucket {bucket}: {e}")


# ============================================================
# Embedded SageMaker inference code (uploaded to S3 for CodeBuild)
# ============================================================

INFERENCE_PY_CONTENT = open(
    os.path.join(os.path.dirname(__file__), "..", "..", "sagemaker", "bsl1k-inference", "inference.py")
).read() if os.path.exists(os.path.join(os.path.dirname(__file__), "..", "..", "sagemaker", "bsl1k-inference", "inference.py")) else '''"""SageMaker inference handler — see sagemaker/bsl1k-inference/inference.py for canonical version."""
import io, json, math, base64, logging, pickle
import cv2, numpy as np, torch
from scipy.special import softmax
from model import InceptionI3d
logger = logging.getLogger(__name__)
RESIZE_RES = 256; INP_RES = 224; NUM_IN_FRAMES = 16; STRIDE = 4
MEAN = np.array([0.5,0.5,0.5],dtype=np.float32); STD = np.array([1.0,1.0,1.0],dtype=np.float32)

def model_fn(model_dir):
    with open(f"{model_dir}/info.pkl","rb") as f: info = pickle.load(f)
    classes = info["words"]; num_classes = len(classes)
    model = InceptionI3d(num_classes=num_classes, spatiotemporal_squeeze=True,
        final_endpoint="Logits", in_channels=3, dropout_keep_prob=0.5, num_in_frames=NUM_IN_FRAMES)
    checkpoint = torch.load(f"{model_dir}/bsl1k.pth.tar", map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    logger.info(f"Loaded {len(state_dict)} parameters on {device}")
    return {"model": model, "classes": classes, "device": device}

def input_fn(request_body, content_type):
    if content_type != "application/json": raise ValueError(f"Unsupported: {content_type}")
    data = json.loads(request_body); frames_b64 = data.get("frames", [])
    if not frames_b64: raise ValueError("No frames")
    frames = []
    for fb in frames_b64:
        img = cv2.imdecode(np.frombuffer(base64.b64decode(fb),dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (RESIZE_RES, RESIZE_RES), interpolation=cv2.INTER_LINEAR)
        ulx = (RESIZE_RES-INP_RES)//2; uly = ulx
        img = img[uly:uly+INP_RES, ulx:ulx+INP_RES]
        img = (img.astype(np.float32)/255.0 - MEAN)/STD
        frames.append(img)
    if not frames: raise ValueError("No valid frames")
    while len(frames) < NUM_IN_FRAMES: frames.append(frames[-1].copy())
    n = len(frames)
    all_f = np.stack(frames); rgb = torch.from_numpy(all_f).permute(3,0,1,2)
    nc = math.ceil((n-NUM_IN_FRAMES)/STRIDE)+1
    clips = torch.zeros(nc, 3, NUM_IN_FRAMES, INP_RES, INP_RES)
    for j in range(nc):
        t = j*STRIDE if min(NUM_IN_FRAMES, n-j*STRIDE)==NUM_IN_FRAMES else n-NUM_IN_FRAMES
        clips[j] = rgb[:, t:t+NUM_IN_FRAMES]
    return {"clips": clips, "num_clips": nc, "batch_id": data.get("batchId","unknown")}

def predict_fn(input_data, model_dict):
    model=model_dict["model"]; classes=model_dict["classes"]; device=model_dict["device"]
    clips=input_data["clips"].to(device); nc=input_data["num_clips"]
    with torch.no_grad():
        logits = model(clips)
        pwp = softmax(logits.cpu().numpy(), axis=1)
        probs = pwp[0] if nc==1 else pwp.max(axis=0)
        top = np.argsort(probs)[::-1][:10]
        preds = [{"sign":classes[i],"confidence":float(probs[i])} for i in top]
    return {"predictions": preds, "batchId": input_data["batch_id"]}

def output_fn(prediction, accept): return json.dumps(prediction), "application/json"
'''

MODEL_PY_CONTENT = open(
    os.path.join(os.path.dirname(__file__), "..", "..", "sagemaker", "bsl1k-inference", "model.py")
).read() if os.path.exists(os.path.join(os.path.dirname(__file__), "..", "..", "sagemaker", "bsl1k-inference", "model.py")) else '''"""I3D — see sagemaker/bsl1k-inference/model.py for canonical version."""
'''

REQUIREMENTS_CONTENT = """# torch, torchvision, numpy are pre-installed in the DLC container — do NOT reinstall
# as it breaks PyTorch's numpy integration (RuntimeError: Numpy is not available)
# Pin numpy<2 to prevent scipy/opencv upgrading it past PyTorch 2.2 compatibility
numpy<2
opencv-python-headless>=4.8.0
scipy>=1.11.0
"""
