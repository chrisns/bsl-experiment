"""
Text-to-BSL-gloss translation Lambda.
Uses Bedrock Claude Haiku with BSL grammar rules from English2BSL ACL 2023 paper.
"""
import json
import os
import logging

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

bedrock_runtime = boto3.client("bedrock-runtime")
MODEL_ID = os.environ.get("MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")

BSL_SYSTEM_PROMPT = """You are an expert BSL (British Sign Language) gloss translator. Convert English text to BSL gloss notation following these rules from BSL linguistics:

## BSL Grammar Rules
1. **Word order**: TIME → TOPIC → COMMENT (not English SVO)
2. **Remove**: articles (a, an, the), copula (is, am, are, was, were), infinitive markers (to)
3. **WH-questions**: Move question words (what, where, when, who, why, how) to sentence END
4. **Tense markers**: Place at sentence START — PAST, FUTURE, NOW (present is unmarked)
5. **Verbs**: Use base form only — BSL verbs do not conjugate
6. **Negation**: Add NOT after the verb
7. **Pronouns**: IX-1P (I/me), IX-2P (you), IX-3P (he/she/they)
8. **Proper nouns and technical terms**: Mark for fingerspelling with FS: prefix
9. **Numbers**: Use numeral form
10. **Plurals**: Add quantity or MANY/FEW before the noun

## Output Format
Return ONLY a JSON object with:
- "gloss": the BSL gloss string (uppercase words separated by spaces)
- "glossWords": array of individual gloss words

Do NOT include any explanation, just the JSON."""

FEW_SHOT_EXAMPLES = [
    {"english": "What is your name?", "gloss": "NAME IX-2P WHAT", "glossWords": ["NAME", "IX-2P", "WHAT"]},
    {"english": "I went to the shops yesterday", "gloss": "YESTERDAY SHOP IX-1P GO", "glossWords": ["YESTERDAY", "SHOP", "IX-1P", "GO"]},
    {"english": "The council tax bill has not been paid", "gloss": "COUNCIL TAX BILL PAY NOT", "glossWords": ["COUNCIL", "TAX", "BILL", "PAY", "NOT"]},
    {"english": "Can you help me with my housing application?", "gloss": "HOUSING APPLICATION IX-1P HELP IX-2P CAN", "glossWords": ["HOUSING", "APPLICATION", "IX-1P", "HELP", "IX-2P", "CAN"]},
    {"english": "Where is the nearest bus stop?", "gloss": "NEAR BUS STOP WHERE", "glossWords": ["NEAR", "BUS", "STOP", "WHERE"]},
    {"english": "My appointment is at 3 o'clock tomorrow", "gloss": "TOMORROW 3 APPOINTMENT IX-1P", "glossWords": ["TOMORROW", "3", "APPOINTMENT", "IX-1P"]},
]


def lambda_handler(event, context):
    """Translate English text to BSL gloss."""
    text = event.get("text", "")
    if not text:
        return {"gloss": "", "glossWords": [], "originalText": ""}

    # Build messages with few-shot examples
    messages = []
    for example in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": [{"type": "text", "text": example["english"]}]})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": json.dumps({"gloss": example["gloss"], "glossWords": example["glossWords"]})}]})

    messages.append({"role": "user", "content": [{"type": "text", "text": text}]})

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 512,
        "system": BSL_SYSTEM_PROMPT,
        "messages": messages,
    })

    response = bedrock_runtime.invoke_model(
        modelId=MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=body,
    )

    result = json.loads(response["body"].read().decode())
    content_text = result["content"][0]["text"]

    try:
        gloss_data = json.loads(content_text)
        gloss = gloss_data.get("gloss", "")
        gloss_words = gloss_data.get("glossWords", gloss.split())
    except json.JSONDecodeError:
        # Fallback: treat response as raw gloss string
        gloss = content_text.strip().upper()
        gloss_words = gloss.split()

    logger.info(f"Translated: '{text}' -> '{gloss}'")

    return {
        "gloss": gloss,
        "glossWords": gloss_words,
        "originalText": text,
    }
