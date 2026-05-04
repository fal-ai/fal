#!/usr/bin/env bash
# Verify a pre-signed POST policy by uploading a tiny PNG with multipart/form-data.
# Requires: python (with boto3), curl, AWS creds, and an existing bucket "vedat-test".

set -uo pipefail

PAYLOAD="$(mktemp -t presigned-test.XXXXXX.png)"
trap 'rm -f "$PAYLOAD"' EXIT

# Tiny 1x1 PNG so the bytes are real image bytes.
printf '\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\xcf\xc0\x00\x00\x00\x03\x00\x01\x5b\xa6\xeb\x4f\x00\x00\x00\x00IEND\xaeB`\x82' > "$PAYLOAD"

echo "==> Generating pre-signed POST policy"
POLICY_JSON="$(python generate_url.py)"
echo "$POLICY_JSON" | python -m json.tool

URL="$(echo "$POLICY_JSON" | python -c 'import json,sys; print(json.load(sys.stdin)["url"])')"

# Build curl -F args for every field in the policy, then for the file last.
mapfile -t FORM_ARGS < <(python -c '
import json, sys
policy = json.loads(sys.argv[1])
for name, value in policy["fields"].items():
    value = value.replace("${filename}", "test.png")
    print("-F")
    print(f"{name}={value}")
print("-F")
print("Content-Type=image/png")
' "$POLICY_JSON")

echo $FORM_ARGS

echo
echo "==> POSTing to $URL"
curl -sS -i -X POST "$URL" "${FORM_ARGS[@]}" -F "file=@${PAYLOAD};type=image/png" \
  | head -40
