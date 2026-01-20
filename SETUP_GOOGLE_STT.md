# Setting Up Google Cloud Speech-to-Text for WhatsApp Voice Calls

Since Gemini Live's built-in speech recognition doesn't work well with phone audio quality, we use Google Cloud Speech-to-Text (STT) to transcribe phone audio first.

## Step 1: Create Google Cloud Service Account

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Select your project (or create one)
3. Go to **IAM & Admin** → **Service Accounts**
4. Click **Create Service Account**
   - Name: `whatsapp-voice-stt`
   - Description: `Service account for WhatsApp voice STT`
5. Click **Create and Continue**
6. Add roles:
   - `Cloud Speech Client`
   - `Cloud Speech Admin` (optional, for full access)
7. Click **Continue** → **Done**

## Step 2: Create and Download Key

1. Click on your new service account
2. Go to **Keys** tab
3. Click **Add Key** → **Create new key**
4. Select **JSON** format
5. Click **Create** (downloads the JSON file)

## Step 3: Enable Speech-to-Text API

1. Go to **APIs & Services** → **Library**
2. Search for "Cloud Speech-to-Text API"
3. Click **Enable**

## Step 4: Configure the Application

### Option A: Using credentials file path

1. Copy the downloaded JSON file to the project directory
2. Add to `.env`:
```
GOOGLE_APPLICATION_CREDENTIALS=/app/google-credentials.json
```

3. Update `docker-compose.yml` to mount the file:
```yaml
gemini-live:
  volumes:
    - ./google-credentials.json:/app/google-credentials.json:ro
```

### Option B: Using credentials JSON directly (recommended for Docker)

1. Open the downloaded JSON file
2. Copy the entire contents
3. Add to `.env` (as a single line):
```
GOOGLE_CREDENTIALS_JSON={"type":"service_account","project_id":"your-project",...}
```

4. Update `docker-compose.yml` to pass the variable:
```yaml
gemini-live:
  environment:
    - GOOGLE_CREDENTIALS_JSON=${GOOGLE_CREDENTIALS_JSON}
```

## Step 5: Update docker-compose.yml

Add the credentials environment variable:

```yaml
gemini-live:
  build:
    context: .
    dockerfile: Dockerfile.gemini
  container_name: gemini-live-service
  ports:
    - "8003:8003"
  environment:
    - GOOGLE_API_KEY=${GOOGLE_API_KEY}
    - GEMINI_LIVE_PORT=8003
    - GOOGLE_APPLICATION_CREDENTIALS=/app/google-credentials.json
    # OR use JSON directly:
    # - GOOGLE_CREDENTIALS_JSON=${GOOGLE_CREDENTIALS_JSON}
  volumes:
    - ./FAWI_Call_BOT.txt:/app/FAWI_Call_BOT.txt:ro
    - ./google-credentials.json:/app/google-credentials.json:ro
```

## Step 6: Test

1. Rebuild and restart:
```bash
docker-compose build gemini-live
docker-compose up -d
```

2. Check logs for STT initialization:
```bash
docker-compose logs -f gemini-live
```

You should see:
```
Google STT Service created
Pipeline created: input → STT → user_agg → LLM → output → assistant_agg
```

3. Make a test call and check for transcriptions in logs:
```
👤 USER: [transcribed text from phone]
🤖 BOT: [Gemini's response]
```

## Troubleshooting

### "Could not automatically determine credentials"
- Ensure `GOOGLE_APPLICATION_CREDENTIALS` or `GOOGLE_CREDENTIALS_JSON` is set
- Check that the JSON file exists and is readable

### "Permission denied" or "API not enabled"
- Enable "Cloud Speech-to-Text API" in Google Cloud Console
- Ensure service account has correct roles

### "Invalid credentials"
- Re-download the JSON key file
- Ensure JSON is valid (no extra whitespace or formatting)

## Costs

Google Cloud Speech-to-Text pricing:
- First 60 minutes/month: FREE
- After that: ~$0.006 per 15 seconds

For testing, you'll likely stay within the free tier.
