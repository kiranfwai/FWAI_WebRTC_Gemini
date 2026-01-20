# How to Generate a Permanent WhatsApp API Token

## The Problem
WhatsApp Business API requires a **System User Access Token** with specific permissions. App Access Tokens (generated from App ID + Secret) don't have user-level permissions.

## The Solution: System User Token (NEVER EXPIRES!)

### Step 1: Go to Meta Business Suite
1. Visit: https://business.facebook.com/settings/system-users
2. Log in with your Facebook account that has admin access

### Step 2: Create a System User (if you don't have one)
1. Click **"Add"** button
2. Enter a name (e.g., "WhatsApp API Bot")
3. Select role: **Admin**
4. Click **Create System User**

### Step 3: Assign Assets to System User
1. Click on your system user
2. Click **"Add Assets"**
3. Select **Apps** → Select your WhatsApp app (ID: 800389698460627)
4. Enable **Full Control**
5. Click **Save Changes**

### Step 4: Generate Permanent Token
1. Click on your system user
2. Click **"Generate New Token"**
3. Select your app (800389698460627)
4. Select these permissions:
   - ✅ `whatsapp_business_management`
   - ✅ `whatsapp_business_messaging`
5. Click **Generate Token**
6. **COPY THE TOKEN** (you won't see it again!)

### Step 5: Update Your Configuration
Add the token to your `.env` file:

```bash
META_ACCESS_TOKEN=EAAL...your_new_token_here...
```

Then restart:
```bash
docker-compose down && docker-compose up -d --build
```

## Why This Token Never Expires
System User tokens are **permanent** by design. They don't expire like regular user tokens. This is Meta's recommended approach for server-to-server API calls.

## Alternative: Graph API Explorer (Temporary Token)
If you need a quick test token:
1. Go to: https://developers.facebook.com/tools/explorer/
2. Select your app
3. Add permissions: `whatsapp_business_messaging`, `whatsapp_business_management`
4. Click **Generate Access Token**
5. Exchange for long-lived token (60 days)

**Note:** Explorer tokens expire in 60 days. System User tokens are permanent.

## Your App Details
- **App ID:** 800389698460627
- **Phone Number ID:** 100948263067135
