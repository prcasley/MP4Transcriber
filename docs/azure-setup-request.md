# Request: Azure AD App Registration for Auto-Transcription

## What I Need

Access to register an app in Azure Active Directory (Entra ID) for our SharePoint video auto-transcription tool, OR have an admin register it on my behalf.

## Why We Need It

We have video/audio files uploaded to SharePoint that need to be transcribed. Currently this is a manual process. I've built a free transcription tool (TranscribeHQ) that uses AI-powered speech-to-text, but it needs permission to read and write files in our SharePoint folder to automate the workflow.

The Azure AD app registration allows our tool to securely access a specific SharePoint folder — it does NOT require any paid licenses, software purchases, or third-party subscriptions.

## What It Replaces

- Power Automate Premium connectors ($15/user/month) — NOT needed
- Copilot Studio ($200/month) — NOT needed
- Third-party transcription services (Otter.ai, Rev, etc.) — NOT needed

## Total Cost: $0

- Azure AD App Registration: Free (included with Microsoft 365)
- GitHub Actions (runs the automation): Free tier (2000 minutes/month)
- Groq Whisper API (AI transcription): Free tier
- Everything runs on infrastructure we already have

## What It Does

1. Every 15 minutes, checks a SharePoint folder for new video/audio files
2. Automatically transcribes them using AI (Whisper large-v3)
3. Saves the transcript (.txt and .srt subtitles) back to the same folder
4. No manual intervention needed after setup

## Security

- The app only gets access to SharePoint files (Sites.ReadWrite.All)
- Credentials are stored as encrypted secrets in our GitHub repository
- No user passwords are involved — it uses a client certificate/secret
- The app can be scoped to specific sites if needed
- Full audit trail via Azure AD logs and GitHub Actions logs
- Can be revoked instantly by deleting the app registration

## What I Need From You (5-Minute Setup)

If you can do it yourself, here are the steps. Otherwise, please grant me access to Azure AD > App Registrations.

### Step-by-Step Setup (Azure Portal)

1. Go to https://portal.azure.com
2. Search for "App registrations" in the top search bar, click it
3. Click "+ New registration"
   - Name: `TranscribeHQ-AutoTranscribe`
   - Supported account types: "Accounts in this organizational directory only"
   - Redirect URI: Leave blank
   - Click "Register"

4. On the app's Overview page, copy these two values and send them to me:
   - **Application (client) ID**
   - **Directory (tenant) ID**

5. Go to "Certificates & secrets" in the left sidebar
   - Click "+ New client secret"
   - Description: `TranscribeHQ`
   - Expires: 24 months
   - Click "Add"
   - **COPY THE VALUE IMMEDIATELY** (it won't be shown again) and send it to me securely

6. Go to "API permissions" in the left sidebar
   - Click "+ Add a permission"
   - Select "Microsoft Graph"
   - Select "Application permissions" (NOT delegated)
   - Search for and check: `Sites.ReadWrite.All`
   - Click "Add permissions"
   - Click "Grant admin consent for [our organization]"
   - Confirm "Yes"

That's it. Send me the 3 values (tenant ID, client ID, client secret) and I'll configure the rest.

### What I'll Do With Those Values

- Store them as encrypted secrets in our private GitHub repository
- Configure the automation workflow to poll our SharePoint folder
- Test with a sample video to verify it works
- The automation will run hands-free from that point on
