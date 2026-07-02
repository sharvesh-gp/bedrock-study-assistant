# Smart Study Buddy

An AI-powered study assistant for engineering students. Upload a PDF of your lecture notes or textbook chapter and get a summary, quiz, or answers to your questions — all powered by Amazon Bedrock.

## Prerequisites

- Python 3.9+
- AWS account with Bedrock access enabled for `amazon.titan-text-premier-v1:0` in `us-east-1`

## AWS Credentials Setup

Smart Study Buddy uses boto3's standard credential resolution. Configure credentials using one of these methods:

**Option 1 — Environment variables:**
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

**Option 2 — AWS credentials file (`~/.aws/credentials`):**
```ini
[default]
aws_access_key_id = your_access_key
aws_secret_access_key = your_secret_key
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
python app.py
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

## Usage

1. Click the upload area and select a PDF file (max 5MB, first 10 pages processed)
2. Choose a study mode:
   - **Summarize** — get a concise academic summary of the content
   - **Quiz Me** — generate 5 multiple-choice questions with answers
   - **Ask a Question** — type a question and get an answer based on the PDF
3. Click **Submit** and wait for the AI response
