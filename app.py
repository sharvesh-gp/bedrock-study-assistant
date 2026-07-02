import json
import os

import boto3
import botocore.exceptions
import PyPDF2
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

MAX_PAGES = 10
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB
ALLOWED_EXTENSIONS = {".pdf"}
ALLOWED_MIME_TYPES = {"application/pdf"}
VALID_MODES = {"summarize", "quiz", "ask"}


def extract_text_from_pdf(file_stream) -> str:
    """Extract text from a PDF file stream, capped at MAX_PAGES pages.

    Args:
        file_stream: A file-like object containing the PDF data.

    Returns:
        Extracted text with page separators.

    Raises:
        ValueError: If no readable text is found in the PDF.
    """
    reader = PyPDF2.PdfReader(file_stream)
    pages = reader.pages[:MAX_PAGES]

    page_texts = []
    for i, page in enumerate(pages, start=1):
        text = page.extract_text() or ""
        page_texts.append((i, text))

    # Join non-empty page texts with separators
    parts = []
    for i, text in page_texts:
        if parts:
            parts.append(f"\n\n--- Page {i} ---\n\n")
        parts.append(text)

    result = "".join(parts).strip()

    if not result:
        raise ValueError("No readable text found in the PDF.")

    return result


def build_prompt(mode: str, text: str, question: str = "") -> str:
    """Construct a mode-specific prompt for the Bedrock model.

    Args:
        mode: One of 'summarize', 'quiz', or 'ask'.
        text: The extracted PDF text to include in the prompt.
        question: The user's question (required when mode is 'ask').

    Returns:
        A prompt string ready to send to the model.
    """
    if mode == "summarize":
        return (
            "Please provide a concise academic summary of the following text. "
            "Focus on the key concepts, definitions, and important points.\n\n"
            f"{text}"
        )
    elif mode == "quiz":
        return (
            "Based on the following text, generate exactly 5 multiple-choice questions "
            "to test understanding of the material. For each question, provide 4 answer "
            "choices labelled A, B, C, and D. After all 5 questions, include an "
            "\"Answers:\" section listing the correct answer for each question.\n\n"
            f"{text}"
        )
    elif mode == "ask":
        return (
            f"Using only the information provided in the text below, answer the following question:\n\n"
            f"Question: {question}\n\n"
            f"Text:\n{text}"
        )
    else:
        raise ValueError(f"Unknown mode: {mode!r}")


def invoke_bedrock(prompt: str) -> str:
    """Invoke the Amazon Bedrock Titan Text Premier model with the given prompt.

    Args:
        prompt: The prompt string to send to the model.

    Returns:
        The generated text from the model.

    Raises:
        RuntimeError: If the API call fails or credentials are missing.
    """
    client = boto3.client("bedrock-runtime", region_name="us-east-1")

    body = {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 2048,
            "temperature": 0.7,
            "topP": 0.9,
        },
    }

    try:
        response = client.invoke_model(
            modelId="amazon.titan-text-premier-v1:0",
            body=json.dumps(body),
        )
        response_body = json.loads(response["body"].read())
        return response_body["results"][0]["outputText"]
    except botocore.exceptions.NoCredentialsError:
        raise RuntimeError(
            "AWS credentials not configured. See README for setup instructions."
        )
    except botocore.exceptions.ClientError:
        raise RuntimeError("AI service error. Please try again.")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    try:
        # --- File presence check ---
        if "file" not in request.files or request.files["file"].filename == "":
            return jsonify({"success": False, "error": "No file uploaded."})

        file = request.files["file"]

        # --- Extension check ---
        _, ext = os.path.splitext(file.filename.lower())
        if ext not in ALLOWED_EXTENSIONS:
            return jsonify({"success": False, "error": "Only PDF files are accepted."})

        # --- MIME type check ---
        if file.mimetype not in ALLOWED_MIME_TYPES:
            return jsonify({"success": False, "error": "Only PDF files are accepted."})

        # --- Size check (read into memory to measure) ---
        file_bytes = file.read()
        if len(file_bytes) > MAX_FILE_SIZE_BYTES:
            return jsonify({"success": False, "error": "File exceeds the 5MB size limit."})

        # --- Mode validation ---
        mode = request.form.get("mode", "").strip()
        if mode not in VALID_MODES:
            return jsonify({"success": False, "error": "Please select a study mode."})

        # --- Question validation for ask mode ---
        question = request.form.get("question", "").strip()
        if mode == "ask" and not question:
            return jsonify({"success": False, "error": "Please enter a question."})

        # --- Orchestration ---
        import io
        text = extract_text_from_pdf(io.BytesIO(file_bytes))
        prompt = build_prompt(mode, text, question=question)
        result = invoke_bedrock(prompt)

        return jsonify({"success": True, "result": result})

    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)})
    except RuntimeError as exc:
        return jsonify({"success": False, "error": str(exc)})
    except Exception:
        return jsonify({"success": False, "error": "An unexpected error occurred. Please try again."})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
