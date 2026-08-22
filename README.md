# Manuscript Restoration Platform

Developed by **Team Vellum Node**.
##Team Workflow

- Adyasha Mishra: Leader, Backend Developer, & Logic Integration.
- Dhruti Pragyan Parida: Communicator & AI Integration, Material Researcher.
- Ritupurna: Frontend Developer & AI Integration, Material Researcher.
- Priyanshu Bisws: Backend Developer.
- Sonalika Naik: PPT Presentation & Video Editing.
- Abhipsa Majhi: PPT Presentation & Video Editing.


## AI-Assisted Restoration and Scholarly Reconstruction of Damaged Manuscripts and Inscriptions

Vellum Node is a beginner-friendly Streamlit application created for the **SOAIDEATHON-S9 hackathon**. It supports the restoration and scholarly study of faded manuscript pages, palm-leaf records, inscriptions, and archival documents.

The application preserves the original uploaded image, creates a separate enhanced working copy, extracts text using OCR, identifies uncertain readings, suggests possible reconstructions, and records human review decisions for provenance.

> Vellum Node provides machine-assisted suggestions for scholarly review. It does not claim that an automated reconstruction is historical truth.

## Features

- Local user registration and login.
- Secure local password hashing using PBKDF2-SHA256.
- User-specific manuscript history and provenance records.
- Separation of each user’s manuscripts and generated image files.
- Preservation of the original uploaded image.
- OpenCV-based image enhancement.
- Tesseract OCR text extraction.
- Confidence-based identification of uncertain words.
- RapidFuzz-based reconstruction suggestions.
- Evidence crops for uncertain OCR regions.
- Accept, reject, or manually edit suggested readings.
- SQLite database for simple local storage.
- Branded Vellum Node interface.
- Home, About, Services, Login, Dashboard, Upload, Process, Review, and Provenance sections.
- Automated authentication and OCR workflow tests.

## Application Workflow

The main workflow is:

```text
Register or log in
        ↓
Upload manuscript image
        ↓
Preserve original image
        ↓
Create enhanced working copy
        ↓
Run Tesseract OCR
        ↓
Identify uncertain words
        ↓
Generate reconstruction suggestions
        ↓
Review evidence and suggestions
        ↓
Accept, reject, or manually edit
        ↓
Save provenance history
```

The original image is never overwritten. Enhancement and OCR operate on separate working files.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application programming language |
| Streamlit | User interface and local web application |
| SQLite | Local database for accounts, manuscripts, OCR, suggestions, and reviews |
| OpenCV | Image enhancement and evidence-crop generation |
| Tesseract OCR | Text extraction from manuscript images |
| RapidFuzz | Approximate matching against the local reference corpus |
| Pandas | Displaying provenance and review tables |
| Pillow | Image handling support |

## Project Structure

```text
manuscript_app/
│
├── app.py                         # Main Streamlit application
├── requirements.txt               # Python dependencies
├── corpus.txt                     # Local demo reference corpus
├── sample_input.jpg               # Bundled sample image for testing
├── smoke_test.py                  # OCR and review workflow test
├── test_auth.py                   # Authentication and ownership test
├── README.md                      # Project documentation
│
├── assets/
│   ├── vellum_node_logo.png       # Vellum Node team logo
│   └── login_layout_reference.jpg # Login design reference
│
├── .vscode/
│   ├── settings.json              # VS Code settings
│   └── launch.json                # VS Code Run and Debug configuration
│
└── data/                          # Created automatically at runtime
    ├── app.db                     # Local SQLite database
    ├── originals/                  # Preserved original images
    ├── enhanced/                  # Enhanced working images
    └── crops/                      # Evidence crops for uncertain words
```

## Requirements

Before running the application, install:

- Python 3.11 or newer.
- Visual Studio Code, recommended for development.
- Tesseract OCR.
- Git, optional but recommended for GitHub use.

For Windows, use the 64-bit Python installer unless you specifically have a 32-bit computer.

## Installation on Windows

### 1. Install Python

Download and install Python from:

[https://www.python.org/downloads/](https://www.python.org/downloads/)

During installation, enable:

```text
Add Python to PATH
```

You can verify Python with:

```powershell
py --version
```

If the `py` command is unavailable, try:

```powershell
python --version
```

### 2. Install Visual Studio Code

Download VS Code from:

[https://code.visualstudio.com/](https://code.visualstudio.com/)

Install the Microsoft Python extension from the VS Code Extensions panel.

### 3. Install Tesseract OCR

Download the latest 64-bit Windows installer from the Tesseract UB Mannheim page:

[https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)

Choose the latest installer whose name looks similar to:

```text
tesseract-ocr-w64-setup-5.x.x.exe
```

During installation, use the suggested installation folder if possible:

```text
C:\Program Files\Tesseract-OCR
```

Close and reopen VS Code after installation.

Verify Tesseract in a new PowerShell terminal:

```powershell
tesseract --version
where.exe tesseract
```

The path should look similar to:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

If `tesseract` is not recognized, add this folder to the Windows PATH environment variable:

```text
C:\Program Files\Tesseract-OCR
```

Then completely close and reopen VS Code.

## Installation on macOS

Install Python from [https://www.python.org/downloads/](https://www.python.org/downloads/) or use Homebrew.

Install Tesseract with Homebrew:

```bash
brew install tesseract
```

Verify the installation:

```bash
tesseract --version
```

## Installation on Ubuntu or Debian

Install Python, pip, and Tesseract:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip tesseract-ocr
```

Verify the installation:

```bash
tesseract --version
```

## Set Up the Project in VS Code

Extract the ZIP file or clone the repository. Open the folder containing `app.py` in VS Code.

Open **Terminal → New Terminal**.

### Windows PowerShell

```powershell
cd "C:\path\to\manuscript_app"
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, run this once:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate the environment again:

```powershell
.venv\Scripts\Activate.ps1
```

### macOS or Linux

```bash
cd /path/to/manuscript_app
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

When the virtual environment is active, the terminal should show a prefix similar to:

```text
(.venv)
```

## Select the VS Code Python Interpreter

In VS Code:

1. Press `Ctrl+Shift+P` on Windows or `Cmd+Shift+P` on macOS.
2. Choose **Python: Select Interpreter**.
3. Select the interpreter inside `.venv`.

On Windows, the path is usually:

```text
.venv\Scripts\python.exe
```

On macOS or Linux, the path is usually:

```text
.venv/bin/python
```

Verify the selected Python:

### Windows

```powershell
where.exe python
python --version
```

### macOS or Linux

```bash
which python
python --version
```

## Verify Python Dependencies

With the virtual environment activated, run:

```bash
python -c "import streamlit, cv2, pytesseract, rapidfuzz, pandas, PIL; print('Python packages: OK')"
```

Expected output:

```text
Python packages: OK
```

Verify that Python can access Tesseract:

```bash
python -c "import pytesseract; print(pytesseract.get_tesseract_version())"
```

A Tesseract version number should be printed.

## Run the Application

From the project folder, with `.venv` activated, run:

```bash
streamlit run app.py
```

If the `streamlit` command is not recognized, use:

```bash
python -m streamlit run app.py
```

Streamlit normally opens a browser automatically. If it does not, open:

```text
http://localhost:8501
```

Keep the terminal open while using the application.

To stop the application, return to the terminal and press:

```text
Ctrl+C
```

## First-Time User Workflow

1. Open the application in a browser.
2. Click **Login**.
3. Choose **Register**.
4. Enter a display name, email address, and password.
5. Log in with the new account.
6. Open the Dashboard.
7. Upload a manuscript image.
8. Open Process and run enhancement plus OCR.
9. Open Review Suggestions.
10. Inspect the evidence crop and candidate readings.
11. Accept, reject, or manually edit a suggestion.
12. Open Provenance to view the recorded review action.
13. Log out and test another account.

## User Data Isolation

Each account has an independent local workspace. Manuscripts, OCR results, suggestions, reviews, and provenance records are filtered using the logged-in user’s owner ID.

Generated files are placed into user-specific folders similar to:

```text
data/originals/user_1/
data/enhanced/user_1/
data/crops/user_1/
```

To verify isolation manually:

1. Create an account for Alice.
2. Upload and process a manuscript as Alice.
3. Log out.
4. Create an account for Bob.
5. Confirm that Bob cannot see Alice’s manuscript.
6. Upload a different manuscript as Bob.
7. Log back into Alice’s account.
8. Confirm that Alice still sees only Alice’s records.

## Automated Tests

### Authentication and ownership test

Run:

```bash
python test_auth.py
```

A successful result should contain:

```text
'ownership_isolation': 'PASS'
'login_verification': 'PASS'
'status': 'PASS'
```

### Full OCR workflow test

Run:

```bash
python smoke_test.py
```

A successful result should contain:

```text
'ocr_items': ...
'uncertain_items': ...
'status': 'PASS'
```

The exact OCR counts may differ depending on the operating system, Tesseract version, image quality, and installed language data.

To test another image:

### Windows PowerShell

```powershell
python smoke_test.py "C:\path\to\your\image.jpg"
```

### macOS or Linux

```bash
python smoke_test.py "/path/to/your/image.jpg"
```

## Reset the Local Demo

Resetting the local demo deletes local user accounts, manuscripts, OCR results, suggestions, provenance history, and generated files.

Stop Streamlit before resetting.

### Windows PowerShell

```powershell
Remove-Item data\app.db -ErrorAction SilentlyContinue
Remove-Item data\originals\* -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item data\enhanced\* -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item data\crops\* -Recurse -Force -ErrorAction SilentlyContinue
```

### macOS or Linux

```bash
rm -f data/app.db
rm -rf data/originals/* data/enhanced/* data/crops/*
```

The application recreates the database and folders automatically when it starts again.

> Do not delete `.venv` when resetting application data. `.venv` is the Python environment, not application data.

## Troubleshooting

### `streamlit` is not recognized

Make sure the virtual environment is active:

```bash
.venv\Scripts\Activate.ps1
```

Then run:

```bash
python -m streamlit run app.py
```

### `tesseract` is not recognized on Windows

Confirm that Tesseract is installed in:

```text
C:\Program Files\Tesseract-OCR
```

Add the folder to Windows PATH, restart VS Code, and run:

```powershell
tesseract --version
where.exe tesseract
```

### PowerShell blocks activation

Run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate:

```powershell
.venv\Scripts\Activate.ps1
```

### The browser does not open

Open the following address manually:

```text
http://localhost:8501
```

### Port 8501 is already in use

Start Streamlit on another port:

```bash
streamlit run app.py --server.port 8502
```

Then open:

```text
http://localhost:8502
```

### Pylance shows nullable-value warnings

These are editor type-checking warnings caused by SQLite values that may be `None`. They do not necessarily mean the application failed. Run the tests from the terminal to verify actual behavior:

```bash
python -m py_compile app.py test_auth.py smoke_test.py
python test_auth.py
python smoke_test.py
```

### Windows reports that a SQLite file is being used

Stop Streamlit with `Ctrl+C`, close any running test or debug session, and run the test again. The application uses explicit SQLite connection cleanup for Windows compatibility.

### No OCR text is detected

Try a clearer, higher-resolution image with better contrast. Confirm that Tesseract is installed and that the image is a supported JPG, JPEG, or PNG file.

### No reconstruction suggestions appear

The application uses the words in `corpus.txt` as its local reference corpus. Add relevant words to that file, one word or phrase per line, and process the image again.

## Limitations

This project is designed for a local hackathon demonstration. It is not a production cloud service.

The current version uses:

- Local SQLite storage.
- Local folders for uploaded and generated files.
- Local password authentication.
- No email verification.
- No password-reset service.
- No cloud backup.
- No multi-machine synchronization.
- No production deployment security configuration.

For a production system, the project would need a hosted database, protected object storage, HTTPS, stronger account-management workflows, rate limiting, backups, access-control auditing, and a production authentication provider.

## Hackathon Value

Vellum Node addresses the challenge through a scholar-in-the-loop workflow:

- The original source is preserved.
- Enhancement is separated from the original image.
- OCR confidence is made visible.
- Uncertain readings are flagged rather than silently changed.
- Candidate reconstructions are linked to evidence crops.
- Scholars can accept, reject, or manually edit suggestions.
- Every review action is stored in a provenance history.
- User accounts separate each scholar’s work locally.

## Team

**Vellum Node**

Heritage reconstruction through AI-assisted scholarly review.

## License

This project was created as a hackathon prototype. Add the appropriate license before public production or commercial use.
