from __future__ import annotations

import base64
import hashlib
import json
import hmac
import os
import re
import secrets
import shutil
import textwrap
import sqlite3
import unicodedata
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st
from pytesseract import Output
from rapidfuzz import fuzz, process


# ============================================================
# TESSERACT AUTO-DISCOVERY
# ============================================================

def configure_tesseract() -> None:
    potential_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
        shutil.which("tesseract"),
    ]
    for path in potential_paths:
        if path and Path(path).exists():
            pytesseract.pytesseract.tesseract_cmd = str(path)
            break

configure_tesseract()


# ============================================================
# PATHS / CONFIG
# ============================================================

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
ORIGINALS_DIR = DATA_DIR / "originals"
ENHANCED_DIR = DATA_DIR / "enhanced"
CROPS_DIR = DATA_DIR / "crops"
ASSETS_DIR = APP_DIR / "assets"
DB_PATH = DATA_DIR / "app.db"
CORPUS_PATH = APP_DIR / "corpus.txt"

LOGO_PATH = ASSETS_DIR / "vellum_node_logo.png"
BACKGROUND_PATH = ASSETS_DIR / "Background.png"

for directory in (ORIGINALS_DIR, ENHANCED_DIR, CROPS_DIR, ASSETS_DIR):
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# OPENCV SAFE FILE I/O
# ============================================================

def safe_imread(path: Path | str) -> np.ndarray | None:
    try:
        data = Path(path).read_bytes()
        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def safe_imwrite(path: Path | str, image: np.ndarray) -> bool:
    try:
        suffix = Path(path).suffix.lower() or ".png"
        success, encoded = cv2.imencode(suffix, image)
        if success:
            Path(path).write_bytes(encoded.tobytes())
            return True
        return False
    except Exception:
        return False


# ============================================================
# DATABASE & AUTH HELPERS
# ============================================================

def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_db() -> None:
    with connect() as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                display_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS manuscripts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                owner_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                collection_name TEXT NOT NULL,
                script_name TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                original_path TEXT NOT NULL,
                enhanced_path TEXT,
                extracted_full_text TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (owner_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS ocr_words (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                manuscript_id INTEGER NOT NULL,
                word_text TEXT NOT NULL,
                confidence REAL NOT NULL,
                x INTEGER NOT NULL,
                y INTEGER NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                crop_path TEXT,
                status TEXT NOT NULL DEFAULT 'normal',
                FOREIGN KEY (manuscript_id) REFERENCES manuscripts(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS suggestions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ocr_word_id INTEGER NOT NULL,
                suggested_text TEXT NOT NULL,
                match_score REAL NOT NULL,
                corpus_name TEXT NOT NULL,
                rank INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                FOREIGN KEY (ocr_word_id) REFERENCES ocr_words(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS review_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                manuscript_id INTEGER NOT NULL,
                ocr_word_id INTEGER NOT NULL,
                suggestion_id INTEGER,
                action TEXT NOT NULL,
                previous_text TEXT,
                final_text TEXT,
                reviewer TEXT NOT NULL,
                reason TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (manuscript_id) REFERENCES manuscripts(id) ON DELETE CASCADE,
                FOREIGN KEY (ocr_word_id) REFERENCES ocr_words(id) ON DELETE CASCADE,
                FOREIGN KEY (suggestion_id) REFERENCES suggestions(id) ON DELETE SET NULL
            );
            CREATE TABLE IF NOT EXISTS pipeline_state (
                manuscript_id INTEGER PRIMARY KEY,
                current_stage TEXT NOT NULL DEFAULT 'Preserved',
                status TEXT NOT NULL DEFAULT 'ready',
                preserved_at TEXT,
                processed_at TEXT,
                reviewed_at TEXT,
                finalized_at TEXT,
                exported_at TEXT,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (manuscript_id) REFERENCES manuscripts(id) ON DELETE CASCADE
            );
            """
        )

        columns = {
            row["name"]
            for row in db.execute("PRAGMA table_info(manuscripts)").fetchall()
        }
        if "extracted_full_text" not in columns:
            db.execute("ALTER TABLE manuscripts ADD COLUMN extracted_full_text TEXT")

        db.execute(
            """
            INSERT INTO pipeline_state (manuscript_id, current_stage, status, preserved_at, updated_at)
            SELECT m.id,
                   CASE WHEN m.extracted_full_text IS NOT NULL THEN 'Processed' ELSE 'Preserved' END,
                   'ready',
                   m.created_at,
                   COALESCE(m.created_at, ?)
            FROM manuscripts m
            LEFT JOIN pipeline_state ps ON ps.manuscript_id = m.id
            WHERE ps.manuscript_id IS NULL
            """,
            (now(),),
        )


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
    return f"{salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        parts = stored.split("$", 1)
        if len(parts) != 2:
            return False
        salt_hex, digest_hex = parts
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def register_user(email: str, display_name: str, password: str) -> tuple[bool, str, int | None]:
    email = email.strip().lower()
    display_name = display_name.strip()

    if "@" not in email or "." not in email.split("@")[-1]:
        return False, "Enter a valid email address.", None
    if len(display_name) < 2:
        return False, "Display name must contain at least 2 characters.", None
    if len(password) < 6:
        return False, "Password must contain at least 6 characters.", None

    try:
        with connect() as db:
            cursor = db.execute(
                """
                INSERT INTO users (email, display_name, password_hash, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (email, display_name, hash_password(password), now()),
            )
            inserted_id = cursor.lastrowid
            if inserted_id is None:
                raise RuntimeError("User ID was not generated by SQLite.")
            return True, "Account created. You can now log in.", int(inserted_id)
    except sqlite3.IntegrityError:
        return False, "An account with this email already exists.", None


def authenticate(email: str, password: str) -> sqlite3.Row | None:
    with connect() as db:
        user = db.execute(
            "SELECT * FROM users WHERE email = ? COLLATE NOCASE",
            (email.strip().lower(),),
        ).fetchone()

    if user and verify_password(password, user["password_hash"]):
        return user
    return None


def get_user(user_id: int) -> sqlite3.Row | None:
    with connect() as db:
        return db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ============================================================
# CORPUS & SCRIPT-AWARE MATCHING
# ============================================================

DEFAULT_INDIC_CORPUS = [
    "भगवद्गीता", "भगवती", "भगवान", "देवी", "महादेवी", "देवता", "प्रकृति", "प्रकृतितः",
    "अध्याय", "श्लोक", "नमः", "महात्म्य", "धर्म", "कर्म", "ब्रह्म", "ईश्वर",
    "स्तोत्र", "पुराण", "वेद", "उपनिषद्", "संस्कृत", "ऋषि", "मुनि", "विद्या",
    "परमेश्वर", "सर्वज्ञ", "ଆତ୍ମା", "ମୋକ୍ଷ", "ଜ୍ଞାନ", "ଭକ୍ତି", "ଶାନ୍ତି",
    "ଭଗବତୀ", "ଭଗବାନ", "ଦେବୀ", "ମହାଦେବୀ", "ପ୍ରକୃତି", "ଧର୍ମ", "କର୍ମ", "ଜଗନ୍ନାଥ",
    "ପୁରାଣ", "ଓଡ଼ିଆ", "ଶ୍ଳୋକ", "ସ୍ତୋତ୍ର", "ଶ୍ରୀମନ୍ଦିର", "ଶ୍ରୀଜୀଉ", "ମହାପ୍ରଭୁ"
]

def clean_ocr_token(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text.strip())
    cleaned = re.sub(r"[।॥|,\.\-_/\\;:'\"!?\(\)\[\]\{\}\<\>+=*~`#@%^0-9]", "", normalized)
    return cleaned.strip()


def load_corpus() -> list[str]:
    words = set()
    if CORPUS_PATH.exists():
        raw = CORPUS_PATH.read_text(encoding="utf-8").splitlines()
        for line in raw:
            for token in line.split():
                cleaned = clean_ocr_token(token)
                if cleaned:
                    words.add(cleaned)

    for term in DEFAULT_INDIC_CORPUS:
        words.add(unicodedata.normalize("NFC", term))

    return sorted(words)


def extract_best_suggestions(ocr_word: str, corpus: list[str], limit: int = 3) -> list[tuple[str, float]]:
    cleaned = clean_ocr_token(ocr_word)
    if not cleaned or not corpus:
        return []

    def score_match(candidate: str) -> float:
        r1 = fuzz.ratio(cleaned, candidate)
        r2 = fuzz.token_sort_ratio(cleaned, candidate)
        r3 = fuzz.partial_ratio(cleaned, candidate)
        return (r1 * 0.5) + (r2 * 0.3) + (r3 * 0.2)

    scored = []
    for candidate in corpus:
        score = score_match(candidate)
        if score > 15.0:
            scored.append((candidate, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:limit]


def user_dir(base: Path, owner_id: int) -> Path:
    directory = base / f"user_{owner_id}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def manuscripts(owner_id: int) -> list[sqlite3.Row]:
    with connect() as db:
        return db.execute(
            "SELECT * FROM manuscripts WHERE owner_id = ? ORDER BY id DESC",
            (owner_id,),
        ).fetchall()


def get_manuscript(manuscript_id: int, owner_id: int | None = None) -> sqlite3.Row | None:
    query = "SELECT * FROM manuscripts WHERE id = ?"
    params: list[Any] = [manuscript_id]
    if owner_id is not None:
        query += " AND owner_id = ?"
        params.append(owner_id)
    with connect() as db:
        return db.execute(query, params).fetchone()



PIPELINE_STAGES = [
    ("Preserved", "Original evidence"),
    ("Restore", "Working image"),
    ("OCR", "Text extraction"),
    ("Uncertainty", "Evidence analysis"),
    ("Review", "Human verification"),
    ("Finalized", "Scholarly approval"),
    ("Export", "Deliverable"),
]


def initialize_pipeline(manuscript_id: int) -> None:
    with connect() as db:
        db.execute(
            """
            INSERT OR REPLACE INTO pipeline_state
            (manuscript_id, current_stage, status, preserved_at, updated_at)
            VALUES (?, 'Preserved', 'ready', ?, ?)
            """,
            (manuscript_id, now(), now()),
        )


def get_pipeline_state(manuscript_id: int, owner_id: int) -> sqlite3.Row | None:
    with connect() as db:
        return db.execute(
            """
            SELECT ps.*
            FROM pipeline_state ps
            JOIN manuscripts m ON m.id = ps.manuscript_id
            WHERE ps.manuscript_id = ? AND m.owner_id = ?
            """,
            (manuscript_id, owner_id),
        ).fetchone()


def update_pipeline_stage(
    manuscript_id: int,
    owner_id: int,
    stage: str,
    status: str = "ready",
) -> None:
    timestamp_columns = {
        "Preserved": "preserved_at",
        "Processed": "processed_at",
        "Review": "reviewed_at",
        "Finalized": "finalized_at",
        "Export": "exported_at",
    }

    with connect() as db:
        valid = db.execute(
            "SELECT id FROM manuscripts WHERE id = ? AND owner_id = ?",
            (manuscript_id, owner_id),
        ).fetchone()
        if valid is None:
            raise ValueError("Manuscript not found for this user.")

        timestamp_column = timestamp_columns.get(stage)
        if timestamp_column:
            db.execute(
                f"""
                UPDATE pipeline_state
                SET current_stage = ?, status = ?, {timestamp_column} = ?, updated_at = ?
                WHERE manuscript_id = ?
                """,
                (stage, status, now(), now(), manuscript_id),
            )
        else:
            db.execute(
                """
                UPDATE pipeline_state
                SET current_stage = ?, status = ?, updated_at = ?
                WHERE manuscript_id = ?
                """,
                (stage, status, now(), manuscript_id),
            )


def pending_review_count(manuscript_id: int, owner_id: int) -> int:
    with connect() as db:
        row = db.execute(
            """
            SELECT COUNT(*) AS n
            FROM ocr_words ow
            JOIN manuscripts m ON m.id = ow.manuscript_id
            WHERE ow.manuscript_id = ? AND m.owner_id = ? AND ow.status = 'uncertain'
            """,
            (manuscript_id, owner_id),
        ).fetchone()
    return int(row["n"] if row else 0)


def pipeline_percent(stage: str) -> int:
    return {
        "Preserved": 14,
        "Restore": 28,
        "OCR": 42,
        "Uncertainty": 56,
        "Review": 70,
        "Finalized": 85,
        "Export": 100,
    }.get(stage, 0)


def save_upload(uploaded_file: Any, title: str, collection: str, script: str, owner_id: int) -> int:
    payload = uploaded_file.getvalue()
    digest = sha256_bytes(payload)[:16]
    safe_suffix = Path(uploaded_file.name).suffix.lower() or ".jpg"

    original_path = user_dir(ORIGINALS_DIR, owner_id) / f"manuscript_{digest}{safe_suffix}"
    if not original_path.exists():
        original_path.write_bytes(payload)

    with connect() as db:
        cursor = db.execute(
            """
            INSERT INTO manuscripts
            (owner_id, title, collection_name, script_name, original_filename, original_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (owner_id, title, collection, script, uploaded_file.name, str(original_path), now()),
        )
        inserted_id = cursor.lastrowid
        if inserted_id is None:
            raise RuntimeError("Manuscript ID was not returned by SQLite.")
        manuscript_id = int(inserted_id)

    initialize_pipeline(manuscript_id)
    return manuscript_id


# ============================================================
# MULTILINGUAL DIRECTORY & SCRIPT MAPPING
# ============================================================

LANGUAGE_LABELS: dict[str, str] = {
    "san": "Sanskrit (संस्कृतम्)",
    "hin": "Hindi (हिन्दी)",
    "ori": "Odia (ଓଡ଼ିଆ)",
    "ben": "Bengali (বাংলা)",
    "tam": "Tamil (தமிழ்)",
    "tel": "Telugu (తెలుగు)",
    "kan": "Kannada (ಕನ್ನಡ)",
    "mal": "Malayalam (മലയാളം)",
    "guj": "Gujarati (ગુજરાતી)",
    "pan": "Punjabi (ਪੰਜਾਬੀ)",
    "mar": "Marathi (मराठी)",
    "urd": "Urdu (اردو)",
    "ara": "Arabic (العربية)",
    "eng": "English",
    "lat": "Latin",
    "bod": "Tibetan (Classical)",
}

def get_installed_tesseract_languages() -> dict[str, str]:
    try:
        installed = pytesseract.get_languages(config="")
    except Exception:
        installed = ["eng"]

    valid_codes = [c for c in installed if c not in ("osd", "equ")]
    priority_order = ["san", "hin", "ori", "ben", "tam", "tel", "kan", "mal", "guj", "pan", "mar", "urd", "ara", "lat", "eng"]

    sorted_codes = sorted(
        valid_codes,
        key=lambda code: (priority_order.index(code) if code in priority_order else 999, code)
    )

    result = {}
    for code in sorted_codes:
        label = LANGUAGE_LABELS.get(code, f"{code.upper()} ({code})")
        result[label] = code

    return result


# ============================================================
# HIGH-ACCURACY DESKEWING & PREPROCESSING PIPELINE
# ============================================================

def deskew_image(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    coords = np.column_stack(np.where(gray < 128))
    if coords.size == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    if abs(angle) > 0.3:
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated
    return image


def preprocess_manuscript(image: np.ndarray, mode: str = "Adaptive Binary (B&W)") -> np.ndarray:
    straight = deskew_image(image)
    gray = cv2.cvtColor(straight, cv2.COLOR_BGR2GRAY) if len(straight.shape) == 3 else straight

    if mode == "Grayscale Contrast (CLAHE)":
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(denoised)

    elif mode == "Bilateral Smooth":
        smoothed = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        return smoothed

    elif mode == "Otsu Binary":
        scaled = cv2.resize(gray, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(scaled, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    else:  # Default: Adaptive Binary (B&W)
        scaled = cv2.resize(gray, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        denoised = cv2.fastNlMeansDenoising(scaled, None, h=10, templateWindowSize=7, searchWindowSize=21)
        binary = cv2.adaptiveThreshold(
            denoised, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            31, 15
        )
        h, w = binary.shape
        pad_y = max(8, int(h * 0.03))
        pad_x = max(8, int(w * 0.02))
        binary[:pad_y, :] = 255
        binary[-pad_y:, :] = 255
        binary[:, :pad_x] = 255
        binary[:, -pad_x:] = 255
        return binary



def restore_manuscript(
    manuscript_id: int,
    owner_id: int,
    preprocess_mode: str = "Adaptive Binary (B&W)",
) -> str:
    """Stage 02: create the scholarly working image without touching OCR records."""
    manuscript = get_manuscript(manuscript_id, owner_id)
    if manuscript is None:
        raise ValueError("Manuscript not found for this user.")

    original = safe_imread(manuscript["original_path"])
    if original is None:
        raise ValueError("The original image could not be read.")

    update_pipeline_stage(manuscript_id, owner_id, "Restore", "running")

    enhanced = preprocess_manuscript(original, mode=preprocess_mode)
    enhanced_path = user_dir(ENHANCED_DIR, owner_id) / f"manuscript_{manuscript_id}_enhanced.png"
    safe_imwrite(enhanced_path, enhanced)

    with connect() as db:
        db.execute(
            """
            UPDATE manuscripts
            SET enhanced_path = ?
            WHERE id = ? AND owner_id = ?
            """,
            (str(enhanced_path), manuscript_id, owner_id),
        )

    update_pipeline_stage(manuscript_id, owner_id, "Restore", "ready")
    return str(enhanced_path)



def ocr_preprocess_variants(enhanced: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """
    Build several OCR-friendly views from the already restored manuscript.
    The original working image is always included so enhancement never forces
    a single visual interpretation.
    """
    if enhanced is None or enhanced.size == 0:
        raise ValueError("Empty restored image.")

    base = enhanced
    if len(base.shape) == 3:
        gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    else:
        gray = base.copy()

    variants: list[tuple[str, np.ndarray]] = [("restored", base)]

    # Contrast-preserving grayscale view.
    denoised = cv2.fastNlMeansDenoising(
        gray, None, h=7, templateWindowSize=7, searchWindowSize=21
    )
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    variants.append(("clahe", clahe.apply(denoised)))

    # Otsu view.
    _, otsu = cv2.threshold(
        denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    variants.append(("otsu", otsu))

    # Adaptive view with milder parameters than the restoration default.
    adaptive = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        41,
        11,
    )
    variants.append(("adaptive", adaptive))

    return variants


def _ocr_data_candidates(
    image: np.ndarray,
    lang_code: str,
    psm: int,
) -> tuple[str, list[dict[str, Any]]]:
    """Run Tesseract and return full text plus structured word candidates."""
    config = f"--oem 1 --psm {psm}"
    full_text = pytesseract.image_to_string(image, lang=lang_code, config=config)
    data = pytesseract.image_to_data(
        image,
        lang=lang_code,
        output_type=Output.DICT,
        config=config,
    )

    candidates: list[dict[str, Any]] = []
    for i, raw_text in enumerate(data.get("text", [])):
        word = (raw_text or "").strip()
        if not word:
            continue
        try:
            confidence = float(data["conf"][i])
        except (ValueError, TypeError, KeyError, IndexError):
            confidence = -1.0
        if confidence < 0:
            continue

        candidates.append(
            {
                "text": word,
                "confidence": confidence,
                "block": int(data["block_num"][i]),
                "paragraph": int(data["par_num"][i]),
                "line": int(data["line_num"][i]),
                "word_num": int(data["word_num"][i]),
                "x": int(data["left"][i]),
                "y": int(data["top"][i]),
                "width": int(data["width"][i]),
                "height": int(data["height"][i]),
            }
        )
    return full_text, candidates


def _choose_ocr_candidates(
    runs: list[tuple[str, str, list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    """
    Select one reading per Tesseract reading position using confidence +
    cross-view agreement. This is intentionally conservative: OCR never
    invents a reading that is not present in at least one OCR pass.
    """
    grouped: dict[tuple[int, int, int, int], list[dict[str, Any]]] = {}

    for variant_name, lang_code, candidates in runs:
        for item in candidates:
            key = (
                item["block"],
                item["paragraph"],
                item["line"],
                item["word_num"],
            )
            enriched = dict(item)
            enriched["variant"] = variant_name
            enriched["lang"] = lang_code
            grouped.setdefault(key, []).append(enriched)

    chosen: list[dict[str, Any]] = []
    for key, items in grouped.items():
        # Aggregate normalized readings across independent views.
        votes: dict[str, list[dict[str, Any]]] = {}
        for item in items:
            norm = clean_ocr_token(item["text"])
            vote_key = norm if norm else item["text"]
            votes.setdefault(vote_key, []).append(item)

        ranked = []
        for vote_key, vote_items in votes.items():
            best = max(vote_items, key=lambda x: x["confidence"])
            mean_conf = sum(x["confidence"] for x in vote_items) / len(vote_items)
            agreement = len(vote_items)
            score = (best["confidence"] * 0.72) + (mean_conf * 0.18) + (agreement * 10.0)
            ranked.append((score, best, agreement))

        _, best, agreement = max(ranked, key=lambda row: row[0])
        best["agreement"] = agreement
        chosen.append(best)

    chosen.sort(
        key=lambda item: (
            item["block"],
            item["paragraph"],
            item["line"],
            item["word_num"],
            item["y"],
            item["x"],
        )
    )
    return chosen


def _reconstruct_full_text(candidates: list[dict[str, Any]]) -> str:
    """Reconstruct readable text from selected OCR words while preserving lines."""
    if not candidates:
        return ""

    lines: list[str] = []
    current_key: tuple[int, int, int] | None = None
    words: list[str] = []

    for item in candidates:
        line_key = (item["block"], item["paragraph"], item["line"])
        if current_key is not None and line_key != current_key:
            lines.append(" ".join(words))
            words = []
        current_key = line_key
        words.append(item["text"])

    if words:
        lines.append(" ".join(words))

    return "\n".join(lines)


def language_code_for_script(script_name: str, available_codes: set[str]) -> str | None:
    """Prefer a script-specific Tesseract model over a broad multi-language model."""
    s = (script_name or "").lower()
    preferences = [
        (("sanskrit", "संस्कृत", "devanagari"), "san"),
        (("hindi", "हिन्दी", "devanagari"), "hin"),
        (("odia", "oriya", "ଓଡ଼ିଆ"), "ori"),
        (("bengali", "বাংলা"), "ben"),
        (("tamil", "தமிழ்"), "tam"),
        (("telugu", "తెలుగు"), "tel"),
        (("kannada", "ಕನ್ನಡ"), "kan"),
        (("malayalam", "മലയാളം"), "mal"),
        (("gujarati", "ગુજરાતી"), "guj"),
        (("punjabi", "ਪੰਜਾਬੀ"), "pan"),
        (("marathi", "मराठी"), "mar"),
        (("urdu", "اردو"), "urd"),
        (("arabic", "العربية"), "ara"),
        (("latin",), "lat"),
        (("english",), "eng"),
    ]
    for needles, code in preferences:
        if code in available_codes and any(token in s for token in needles):
            return code
    return None


def run_ocr_stage(
    manuscript_id: int,
    owner_id: int,
    threshold: float,
    lang_code: str = "san",
    psm: int = 6,
    ensemble: bool = True,
) -> tuple[int, int]:
    """
    Stage 03: high-accuracy OCR on a restored working image.

    Ensemble mode runs multiple OCR-friendly image views independently and
    selects readings using confidence + cross-view agreement. For multi-language
    selections such as san+hin, each language is also evaluated independently.
    """
    manuscript = get_manuscript(manuscript_id, owner_id)
    if manuscript is None:
        raise ValueError("Manuscript not found for this user.")

    enhanced_path = manuscript["enhanced_path"]
    if not enhanced_path:
        raise ValueError("Restore the manuscript before running OCR.")

    enhanced = safe_imread(enhanced_path)
    if enhanced is None:
        raise ValueError("The restored image could not be read.")

    update_pipeline_stage(manuscript_id, owner_id, "OCR", "running")

    # Avoid feeding a broad multi-language bundle to every pass when several
    # language models were selected. Separate model runs are more interpretable.
    language_codes = [code for code in lang_code.split("+") if code]
    if not language_codes:
        language_codes = ["san"]

    variants = (
        ocr_preprocess_variants(enhanced)
        if ensemble
        else [("restored", enhanced)]
    )

    runs: list[tuple[str, str, list[dict[str, Any]]]] = []
    for variant_name, image in variants:
        # If a single language is selected, use it directly. For multiple
        # languages, run each model separately and let the selector compare them.
        for code in language_codes:
            _, candidates = _ocr_data_candidates(image, code, psm)
            runs.append((variant_name, code, candidates))

    selected = _choose_ocr_candidates(runs)
    full_text = _reconstruct_full_text(selected)

    # If an ensemble pass returns nothing useful, preserve a direct Tesseract
    # result rather than returning an empty transcription.
    if not selected:
        direct_text, _ = _ocr_data_candidates(enhanced, language_codes[0], psm)
        full_text = direct_text

    with connect() as db:
        db.execute(
            """
            DELETE FROM suggestions
            WHERE ocr_word_id IN (
                SELECT id FROM ocr_words WHERE manuscript_id = ?
            )
            """,
            (manuscript_id,),
        )
        db.execute("DELETE FROM ocr_words WHERE manuscript_id = ?", (manuscript_id,))
        db.execute(
            """
            UPDATE manuscripts
            SET extracted_full_text = ?
            WHERE id = ? AND owner_id = ?
            """,
            (full_text, manuscript_id, owner_id),
        )

        word_count = 0
        uncertain_count = 0

        img_h, img_w = enhanced.shape[:2]

        for item in selected:
            word = str(item["text"]).strip()
            confidence = float(item["confidence"])
            x = int(item["x"])
            y = int(item["y"])
            w = int(item["width"])
            h = int(item["height"])

            status = "uncertain" if confidence < threshold else "normal"
            crop_path = None

            if status == "uncertain" and w > 0 and h > 0:
                pad_x = max(12, int(w * 0.18))
                pad_y = max(12, int(h * 0.45))
                x0 = max(0, x - pad_x)
                y0 = max(0, y - pad_y)
                x1 = min(img_w, x + w + pad_x)
                y1 = min(img_h, y + h + pad_y)

                crop_path_obj = (
                    user_dir(CROPS_DIR, owner_id)
                    / f"manuscript_{manuscript_id}_ocr_{word_count + 1}.png"
                )
                crop_img = enhanced[y0:y1, x0:x1]
                if crop_img.size > 0:
                    safe_imwrite(crop_path_obj, crop_img)
                    crop_path = str(crop_path_obj)

            db.execute(
                """
                INSERT INTO ocr_words
                (manuscript_id, word_text, confidence, x, y, width, height, crop_path, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (manuscript_id, word, confidence, x, y, w, h, crop_path, status),
            )

            if status == "uncertain":
                uncertain_count += 1
            word_count += 1

    update_pipeline_stage(manuscript_id, owner_id, "OCR", "ready")
    return word_count, uncertain_count


def analyze_uncertainty(
    manuscript_id: int,
    owner_id: int,
    limit: int = 3,
) -> int:
    """Stage 04 preparation: generate corpus evidence for low-confidence OCR."""
    update_pipeline_stage(manuscript_id, owner_id, "Uncertainty", "running")
    corpus = load_corpus()

    with connect() as db:
        rows = db.execute(
            """
            SELECT ow.id, ow.word_text
            FROM ocr_words ow
            JOIN manuscripts m ON m.id = ow.manuscript_id
            WHERE ow.manuscript_id = ? AND m.owner_id = ? AND ow.status = 'uncertain'
            ORDER BY ow.id
            """,
            (manuscript_id, owner_id),
        ).fetchall()

        db.execute(
            """
            DELETE FROM suggestions
            WHERE ocr_word_id IN (
                SELECT id FROM ocr_words WHERE manuscript_id = ?
            )
            """,
            (manuscript_id,),
        )

        for row in rows:
            matches = extract_best_suggestions(str(row["word_text"]), corpus, limit=limit)
            for rank, (candidate, score) in enumerate(matches, start=1):
                db.execute(
                    """
                    INSERT INTO suggestions
                    (ocr_word_id, suggested_text, match_score, corpus_name, rank)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (int(row["id"]), candidate, float(score), "Scholar Corpus", rank),
                )

    count = len(rows)
    update_pipeline_stage(
        manuscript_id,
        owner_id,
        "Review" if count else "Finalized",
        "needs_review" if count else "ready",
    )
    return count


def enhance_and_ocr(
    manuscript_id: int,
    threshold: float,
    owner_id: int,
    lang_code: str = "san+hin",
    psm: int = 6,
    preprocess_mode: str = "Adaptive Binary (B&W)",
    ensemble: bool = True,
) -> tuple[int, int]:
    """Backward-compatible convenience action: Restore → OCR → Uncertainty analysis."""
    restore_manuscript(
        manuscript_id,
        owner_id,
        preprocess_mode=preprocess_mode,
    )
    total, _ = run_ocr_stage(
        manuscript_id,
        owner_id,
        threshold,
        lang_code=lang_code,
        psm=psm,
        ensemble=ensemble,
    )
    uncertain = analyze_uncertainty(manuscript_id, owner_id)
    return total, uncertain


def finalize_manuscript(manuscript_id: int, owner_id: int) -> None:
    """Stage 05: finalize only after no uncertain OCR readings remain."""
    remaining = pending_review_count(manuscript_id, owner_id)
    if remaining:
        raise ValueError(f"{remaining} uncertain reading(s) still require review.")

    manuscript = get_manuscript(manuscript_id, owner_id)
    if manuscript is None:
        raise ValueError("Manuscript not found for this user.")

    update_pipeline_stage(manuscript_id, owner_id, "Finalized", "running")
    update_pipeline_stage(manuscript_id, owner_id, "Finalized", "ready")


def export_manuscript_package(manuscript_id: int, owner_id: int) -> Path:
    """Stage 06: create a compact scholarly export package."""
    manuscript = get_manuscript(manuscript_id, owner_id)
    if manuscript is None:
        raise ValueError("Manuscript not found for this user.")

    remaining = pending_review_count(manuscript_id, owner_id)
    if remaining:
        raise ValueError(f"Cannot export while {remaining} uncertain reading(s) remain.")

    finalize_manuscript(manuscript_id, owner_id)
    update_pipeline_stage(manuscript_id, owner_id, "Export", "running")

    export_dir = user_dir(APP_DIR / "exports", owner_id)
    export_dir.mkdir(parents=True, exist_ok=True)
    package_dir = export_dir / f"manuscript_{manuscript_id}_package"
    package_dir.mkdir(parents=True, exist_ok=True)

    text_path = package_dir / "transcription.txt"
    text_path.write_text(
        str(manuscript["extracted_full_text"] or ""),
        encoding="utf-8",
    )

    log_df = review_log(manuscript_id, owner_id)
    log_path = package_dir / "provenance.csv"
    log_df.to_csv(log_path, index=False, encoding="utf-8")

    metadata = {
        "manuscript_id": int(manuscript_id),
        "title": manuscript["title"],
        "collection": manuscript["collection_name"],
        "script": manuscript["script_name"],
        "original_filename": manuscript["original_filename"],
        "original_path": manuscript["original_path"],
        "enhanced_path": manuscript["enhanced_path"],
    }
    (package_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    archive = shutil.make_archive(str(package_dir), "zip", root_dir=package_dir)
    update_pipeline_stage(manuscript_id, owner_id, "Export", "ready")
    return Path(archive)


def run_full_pipeline(
    manuscript_id: int,
    owner_id: int,
    threshold: float,
    lang_code: str,
    psm: int,
    preprocess_mode: str,
    ensemble: bool = True,
) -> dict[str, Any]:
    """
    Execute the real pipeline in sequence.

    The pipeline intentionally stops at human review when uncertain readings
    exist. It never silently chooses scholarly readings on behalf of the reviewer.
    """
    try:
        update_pipeline_stage(manuscript_id, owner_id, "Preserved", "ready")

        enhanced_path = restore_manuscript(
            manuscript_id, owner_id, preprocess_mode=preprocess_mode
        )

        total, _ = run_ocr_stage(
            manuscript_id,
            owner_id,
            threshold,
            lang_code=lang_code,
            psm=psm,
            ensemble=ensemble,
        )

        uncertain = analyze_uncertainty(manuscript_id, owner_id)

        if uncertain:
            return {
                "stage": "Review",
                "status": "needs_review",
                "total": total,
                "uncertain": uncertain,
                "enhanced_path": enhanced_path,
                "message": "Pipeline paused for scholarly review.",
            }

        finalize_manuscript(manuscript_id, owner_id)

        return {
            "stage": "Finalized",
            "status": "ready",
            "total": total,
            "uncertain": 0,
            "enhanced_path": enhanced_path,
            "message": "Pipeline reached finalization.",
        }

    except Exception:
        try:
            update_pipeline_stage(manuscript_id, owner_id, "Pipeline", "failed")
        except Exception:
            pass
        raise


def ocr_items(manuscript_id: int, owner_id: int, status: str | None = None) -> list[sqlite3.Row]:
    query = """
        SELECT ow.*
        FROM ocr_words ow
        JOIN manuscripts m ON m.id = ow.manuscript_id
        WHERE ow.manuscript_id = ? AND m.owner_id = ?
    """
    params: list[Any] = [manuscript_id, owner_id]

    if status:
        query += " AND ow.status = ?"
        params.append(status)

    query += " ORDER BY ow.id"
    with connect() as db:
        return db.execute(query, params).fetchall()


def suggestions_for(word_id: int, owner_id: int) -> list[sqlite3.Row]:
    with connect() as db:
        return db.execute(
            """
            SELECT s.*
            FROM suggestions s
            JOIN ocr_words ow ON ow.id = s.ocr_word_id
            JOIN manuscripts m ON m.id = ow.manuscript_id
            WHERE s.ocr_word_id = ? AND m.owner_id = ?
            ORDER BY s.rank
            """,
            (word_id, owner_id),
        ).fetchall()


def record_review(
    manuscript_id: int,
    word_id: int,
    action: str,
    final_text: str | None,
    suggestion_id: int | None,
    reason: str,
    reviewer: str,
    owner_id: int,
) -> None:
    with connect() as db:
        word = db.execute(
            """
            SELECT ow.word_text
            FROM ocr_words ow
            JOIN manuscripts m ON m.id = ow.manuscript_id
            WHERE ow.id = ? AND ow.manuscript_id = ? AND m.owner_id = ?
            """,
            (word_id, manuscript_id, owner_id),
        ).fetchone()

        if word is None:
            raise ValueError("OCR word record was not found.")

        db.execute(
            """
            INSERT INTO review_log
            (manuscript_id, ocr_word_id, suggestion_id, action, previous_text, final_text, reviewer, reason, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (manuscript_id, word_id, suggestion_id, action, word["word_text"], final_text, reviewer, reason, now()),
        )

        if action == "accept":
            db.execute("UPDATE ocr_words SET status = 'accepted' WHERE id = ? AND manuscript_id = ?", (word_id, manuscript_id))
            db.execute("UPDATE suggestions SET status = 'rejected' WHERE ocr_word_id = ?", (word_id,))
            if suggestion_id:
                db.execute("UPDATE suggestions SET status = 'accepted' WHERE id = ?", (suggestion_id,))

        elif action == "reject":
            db.execute("UPDATE ocr_words SET status = 'rejected' WHERE id = ? AND manuscript_id = ?", (word_id, manuscript_id))
            if suggestion_id:
                db.execute("UPDATE suggestions SET status = 'rejected' WHERE id = ?", (suggestion_id,))

        elif action == "manual_edit":
            db.execute(
                "UPDATE ocr_words SET status = 'manual_edit', word_text = ? WHERE id = ? AND manuscript_id = ?",
                (final_text, word_id, manuscript_id),
            )

    remaining = pending_review_count(manuscript_id, owner_id)
    if remaining == 0:
        update_pipeline_stage(manuscript_id, owner_id, "Finalized", "ready")
    else:
        update_pipeline_stage(manuscript_id, owner_id, "Review", "needs_review")


def review_log(manuscript_id: int, owner_id: int) -> pd.DataFrame:
    with connect() as db:
        rows = db.execute(
            """
            SELECT
                rl.created_at AS Time,
                rl.action AS Action,
                rl.previous_text AS 'Original OCR',
                rl.final_text AS 'Final value',
                rl.reviewer AS Reviewer,
                rl.reason AS Reason
            FROM review_log rl
            JOIN manuscripts m ON m.id = rl.manuscript_id
            WHERE rl.manuscript_id = ? AND m.owner_id = ?
            ORDER BY rl.id DESC
            """,
            (manuscript_id, owner_id),
        ).fetchall()
    return pd.DataFrame([dict(row) for row in rows])


# ============================================================
# UI HELPERS
# ============================================================

def display_image(path: str | None, caption: str) -> None:
    if path and Path(path).exists():
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.info(f"{caption} is not ready yet.")


def asset_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def page_header(eyebrow: str, title: str, description: str = "") -> None:
    st.markdown(
        f"""
        <div class="vn-page-header">
            <div class="vn-eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            {"<p>" + description + "</p>" if description else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_state(title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="vn-empty">
            <div class="vn-empty-icon">⌁</div>
            <h3>{title}</h3>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# BRANDING & GLOBAL STYLES
# ============================================================

def apply_branding(is_authenticated: bool = False, show_top_brand: bool = True) -> None:
    logo_uri = asset_data_uri(LOGO_PATH)
    background_uri = asset_data_uri(BACKGROUND_PATH)

    if not is_authenticated:
        if background_uri:
            bg_css = f"""
            background-image: linear-gradient(rgba(8, 4, 1, 0.45), rgba(8, 4, 1, 0.45)), url("{background_uri}");
            background-size: cover;
            background-position: center center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-color: #100803 !important;
            """
        else:
            bg_css = "background: linear-gradient(135deg, #120b05, #2b1809) !important;"
    else:
        bg_css = """
        background: radial-gradient(circle at top right, #1a140d 0%, #100d09 50%, #080605 100%) !important;
        """

    st.markdown(
        f"""
<style>
html, body {{
    margin: 0;
    padding: 0;
    background: transparent !important;
}}
.stApp {{
    min-height: 100vh;
    {bg_css}
    color: #f3dfb3;
}}

[data-testid="stHeader"] {{
    background: transparent !important;
    height: 40px !important;
    min-height: 40px !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 999999 !important;
    box-shadow: none !important;
    border: 0 !important;
}}
[data-testid="stHeader"] > div {{
    background: transparent !important;
    box-shadow: none !important;
}}
[data-testid="stHeader"] button {{
    visibility: visible !important;
    opacity: 1 !important;
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
}}
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {{
    background: transparent !important;
    padding-top: 0rem !important;
}}
[data-testid="stDecoration"] {{
    display: none !important;
}}
[data-testid="stToolbar"] {{
    visibility: hidden !important;
}}


[data-testid="stSidebar"] {{
    z-index: 999998 !important;
}}
[data-testid="stSidebar"] button {{
    visibility: visible !important;
}}
.block-container {{
    max-width: 1100px;
    padding-top: 1.5rem !important;
    padding-bottom: 3rem;
    background: transparent !important;
}}

h1, h2, h3, h4 {{
    font-family: Georgia, "Times New Roman", serif !important;
    color: #f7dca0 !important;
    text-shadow: 0 2px 6px rgba(0, 0, 0, 0.95);
}}
p, label, .stMarkdown {{
    color: #ead3a5;
}}

.vn-top-brand {{
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 10px 0 16px 0;
    margin-bottom: 24px;
    border-bottom: 1px solid rgba(204, 147, 48, 0.75);
    text-shadow: 0 2px 5px rgba(0, 0, 0, 0.95);
}}
.vn-top-brand img {{
    width: 56px;
    height: 56px;
    object-fit: contain;
    flex-shrink: 0;
    border-radius: 6px;
    border: 1px solid #d49a32;
    background: rgba(18, 11, 5, 0.7);
    padding: 3px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.85);
}}
.vn-top-name {{
    font-family: Georgia, serif;
    font-size: 1.45rem;
    font-weight: 700;
    letter-spacing: 1.4px;
    color: #f4cf7c;
    text-shadow: 0 2px 6px rgba(0, 0, 0, 1);
}}
.vn-top-subtitle {{
    font-family: Georgia, serif;
    font-size: .8rem;
    color: #d8bc83;
}}

.vn-auth-wrapper {{
    margin-top: 7vh;
    margin-bottom: 7vh;
}}

.vn-landing-nav {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
    padding: 4px 0 18px;
    margin-bottom: 5vh;
    border-bottom: 1px solid rgba(204, 147, 48, 0.6);
}}
.vn-landing-mark {{
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: Georgia, serif;
    color: #f4cf7c;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    white-space: nowrap;
}}
.vn-landing-mark img {{
    width: 54px;
    height: 54px;
    object-fit: cover;
    object-position: center 42%;
    border-radius: 6px;
    border: 1px solid rgba(212, 154, 50, .8);
    background: #130b05;
    box-shadow: 0 4px 14px rgba(0, 0, 0, .75);
}}
.vn-hero {{
    min-height: 54vh;
    display: flex;
    align-items: center;
    justify-content: flex-start;
    text-align: left;
    padding: 24px 7% 48px;
}}
.vn-hero-inner {{
    max-width: 620px;
}}
.vn-hero-kicker {{
    color: #d7a84d;
    font-size: .7rem;
    font-weight: 800;
    letter-spacing: 2.6px;
    text-transform: uppercase;
    margin-bottom: 22px;
}}
.vn-hero-quote {{
    margin: 0;
    color: #f6dda6;
    font-family: Georgia, "Times New Roman", serif;
    font-size: clamp(1.8rem, 3vw, 2.9rem);
    line-height: 1.22;
    font-weight: 600;
    text-shadow: 0 3px 14px rgba(0, 0, 0, .9);
}}
.vn-hero-rule {{
    width: 70px;
    height: 1px;
    background: #d39a32;
    margin: 28px 0 18px;
}}
.vn-hero-attribution {{
    color: #d1b883;
    font-family: Georgia, serif;
    font-size: .88rem;
    letter-spacing: 1px;
}}
.vn-landing-section {{
    padding: 55px 0 16px;
    scroll-margin-top: 20px;
}}
.vn-feature-grid {{
    margin-top: 8px;
}}
.vn-section-heading {{
    color: #f4d18a;
    font-family: Georgia, serif;
    font-size: 2rem;
    margin: 0 0 10px;
}}
.vn-section-intro {{
    max-width: 700px;
    color: #d8c092;
    font-family: Georgia, serif;
    line-height: 1.65;
    margin: 0 0 22px;
}}
.vn-feature-card {{
    min-height: 156px;
    padding: 21px;
    background: rgba(26, 18, 10, .88);
    border: 1px solid rgba(190, 137, 45, .62);
    box-shadow: 0 10px 30px rgba(0, 0, 0, .45);
}}
.vn-feature-num {{
    color: #d39a32;
    font-size: .68rem;
    font-weight: 800;
    letter-spacing: 1.4px;
}}
.vn-feature-title {{
    color: #f0cf86;
    font-family: Georgia, serif;
    font-size: 1.12rem;
    font-weight: 700;
    margin: 10px 0 8px;
}}
.vn-feature-copy {{
    color: #cdb98e;
    font-family: Georgia, serif;
    font-size: .9rem;
    line-height: 1.55;
}}
.vn-contact-panel {{
    padding: 27px;
    background: rgba(26, 18, 10, .9);
    border: 1px solid rgba(190, 137, 45, .7);
    text-align: center;
}}
@media (max-width: 700px) {{
    .vn-hero {{
        min-height: 46vh;
        padding-left: 5%;
        padding-right: 5%;
    }}
}}
.vn-landing-nav + div [data-testid="stButton"] button {{
    min-height: 36px;
}}

.vn-pipeline-side-heading {{
    margin-bottom: 10px;
}}
.vn-pipeline-compact {{
    padding: 14px 12px 16px !important;
    margin: 0 !important;
}}
.vn-pipeline-compact .vn-pipeline-diagram-heading {{
    display: block !important;
    padding-bottom: 11px !important;
}}
.vn-pipeline-compact .vn-pipeline-diagram-title {{
    font-size: 1rem !important;
    line-height: 1.2;
}}
.vn-pipeline-compact .vn-pipeline-diagram-subtitle {{
    display: none;
}}
.vn-pipeline-compact .vn-pipeline-live-status {{
    margin-top: 7px;
    font-size: .58rem;
}}
.vn-pipeline-compact .vn-pipeline-flow {{
    width: 100% !important;
    margin-top: 13px !important;
}}
.vn-pipeline-compact .vn-pipeline-node {{
    gap: 7px !important;
}}
.vn-pipeline-compact .vn-pipeline-node-marker {{
    width: 21px !important;
    min-width: 21px !important;
    font-size: .62rem !important;
}}
.vn-pipeline-compact .vn-pipeline-node-marker span {{
    width: 20px !important;
    height: 20px !important;
}}
.vn-pipeline-compact .vn-pipeline-node-card {{
    padding: 9px 10px !important;
}}
.vn-pipeline-compact .vn-pipeline-node-top {{
    gap: 5px !important;
}}
.vn-pipeline-compact .vn-pipeline-node-number {{
    font-size: .54rem !important;
}}
.vn-pipeline-compact .vn-pipeline-node-label {{
    font-size: .50rem !important;
    letter-spacing: 1px !important;
}}
.vn-pipeline-compact .vn-pipeline-node-state {{
    font-size: .48rem !important;
}}
.vn-pipeline-compact .vn-pipeline-node-title {{
    margin-top: 4px !important;
    font-size: .77rem !important;
}}
.vn-pipeline-compact .vn-pipeline-node-copy {{
    margin-top: 2px !important;
    font-size: .55rem !important;
    line-height: 1.25 !important;
}}
.vn-pipeline-compact .vn-pipeline-connector {{
    height: 18px !important;
    width: 21px !important;
    font-size: .34rem !important;
}}
.vn-pipeline-compact .vn-pipeline-connector::before {{
    height: 18px !important;
}}
.vn-pipeline-compact .vn-pipeline-connector span {{
    padding: 2px !important;
}}
@media (max-width: 900px) {{
    .vn-pipeline-compact .vn-pipeline-node-copy {{
        display: none !important;
    }}
}}

.vn-pipeline-diagram {{
    margin: 8px 0 18px;
    padding: 28px 28px 34px;
    background:
        radial-gradient(circle at 50% 0%, rgba(176, 124, 45, .10), transparent 40%),
        linear-gradient(145deg, rgba(30, 20, 11, .97), rgba(15, 9, 5, .98));
    border: 1px solid rgba(190, 137, 45, .58);
    box-shadow: 0 16px 40px rgba(0, 0, 0, .42), inset 0 1px 0 rgba(255,255,255,.025);
}}
.vn-pipeline-diagram-heading {{
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    gap: 24px;
    padding-bottom: 22px;
    border-bottom: 1px solid rgba(190, 137, 45, .24);
}}
.vn-pipeline-diagram-title {{
    margin-top: 5px;
    color: #f0d69d;
    font-family: Georgia, "Times New Roman", serif;
    font-size: 1.55rem;
}}
.vn-pipeline-diagram-subtitle {{
    max-width: 650px;
    margin-top: 6px;
    color: #a99674;
    font-family: Georgia, serif;
    font-size: .78rem;
    line-height: 1.55;
}}
.vn-pipeline-live-status {{
    white-space: nowrap;
    color: #caa45f;
    font-family: Georgia, serif;
    font-size: .68rem;
    font-weight: 700;
    letter-spacing: 1px;
}}
.vn-pipeline-live-dot {{
    display: inline-block;
    width: 7px;
    height: 7px;
    margin-right: 7px;
    border-radius: 50%;
    background: #c9953d;
    box-shadow: 0 0 9px rgba(201,149,61,.7);
}}
.vn-pipeline-flow {{
    width: min(760px, 100%);
    margin: 28px auto 0;
}}
.vn-pipeline-node {{
    display: flex;
    align-items: stretch;
    gap: 16px;
    position: relative;
}}
.vn-pipeline-node-marker {{
    width: 34px;
    min-width: 34px;
    display: flex;
    justify-content: center;
    align-items: center;
    color: #6e604c;
    font-family: Georgia, serif;
    font-size: .9rem;
}}
.vn-pipeline-node-marker span {{
    width: 27px;
    height: 27px;
    display: flex;
    justify-content: center;
    align-items: center;
    border: 1px solid #5d4b31;
    border-radius: 50%;
    background: #120b06;
}}
.vn-pipeline-node-card {{
    flex: 1;
    padding: 17px 20px;
    border: 1px solid rgba(120, 91, 50, .45);
    background: rgba(25, 16, 8, .84);
    transition: border-color .2s ease, transform .2s ease;
}}
.vn-pipeline-node-top {{
    display: flex;
    align-items: center;
    gap: 10px;
}}
.vn-pipeline-node-number {{
    color: #987543;
    font-family: Georgia, serif;
    font-size: .68rem;
    font-weight: 700;
}}
.vn-pipeline-node-label {{
    color: #a68a5b;
    font-family: Georgia, serif;
    font-size: .67rem;
    font-weight: 700;
    letter-spacing: 1.6px;
}}
.vn-pipeline-node-state {{
    margin-left: auto;
    color: #655843;
    font-family: Georgia, serif;
    font-size: .61rem;
    letter-spacing: 1px;
}}
.vn-pipeline-node-title {{
    margin-top: 7px;
    color: #e7d09c;
    font-family: Georgia, "Times New Roman", serif;
    font-size: 1.08rem;
    font-weight: 700;
}}
.vn-pipeline-node-copy {{
    margin-top: 4px;
    color: #a9987a;
    font-family: Georgia, serif;
    font-size: .76rem;
    line-height: 1.5;
}}
.vn-pipeline-node.completed .vn-pipeline-node-marker {{
    color: #d0a35a;
}}
.vn-pipeline-node.completed .vn-pipeline-node-marker span {{
    border-color: #a97a34;
    background: rgba(115, 77, 25, .28);
}}
.vn-pipeline-node.completed .vn-pipeline-node-card {{
    border-color: rgba(169, 122, 52, .48);
}}
.vn-pipeline-node.current .vn-pipeline-node-marker,
.vn-pipeline-node.running .vn-pipeline-node-marker {{
    color: #e0b767;
}}
.vn-pipeline-node.current .vn-pipeline-node-marker span,
.vn-pipeline-node.running .vn-pipeline-node-marker span {{
    border-color: #d29a3c;
    background: rgba(137, 91, 25, .30);
    box-shadow: 0 0 18px rgba(196, 139, 48, .18);
}}
.vn-pipeline-node.current .vn-pipeline-node-card,
.vn-pipeline-node.running .vn-pipeline-node-card {{
    border-color: rgba(211, 155, 58, .8);
    box-shadow: inset 3px 0 0 #b47c2d;
}}
.vn-pipeline-node.current .vn-pipeline-node-state,
.vn-pipeline-node.running .vn-pipeline-node-state {{
    color: #d2a557;
}}
.vn-pipeline-connector {{
    width: 34px;
    height: 31px;
    display: flex;
    justify-content: center;
    align-items: center;
    color: #4d3e2b;
    font-size: .48rem;
}}
.vn-pipeline-connector::before {{
    content: "";
    position: absolute;
    width: 1px;
    height: 31px;
    background: #493a28;
}}
.vn-pipeline-connector span {{
    position: relative;
    z-index: 1;
    padding: 3px;
    background: #171006;
}}
.vn-pipeline-connector.done {{
    color: #a97831;
}}
.vn-pipeline-connector.done::before {{
    background: #8a6029;
}}
@media (max-width: 720px) {{
    .vn-pipeline-diagram {{
        padding: 22px 16px 26px;
    }}
    .vn-pipeline-diagram-heading {{
        display: block;
    }}
    .vn-pipeline-live-status {{
        margin-top: 14px;
    }}
    .vn-pipeline-node {{
        gap: 9px;
    }}
    .vn-pipeline-node-marker {{
        width: 27px;
        min-width: 27px;
    }}
    .vn-pipeline-node-marker span {{
        width: 23px;
        height: 23px;
    }}
    .vn-pipeline-node-card {{
        padding: 14px;
    }}
    .vn-pipeline-node-state {{
        display: none;
    }}
}}

.vn-pipeline-card {{
    margin: 0 0 12px;
    padding: 18px 20px;
    background: rgba(26, 18, 10, .9);
    border: 1px solid rgba(190, 137, 45, .5);
}}
.vn-pipeline-head {{
    display: flex;
    justify-content: space-between;
    gap: 18px;
    align-items: center;
}}
.vn-pipeline-percent {{
    color: #e3bc67;
    font-family: Georgia, serif;
    font-weight: 700;
}}
.vn-pipeline-track {{
    height: 5px;
    margin: 13px 0 12px;
    background: #2a1b0e;
    border-radius: 8px;
    overflow: hidden;
}}
.vn-pipeline-fill {{
    height: 100%;
    background: linear-gradient(90deg, #8f6124, #d3a044);
}}
.vn-pipeline-steps {{
    display: flex;
    justify-content: space-between;
    gap: 8px;
    flex-wrap: wrap;
}}
.vn-pipeline-step {{
    font-family: Georgia, serif;
    font-size: .73rem;
    color: #7f7059;
}}
.vn-pipeline-step.done, .vn-pipeline-step.active {{
    color: #d5aa5b;
}}
.vn-pipeline-step.active {{
    font-weight: 700;
}}
.vn-pipeline-meta {{
    margin-top: 10px;
    color: #9d8a6b;
    font-family: Georgia, serif;
    font-size: .74rem;
}}
@media (max-width: 800px) {{
    .vn-pipeline-steps {{
        display: grid;
        grid-template-columns: repeat(2, 1fr);
    }}
}}

.vn-public-page {{
    max-width: 900px;
    margin: 8vh auto 38px;
    padding: 45px 48px;
    text-align: center;
    background: rgba(26, 18, 10, .92);
    border: 1px solid rgba(190, 137, 45, .7);
    box-shadow: 0 18px 45px rgba(0, 0, 0, .55);
}}
.vn-public-eyebrow {{
    color: #d39a32;
    font-family: Georgia, serif;
    font-size: .72rem;
    font-weight: 800;
    letter-spacing: 2px;
}}
.vn-public-page h1 {{
    margin: 12px 0 0;
    color: #f4d99d !important;
    font-family: Georgia, "Times New Roman", serif;
    font-size: clamp(2.1rem, 4vw, 3.6rem);
    line-height: 1.12;
}}
.vn-public-rule {{
    width: 90px;
    height: 1px;
    margin: 24px auto;
    background: #d39a32;
}}
.vn-public-intro {{
    max-width: 720px;
    margin: 0 auto;
    color: #d8c092;
    font-family: Georgia, serif;
    font-size: 1.02rem;
    line-height: 1.75;
}}
.vn-public-card {{
    min-height: 170px;
    margin-bottom: 18px;
    padding: 26px;
    background: rgba(26, 18, 10, .92);
    border: 1px solid rgba(190, 137, 45, .62);
    box-shadow: 0 10px 30px rgba(0, 0, 0, .4);
}}
.vn-public-card-num {{
    color: #d39a32;
    font-family: Georgia, serif;
    font-size: .7rem;
    font-weight: 800;
    letter-spacing: 1.5px;
}}
.vn-public-card h3, .vn-doc-step h3 {{
    color: #f0cf86;
    font-family: Georgia, serif;
    margin: 10px 0 8px;
}}
.vn-public-card p, .vn-doc-step p {{
    color: #cdb98e;
    font-family: Georgia, serif;
    line-height: 1.65;
    margin: 0;
}}
.vn-doc-step {{
    display: flex;
    gap: 22px;
    align-items: flex-start;
    margin: 0 auto 14px;
    padding: 22px 25px;
    max-width: 900px;
    background: rgba(26, 18, 10, .92);
    border: 1px solid rgba(190, 137, 45, .5);
}}
.vn-doc-step > span {{
    color: #d39a32;
    font-family: Georgia, serif;
    font-weight: 800;
    padding-top: 3px;
}}
.vn-contact-card {{
    text-align: center;
}}
@media (max-width: 700px) {{
    .vn-public-page {{
        padding: 30px 22px;
        margin-top: 5vh;
    }}
}}
.vn-auth-card, .vn-card, .vn-review-card, .vn-empty, .vn-metric {{
    background: rgba(26, 18, 10, 0.92) !important;
    border: 1px solid rgba(190, 137, 45, 0.75) !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.7);
}}
.vn-auth-card {{
    padding: 24px 28px 18px;
    margin-bottom: 16px;
    border-radius: 4px;
}}
.vn-auth-heading {{
    text-align: center;
    font-family: Georgia, serif;
    font-size: 1.45rem;
    font-weight: 700;
    color: #f3d28c;
}}
.vn-auth-copy {{
    text-align: center;
    font-family: Georgia, serif;
    color: #cdb27d;
    font-size: .85rem;
    margin-top: 4px;
}}

div[data-baseweb="input"],
div[data-baseweb="input"] > div,
div[data-baseweb="base-input"],
div[data-baseweb="textarea"],
div[data-baseweb="textarea"] > div {{
    background-color: rgba(18, 10, 4, 0.96) !important;
    border: 1px solid #d49a32 !important;
    border-radius: 4px !important;
}}

input, 
textarea {{
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-family: Georgia, serif !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
}}

input::placeholder,
textarea::placeholder {{
    color: #c0ab89 !important;
    -webkit-text-fill-color: #c0ab89 !important;
}}

.vn-card {{
    padding: 22px;
    margin-bottom: 18px;
}}
.vn-card-title {{
    font-family: Georgia, serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #f0cf86;
}}
.vn-card-copy {{
    color: #d6bd8a;
    font-family: Georgia, serif;
    line-height: 1.55;
}}
.vn-page-header {{
    margin: 10px 0 25px;
    padding-bottom: 15px;
    border-bottom: 1px solid rgba(190, 137, 45, 0.75);
}}
.vn-eyebrow {{
    color: #d39a32;
    font-family: Georgia, serif;
    font-size: .7rem;
    font-weight: 800;
    letter-spacing: 1.7px;
    text-transform: uppercase;
}}
.vn-metric {{
    padding: 18px;
    min-height: 105px;
}}
.vn-metric-label {{
    color: #c4a66f;
    font-size: .7rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 700;
}}
.vn-metric-value {{
    color: #f2d087;
    font-family: Georgia, serif;
    font-size: 2rem;
    font-weight: 700;
    margin-top: 10px;
}}
.vn-manuscript-row {{
    display: flex;
    justify-content: space-between;
    gap: 16px;
    padding: 15px 4px;
    border-bottom: 1px solid rgba(190, 137, 45, 0.35);
}}
.vn-manuscript-title {{
    font-family: Georgia, serif;
    font-weight: 700;
    color: #edd095;
}}
.vn-manuscript-meta {{
    color: #c0a678;
    font-family: Georgia, serif;
    font-size: .8rem;
    margin-top: 4px;
}}
.vn-badge {{
    display: inline-block;
    padding: 4px 8px;
    border: 1px solid rgba(190, 137, 45, 0.7);
    border-radius: 3px;
    font-family: Georgia, serif;
    font-size: .68rem;
    font-weight: 700;
}}
.vn-badge-green {{
    background: rgba(58, 78, 45, 0.85);
    color: #d9e9c8;
}}
.vn-badge-gold {{
    background: rgba(117, 79, 22, 0.9);
    color: #ffe0a0;
}}

.stButton > button {{
    background: linear-gradient(180deg, #70491d, #321b09) !important;
    color: #fff0c8 !important;
    border: 1px solid #b77b24 !important;
    border-radius: 4px !important;
    min-height: 40px;
    font-family: Georgia, serif !important;
    font-weight: 700 !important;
}}
.stButton > button:hover {{
    background: linear-gradient(180deg, #98662a, #4d2d11) !important;
    border-color: #e0a83a !important;
    color: #fff9e9 !important;
}}

.vn-review-card {{
    padding: 22px;
    margin-bottom: 20px;
}}
.vn-review-label, .vn-image-label {{
    color: #d2a04c;
    font-family: Georgia, serif;
    font-size: .68rem;
    font-weight: 800;
    letter-spacing: 1px;
    text-transform: uppercase;
}}
.vn-ocr-text {{
    font-family: Georgia, serif;
    font-size: 1.75rem;
    color: #f4d99d;
    margin: 7px 0;
}}
.vn-confidence {{
    color: #c8ad79;
    font-family: Georgia, serif;
    font-size: .8rem;
}}
.vn-empty {{
    text-align: center;
    padding: 40px 20px;
}}
.vn-empty-icon {{
    color: #d49a32;
    font-family: Georgia, serif;
    font-size: 2.2rem;
}}
</style>
""",
        unsafe_allow_html=True,
    )

    if show_top_brand:
        img_tag = f"<img src='{logo_uri}' alt='Logo' />" if logo_uri else ""
        st.markdown(
            f"""<div class="vn-top-brand">{img_tag}<div><div class="vn-top-name">VELLUM NODE</div><div class="vn-top-subtitle">Manuscript restoration &amp; scholarly review</div></div></div>""",
            unsafe_allow_html=True,
        )


# ============================================================
# APP WORKSPACE
# ============================================================


def render_pipeline_diagram(manuscript_id: int, owner_id: int, compact: bool = False) -> None:
    """Render the manuscript workflow as a vertical scholarly pipeline."""
    state = get_pipeline_state(manuscript_id, owner_id)
    current = state["current_stage"] if state else "Preserved"
    status = state["status"] if state else "ready"

    stage_data = [
        ("Preserved", "01", "PRESERVE", "Original Manuscript",
         "Upload, metadata, and immutable source evidence"),
        ("Restore", "02", "RESTORE", "Image Restoration",
         "Deskew, enhance, binarize, denoise, and prepare the working image"),
        ("OCR", "03", "OCR", "Text Extraction",
         "Multilingual OCR, segmentation, bounding boxes, and confidence scoring"),
        ("Uncertainty", "04", "ANALYZE", "Uncertainty Analysis",
         "Detect low-confidence readings and generate corpus evidence"),
        ("Review", "05", "REVIEW", "Scholarly Review",
         "Human verification of uncertain readings and editorial decisions"),
        ("Finalized", "06", "FINALIZE", "Scholarly Approval",
         "Finalize the reviewed transcription without overwriting source evidence"),
        ("Export", "07", "EXPORT", "Provenance & Export",
         "Package transcription, metadata, and the review history"),
    ]

    display_current = current

    order = {name: i for i, item in enumerate(stage_data) for name in [item[0]]}
    current_index = order.get(display_current, 0)

    cards = []
    for index, (stage_key, number, label, title, copy) in enumerate(stage_data):
        if index < current_index:
            cls, icon, state_text = "completed", "✓", "COMPLETED"
        elif index == current_index:
            cls = "running" if status == "running" else "current"
            icon = "●" if status == "running" else "◆"
            state_text = "RUNNING" if status == "running" else "CURRENT STAGE"
        else:
            cls, icon, state_text = "upcoming", "○", "UP NEXT"

        cards.append(
            textwrap.dedent(
                f"""
                <div class="vn-pipeline-node {cls}">
                    <div class="vn-pipeline-node-marker">
                        <span>{icon}</span>
                    </div>
                    <div class="vn-pipeline-node-card">
                        <div class="vn-pipeline-node-top">
                            <span class="vn-pipeline-node-number">{number}</span>
                            <span class="vn-pipeline-node-label">{label}</span>
                            <span class="vn-pipeline-node-state">{state_text}</span>
                        </div>
                        <div class="vn-pipeline-node-title">{title}</div>
                        <div class="vn-pipeline-node-copy">{copy}</div>
                    </div>
                </div>
                """
            ).strip()
        )

        if index < len(stage_data) - 1:
            connector_cls = "done" if index < current_index else ""
            cards.append(
                f'<div class="vn-pipeline-connector {connector_cls}"><span>◆</span></div>'
            )

    shell = textwrap.dedent(
        f"""
        <div class="vn-pipeline-diagram{" vn-pipeline-compact" if compact else ""}>
            <div class="vn-pipeline-diagram-heading">
                <div>
                    <div class="vn-eyebrow">MANUSCRIPT PIPELINE</div>
                    <div class="vn-pipeline-diagram-title">From Evidence to Scholarly Record</div>
                    <div class="vn-pipeline-diagram-subtitle">
                        A traceable sequence for preservation, restoration, transcription, review, and provenance.
                    </div>
                </div>
                <div class="vn-pipeline-live-status">
                    <span class="vn-pipeline-live-dot"></span>
                    {display_current.upper()} · {status.replace("_", " ").upper()}
                </div>
            </div>

            <div class="vn-pipeline-flow">
                __PIPELINE_CARDS__
            </div>
        </div>
        """
    ).strip()

    # Insert dynamic cards only AFTER dedenting the static shell. This is the
    # critical fix that prevents Streamlit Markdown from rendering raw HTML.
    pipeline_html = shell.replace(
        "__PIPELINE_CARDS__",
        "".join(cards),
    )

    # st.html renders HTML as HTML instead of allowing Markdown to treat
    # nested div blocks as code. This prevents the raw <div ...> text issue.
    try:
        st.html(pipeline_html)
    except AttributeError:
        st.markdown(pipeline_html, unsafe_allow_html=True)


def render_dashboard(user: sqlite3.Row, owned: list[sqlite3.Row]) -> None:
    owner_id = int(user["id"])

    with connect() as db:
        review_count = db.execute(
            """
            SELECT COUNT(*) AS n
            FROM review_log rl
            JOIN manuscripts m ON m.id = rl.manuscript_id
            WHERE m.owner_id = ?
            """,
            (owner_id,),
        ).fetchone()["n"]

        uncertain_count = db.execute(
            """
            SELECT COUNT(*) AS n
            FROM ocr_words ow
            JOIN manuscripts m ON m.id = ow.manuscript_id
            WHERE m.owner_id = ? AND ow.status = 'uncertain'
            """,
            (owner_id,),
        ).fetchone()["n"]

        processed_count = db.execute(
            """
            SELECT COUNT(*) AS n
            FROM manuscripts
            WHERE owner_id = ? AND enhanced_path IS NOT NULL
            """,
            (owner_id,),
        ).fetchone()["n"]

    page_header(
        "Scholar workspace",
        f"Welcome, {user['display_name']}",
        "Preserve. Process. Review. Document. A complete workspace for rigorous manuscript restoration and evidence-based decision-making.",
    )

    metrics = [
        ("Manuscripts", len(owned)),
        ("Processed", processed_count),
        ("Needs review", uncertain_count),
        ("Review actions", review_count),
    ]

    metric_columns = st.columns(4)
    for column, (label, value) in zip(metric_columns, metrics):
        with column:
            st.markdown(
                f"""
                <div class="vn-metric">
                    <div class="vn-metric-label">{label}</div>
                    <div class="vn-metric-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------
    # MAIN WORK AREA FIRST
    # --------------------------------------------------------
    st.write("")
    left, right = st.columns([1.55, 1], gap="large")

    with left:
        st.markdown(
            """
            <div class="vn-card">
                <div class="vn-card-title">Recent manuscripts</div>
                <div class="vn-card-copy">Your most recently added records.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if owned:
            for row in owned[:6]:
                created = row["created_at"][:10]
                state = "Processed" if row["enhanced_path"] else "Original preserved"
                badge_kind = "green" if row["enhanced_path"] else "gold"

                st.markdown(
                    f"""
                    <div class="vn-manuscript-row">
                        <div>
                            <div class="vn-manuscript-title">{row["title"]}</div>
                            <div class="vn-manuscript-meta">
                                {row["collection_name"]} · {row["script_name"]} · {created}
                            </div>
                        </div>
                        <span class="vn-badge vn-badge-{badge_kind}">{state}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            empty_state(
                "No manuscripts yet",
                "Upload your first manuscript to begin the restoration workflow.",
            )

    with right:
        st.markdown(
            """
            <div class="vn-card" style="margin-bottom: 12px;">
                <div class="vn-card-title">Workflow Actions</div>
                <div class="vn-card-copy">Jump directly to a workspace stage:</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("01 · Upload Original", use_container_width=True, key="wf_upload"):
            st.session_state["active_workspace_tab"] = "Upload"
            st.rerun()

        if st.button("02 · Enhance & Run OCR", use_container_width=True, key="wf_process"):
            st.session_state["active_workspace_tab"] = "Process"
            st.rerun()

        if st.button("03 · Review Suggestions", use_container_width=True, key="wf_review"):
            st.session_state["active_workspace_tab"] = "Review Suggestions"
            st.rerun()

        if st.button("04 · Inspect Provenance", use_container_width=True, key="wf_provenance"):
            st.session_state["active_workspace_tab"] = "Provenance"
            st.rerun()

    # --------------------------------------------------------
    # PIPELINE BELOW THE WORK AREA
    # --------------------------------------------------------
    st.write("")
    st.markdown(
        """
        <div class="vn-card vn-pipeline-section-heading">
            <div class="vn-eyebrow">MANUSCRIPT PIPELINE</div>
            <div class="vn-card-title">Restoration flow</div>
            <div class="vn-card-copy">
                A compact trace of the manuscript from preserved evidence to scholarly record.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if owned:
        manuscript_options = {
            f"{row['id']} · {row['title']}": int(row['id'])
            for row in owned
        }
        selected_pipeline_label = st.selectbox(
            "Manuscript",
            list(manuscript_options.keys()),
            key="pipeline_manuscript_selector",
            label_visibility="collapsed",
        )
        manuscript_id = manuscript_options[selected_pipeline_label]

        render_pipeline_diagram(manuscript_id, owner_id, compact=True)

        pipeline_actions = st.columns(5)
        with pipeline_actions[0]:
            if st.button(
                "Run Full Pipeline",
                type="primary",
                key=f"pipe_run_{manuscript_id}",
                use_container_width=True,
            ):
                try:
                    with st.spinner("Running the manuscript pipeline..."):
                        result = run_full_pipeline(
                            manuscript_id,
                            owner_id,
                            threshold=75.0,
                            lang_code="san+hin",
                            psm=6,
                            preprocess_mode="Adaptive Binary (B&W)",
                        )
                    if result["uncertain"]:
                        st.warning(
                            f"Pipeline paused at Review: {result['uncertain']} "
                            "uncertain reading(s) require human verification."
                        )
                    else:
                        st.success("Pipeline reached finalization.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Pipeline failed: {exc}")

        with pipeline_actions[1]:
            if st.button("Open Upload", key=f"pipe_upload_{manuscript_id}", use_container_width=True):
                st.session_state["active_workspace_tab"] = "Upload"
                st.rerun()

        with pipeline_actions[2]:
            if st.button("Open Process", key=f"pipe_process_{manuscript_id}", use_container_width=True):
                st.session_state["active_workspace_tab"] = "Process"
                st.rerun()

        with pipeline_actions[3]:
            if st.button("Open Review", key=f"pipe_review_{manuscript_id}", use_container_width=True):
                st.session_state["active_workspace_tab"] = "Review Suggestions"
                st.rerun()

        with pipeline_actions[4]:
            if st.button("Open Trace", key=f"pipe_trace_{manuscript_id}", use_container_width=True):
                st.session_state["active_workspace_tab"] = "Provenance"
                st.rerun()
    else:
        st.info("Upload a manuscript to start the restoration pipeline.")


def render_upload(owner_id: int) -> None:
    page_header(
        "01 · Preserve",
        "Upload manuscript",
        "Add an original image to your private workspace. The original file is preserved separately and is never overwritten.",
    )

    left, right = st.columns([1, 1.15])
    with left:
        title = st.text_input("Manuscript title", placeholder="Example: Palm-leaf record A")
        collection = st.text_input("Collection", value="Local Collection")
        script = st.text_input("Script", value="Devanagari / Odia / Sanskrit")

    with right:
        uploaded = st.file_uploader(
            "Choose manuscript image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        if uploaded:
            st.image(uploaded, caption="Preview of the preserved original", use_container_width=True)

    st.write("")
    if st.button("Preserve original", type="primary", disabled=not (title.strip() and uploaded)):
        try:
            manuscript_id = save_upload(uploaded, title.strip(), collection.strip(), script.strip(), owner_id)
            st.success(f"Original preserved successfully. Manuscript ID: {manuscript_id}")
            st.session_state["active_workspace_tab"] = "Process"
            st.rerun()
        except Exception as exc:
            st.error(f"Could not save the manuscript: {exc}")


def render_process(owner_id: int, owned: list[sqlite3.Row]) -> None:
    page_header(
        "02 · High-Accuracy OCR & Enhance",
        "Extract text & flag uncertainties",
        "Advanced deskewing, binarization, and Indic morphological matching.",
    )

    if not owned:
        empty_state("No manuscript available", "Upload a manuscript first. Your original evidence will remain untouched.")
        return

    labels = {f"{row['id']} · {row['title']}": row["id"] for row in owned}
    selected_label = st.selectbox("Select Manuscript", list(labels))
    selected_id = labels[selected_label]

    lang_map = get_installed_tesseract_languages()
    display_names = list(lang_map.keys())

    default_selected = []
    for candidate in ["Sanskrit (संस्कृतम्)", "Hindi (हिन्दी)", "Odia (ଓଡ଼ିଆ)", "English"]:
        if candidate in display_names and candidate not in default_selected:
            default_selected.append(candidate)
            break
    if not default_selected and display_names:
        default_selected = [display_names[0]]

    # Prefer the Tesseract model matching the manuscript's declared script.
    current_manuscript = get_manuscript(selected_id, owner_id)
    preferred_code = language_code_for_script(
        current_manuscript["script_name"] if current_manuscript else "",
        set(lang_map.values()),
    )

    if preferred_code:
        preferred_labels = [
            label for label, code in lang_map.items() if code == preferred_code
        ]
        default_selected = preferred_labels[:1] or default_selected

    col1, col2, col3 = st.columns([1.3, 1, 0.9])
    with col1:
        chosen_labels = st.multiselect(
            "OCR Language Models",
            options=display_names,
            default=default_selected,
            help="Select one or more languages for simultaneous multi-script inference.",
        )
        active_codes = [lang_map[label] for label in chosen_labels] if chosen_labels else ["san", "hin"]
        final_lang_str = "+".join(active_codes)

    with col2:
        preprocess_mode = st.selectbox(
            "Preprocessing / Filter Mode",
            [
                "Adaptive Binary (B&W)",
                "Grayscale Contrast (CLAHE)",
                "Bilateral Smooth",
                "Otsu Binary",
            ],
            help="Switch preprocessing views to preserve faint ink gradations or clean up text.",
        )
    with col3:
        threshold = st.slider("Uncertainty Threshold (%)", 0, 100, 40)

    psm_choice = st.selectbox(
        "Segmentation Mode (PSM)",
        [6, 4, 3, 11],
        index=0,
        format_func=lambda x: {
            6: "PSM 6 · Single Uniform Block (Recommended)",
            4: "PSM 4 · Single Column Variable Size",
            3: "PSM 3 · Fully Automatic",
            11: "PSM 11 · Sparse Scattered Words",
        }[x],
    )

    ensemble_ocr = st.checkbox(
        "High-Accuracy Ensemble OCR",
        value=True,
        help=(
            "Runs several image-processing views and, for multiple selected languages, "
            "separate language models. Readings are selected using confidence and cross-view agreement. "
            "This is slower than a single OCR pass."
        ),
    )

    run_col, pipeline_col = st.columns(2)

    with run_col:
        if st.button("Run High-Accuracy OCR", type="primary", use_container_width=True):
            try:
                with st.spinner(
    f"Running {'ensemble ' if ensemble_ocr else ''}OCR with {preprocess_mode}..."
):
                    total, uncertain = enhance_and_ocr(
                        selected_id,
                        threshold,
                        owner_id,
                        final_lang_str,
                        psm_choice,
                        preprocess_mode=preprocess_mode,
                        ensemble=ensemble_ocr,
                    )
                st.success(
                    f"Processing complete: {total} words detected "
                    f"({uncertain} uncertain)."
                )
            except Exception as exc:
                st.error(f"Processing failed: {exc}")

    with pipeline_col:
        if st.button("Run Full Pipeline", use_container_width=True):
            try:
                with st.spinner("Running Preserve → Restore → OCR → Analysis..."):
                    result = run_full_pipeline(
                        selected_id,
                        owner_id,
                        threshold,
                        final_lang_str,
                        psm_choice,
                        preprocess_mode,
                        ensemble_ocr,
                    )
                if result["uncertain"]:
                    st.warning(
                        f"Pipeline paused at Review: {result['uncertain']} "
                        "uncertain reading(s) require human verification."
                    )
                else:
                    st.success("Pipeline reached Finalized.")
                st.rerun()
            except Exception as exc:
                st.error(f"Pipeline failed: {exc}")

    manuscript = get_manuscript(selected_id, owner_id)
    if manuscript:
        st.write("")
        image_left, image_right = st.columns(2)
        with image_left:
            display_image(manuscript["original_path"], "Preserved original")
        with image_right:
            display_image(manuscript["enhanced_path"], f"Processed View ({preprocess_mode})")

        if manuscript["extracted_full_text"]:
            st.write("")
            st.markdown(
                """
                <div class="vn-card">
                    <div class="vn-eyebrow">Extracted Text Output</div>
                    <div class="vn-card-title">Continuous Document Text</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.text_area(
                "Full Text",
                value=manuscript["extracted_full_text"],
                height=150,
                label_visibility="collapsed",
            )

        items = ocr_items(selected_id, owner_id)
        if items:
            st.write("")
            st.markdown(
                """
                <div class="vn-card">
                    <div class="vn-card-title">Tokenized OCR Entities</div>
                    <div class="vn-card-copy">Detailed character tokens, positions, and confidence scores.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            table = pd.DataFrame(
                [
                    {
                        "Text": item["word_text"],
                        "Confidence": f"{item['confidence']:.1f}%",
                        "Status": item["status"],
                        "Position (X, Y)": f"{item['x']}, {item['y']}",
                    }
                    for item in items
                ]
            )
            st.dataframe(table, use_container_width=True, hide_index=True)


def render_review(user: sqlite3.Row, owner_id: int, owned: list[sqlite3.Row]) -> None:
    page_header(
        "03 · Review",
        "Review uncertain readings",
        "Compare OCR evidence with corpus suggestions. Every decision is recorded in the provenance log.",
    )

    if not owned:
        empty_state("No manuscript available", "Upload and process a manuscript before reviewing uncertain readings.")
        return

    labels = {f"{row['id']} · {row['title']}": row["id"] for row in owned}
    selected_label = st.selectbox("Manuscript", list(labels), key="review_manuscript")
    selected_id = labels[selected_label]

    reviewer = st.text_input("Reviewer", value=str(user["display_name"] or ""))
    uncertain_items = ocr_items(selected_id, owner_id, "uncertain")

    if not uncertain_items:
        empty_state("Nothing waiting for review", "This manuscript currently has no unresolved uncertain OCR readings.")
        return

    for index, item in enumerate(uncertain_items, start=1):
        st.markdown(
            f"""
            <div class="vn-review-card">
                <div class="vn-review-label">Reading {index}</div>
                <div class="vn-ocr-text">{item["word_text"]}</div>
                <div class="vn-confidence">OCR confidence: <strong>{item["confidence"]:.1f}%</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        left, right = st.columns([0.9, 1.35])
        with left:
            display_image(item["crop_path"], f"Evidence crop #{item['id']}")

        with right:
            choices = suggestions_for(item["id"], owner_id)
            chosen_text = None
            chosen_id = None

            if choices:
                choice_labels = [f"{s['suggested_text']}  ·  {s['match_score']:.1f}% match" for s in choices]
                chosen_index = st.radio(
                    "Suggestions",
                    range(len(choices)),
                    format_func=lambda i: choice_labels[i],
                    key=f"choice_{item['id']}",
                    label_visibility="collapsed",
                )
                chosen = choices[chosen_index]
                chosen_text = str(chosen["suggested_text"] or "")
                chosen_id = int(chosen["id"]) if chosen["id"] is not None else None

                reason = st.text_area(
                    "Reason",
                    placeholder="Scholarly reasoning...",
                    key=f"reason_{item['id']}",
                    height=70,
                )

                b1, b2 = st.columns(2)
                with b1:
                    if st.button("Accept", type="primary", key=f"accept_{item['id']}", use_container_width=True):
                        record_review(selected_id, int(item["id"]), "accept", chosen_text, chosen_id, reason, reviewer, owner_id)
                        st.rerun()

                with b2:
                    if st.button("Reject", key=f"reject_{item['id']}", use_container_width=True):
                        record_review(selected_id, int(item["id"]), "reject", None, chosen_id, reason, reviewer, owner_id)
                        st.rerun()
            else:
                st.warning("No corpus match found. Use a manual correction.")

            manual = st.text_input("Manual correction", key=f"manual_{item['id']}", placeholder="Enter final reading...")
            manual_reason = st.text_area("Manual reason", key=f"manual_reason_{item['id']}", placeholder="Reason...", height=70)

            if st.button("Save manual correction", key=f"manual_save_{item['id']}", disabled=not manual.strip(), use_container_width=True):
                record_review(selected_id, int(item["id"]), "manual_edit", str(manual).strip(), None, str(manual_reason), reviewer, owner_id)
                st.rerun()


def render_provenance(owner_id: int, owned: list[sqlite3.Row]) -> None:
    page_header(
        "04 · Trace",
        "Provenance & review history",
        "Inspect the recorded history of scholarly decisions for each manuscript.",
    )

    if not owned:
        empty_state("No manuscript history", "Upload a manuscript to begin building a provenance record.")
        return

    labels = {f"{row['id']} · {row['title']}": row["id"] for row in owned}
    selected_label = st.selectbox("Manuscript", list(labels), key="provenance_manuscript")
    selected_id = labels[selected_label]

    state = get_pipeline_state(selected_id, owner_id)
    if state:
        st.markdown(
            f"""
            <div class="vn-card">
                <div class="vn-eyebrow">PIPELINE STATE</div>
                <div class="vn-card-title">{state["current_stage"]} · {state["status"].replace("_", " ")}</div>
                <div class="vn-card-copy">Updated {state["updated_at"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    remaining = pending_review_count(selected_id, owner_id)
    fin_col, exp_col = st.columns(2)

    with fin_col:
        if st.button(
            "Finalize Manuscript",
            type="primary",
            use_container_width=True,
            key=f"finalize_{selected_id}",
        ):
            if remaining > 0:
                st.warning(
                    f"Cannot finalize yet: {remaining} uncertain reading(s) "
                    "still require scholarly review."
                )
                if st.button(
                    "Go to Review",
                    key=f"goto_review_finalize_{selected_id}",
                    use_container_width=True,
                ):
                    st.session_state["active_workspace_tab"] = "Review Suggestions"
                    st.rerun()
            else:
                try:
                    finalize_manuscript(selected_id, owner_id)
                    st.success("Manuscript finalized.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Finalization failed: {exc}")

    with exp_col:
        if st.button(
            "Create Export Package",
            use_container_width=True,
            key=f"export_{selected_id}",
        ):
            if remaining > 0:
                st.warning(
                    f"Cannot export yet: {remaining} uncertain reading(s) "
                    "still require scholarly review."
                )
                if st.button(
                    "Go to Review",
                    key=f"goto_review_export_{selected_id}",
                    use_container_width=True,
                ):
                    st.session_state["active_workspace_tab"] = "Review Suggestions"
                    st.rerun()
            else:
                try:
                    archive = export_manuscript_package(selected_id, owner_id)
                    st.success(f"Export package created: {archive.name}")
                    st.download_button(
                        "Download Export ZIP",
                        data=archive.read_bytes(),
                        file_name=archive.name,
                        mime="application/zip",
                        use_container_width=True,
                        key=f"download_export_{selected_id}",
                    )
                except Exception as exc:
                    st.error(f"Export failed: {exc}")

    if remaining > 0:
        st.info(
            f"Review progress: {remaining} uncertain reading(s) remain. "
            "Finalize and Export will become available after the review queue is cleared."
        )

    log = review_log(selected_id, owner_id)
    if log.empty:
        empty_state("No review actions yet", "Decisions will appear here once uncertain readings are reviewed.")
    else:
        st.dataframe(log, use_container_width=True, hide_index=True)


def render_authenticated_app(user: sqlite3.Row) -> None:
    owner_id = int(user["id"])
    owned = manuscripts(owner_id)

    logo_uri = asset_data_uri(LOGO_PATH)
    if logo_uri:
        st.sidebar.markdown(
            f"""
            <div style="text-align:center; padding:5px 0 15px;">
                <img src="{logo_uri}" style="width:62px; height:62px; object-fit:contain; border-radius:6px; border:1px solid #c7b17d; padding:2px; background:rgba(0,0,0,0.5);">
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.sidebar.markdown(
        f"""
        <div style="background:#24180f; border:1px solid #4a331c; border-radius:8px; padding:12px; margin-bottom:16px;">
            <div style="font-weight:700; color:#f5d58c;">{user["display_name"]}</div>
            <div style="color:#aeb7bc; font-size:.75rem; margin-top:3px; word-break:break-word;">{user["email"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    workspace_tabs = ["Dashboard", "Upload", "Process", "Review Suggestions", "Provenance"]
    default_index = 0
    if "active_workspace_tab" in st.session_state and st.session_state["active_workspace_tab"] in workspace_tabs:
        default_index = workspace_tabs.index(st.session_state["active_workspace_tab"])

    section = st.sidebar.radio(
        "Workspace",
        workspace_tabs,
        index=default_index,
        label_visibility="collapsed",
    )
    st.session_state["active_workspace_tab"] = section

    st.sidebar.divider()
    if st.sidebar.button("Log out", use_container_width=True, key="logout"):
        st.session_state.pop("user_id", None)
        st.session_state["auth_view"] = "landing"
        st.rerun()

    if section == "Dashboard":
        render_dashboard(user, owned)
    elif section == "Upload":
        render_upload(owner_id)
    elif section == "Process":
        render_process(owner_id, owned)
    elif section == "Review Suggestions":
        render_review(user, owner_id, owned)
    elif section == "Provenance":
        render_provenance(owner_id, owned)


# ============================================================
# HOME / LANDING PAGE & AUTH FLOW
# ============================================================


def render_public_page(page: str) -> None:
    """Render a dedicated public-facing page selected from the landing navbar."""
    pages = {
        "about": {
            "eyebrow": "ABOUT VELLUM NODE",
            "title": "Preserving the Past with Scholarly Precision.",
            "intro": "Vellum Node is a private digital workspace for manuscript restoration, transcription, and evidence-based scholarly review.",
        },
        "features": {
            "eyebrow": "THE PLATFORM",
            "title": "Tools Built for Careful Manuscript Work.",
            "intro": "From preserved originals to reviewed readings, Vellum Node connects image processing, OCR, scholarly comparison, and provenance in one workflow.",
        },
        "documentation": {
            "eyebrow": "DOCUMENTATION",
            "title": "A Transparent Restoration Workflow.",
            "intro": "Follow a documented path from the original manuscript image to reviewed and traceable transcription decisions.",
        },
        "contact": {
            "eyebrow": "CONTACT",
            "title": "Begin a Scholarly Restoration.",
            "intro": "Create a private workspace to preserve manuscript evidence and begin a structured restoration and review process.",
        },
    }

    # Top-left navigation on every dedicated public page.
    top_left, _ = st.columns([0.22, 0.78])
    with top_left:
        if st.button("← Home", use_container_width=False, key=f"public_{page}_top_home"):
            st.session_state["auth_view"] = "landing"
            st.rerun()

    content = pages[page]
    st.markdown(
        f"""
        <div class="vn-public-page">
            <div class="vn-public-eyebrow">{content["eyebrow"]}</div>
            <h1>{content["title"]}</h1>
            <div class="vn-public-rule"></div>
            <p class="vn-public-intro">{content["intro"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if page == "about":
        cards = [
            ("01", "Evidence First", "Original manuscript files remain preserved separately while enhanced images and OCR outputs are treated as working derivatives."),
            ("02", "Scholarly Accountability", "Uncertain readings can be compared with corpus suggestions and the reasoning behind editorial decisions can be retained."),
        ]
        cols = st.columns(2)
        for col, (num, title, copy) in zip(cols, cards):
            with col:
                st.markdown(f"""
                <div class="vn-public-card">
                    <div class="vn-public-card-num">{num}</div>
                    <h3>{title}</h3>
                    <p>{copy}</p>
                </div>
                """, unsafe_allow_html=True)

    elif page == "features":
        feature_rows = [
            ("01", "Advanced OCR", "Multilingual OCR, configurable segmentation, confidence scoring, and token-level output."),
            ("02", "Image Restoration", "Deskewing, adaptive binarization, contrast enhancement, denoising, and processed views."),
            ("03", "Uncertainty Review", "Low-confidence readings are surfaced with evidence crops and corpus-based alternatives."),
            ("04", "Provenance", "Review actions retain the original reading, final value, reviewer, reason, and timestamp."),
        ]
        cols = st.columns(2)
        for i, (num, title, copy) in enumerate(feature_rows):
            with cols[i % 2]:
                st.markdown(f"""
                <div class="vn-public-card">
                    <div class="vn-public-card-num">{num}</div>
                    <h3>{title}</h3>
                    <p>{copy}</p>
                </div>
                """, unsafe_allow_html=True)

    elif page == "documentation":
        steps = [
            ("01", "Preserve", "Upload the original manuscript image and record its title, collection, and script."),
            ("02", "Process", "Choose OCR languages and preprocessing settings, then generate an enhanced working image."),
            ("03", "Review", "Inspect uncertain OCR readings, compare suggestions, accept or reject them, or enter a manual correction."),
            ("04", "Trace", "Review provenance history to understand what changed, who reviewed it, and why."),
        ]
        for num, title, copy in steps:
            st.markdown(f"""
            <div class="vn-doc-step">
                <span>{num}</span>
                <div>
                    <h3>{title}</h3>
                    <p>{copy}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

    elif page == "contact":
        st.markdown("""
        <div class="vn-public-card vn-contact-card">
            <h3>Ready to enter the workspace?</h3>
            <p>Sign in to an existing scholar workspace or create a new account to begin preserving and reviewing manuscripts.</p>
        </div>
        """, unsafe_allow_html=True)

        a, b, _ = st.columns([1, 1, 1])
        with a:
            if st.button("Sign In to Workspace", type="primary", use_container_width=True, key="public_contact_signin"):
                st.session_state["auth_view"] = "login"
                st.rerun()
        with b:
            if st.button("Create New Account", use_container_width=True, key="public_contact_signup"):
                st.session_state["auth_view"] = "signup"
                st.rerun()

    st.write("")


def render_entry() -> None:
    auth_view = st.session_state.get("auth_view", "landing")

    # 1. LANDING / HOME PAGE
    if auth_view == "landing":
        logo_uri = asset_data_uri(LOGO_PATH)
        logo = f"<img src='{logo_uri}' alt='Vellum Node logo' />" if logo_uri else ""
        st.markdown(
            f"""
            <div class="vn-landing-nav">
                <div class="vn-landing-mark">{logo}<span>VELLUM NODE</span></div>
                <div class="vn-top-subtitle">Heritage manuscript restoration</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        nav_home, nav_about, nav_features, nav_docs, nav_contact, nav_signin = st.columns(
            [0.78, 0.78, 0.92, 1.1, 0.9, 1.1]
        )
        with nav_home:
            if st.button("Home", use_container_width=True, key="nav_home"):
                st.session_state["auth_view"] = "landing"
                st.rerun()
        with nav_about:
            if st.button("About", use_container_width=True, key="nav_about"):
                st.session_state["auth_view"] = "about"
                st.rerun()
        with nav_features:
            if st.button("Features", use_container_width=True, key="nav_features"):
                st.session_state["auth_view"] = "features"
                st.rerun()
        with nav_docs:
            if st.button("Documentation", use_container_width=True, key="nav_docs"):
                st.session_state["auth_view"] = "documentation"
                st.rerun()
        with nav_contact:
            if st.button("Contact", use_container_width=True, key="nav_contact"):
                st.session_state["auth_view"] = "contact"
                st.rerun()
        with nav_signin:
            if st.button("Sign In", type="primary", use_container_width=True, key="nav_signin"):
                st.session_state["auth_view"] = "login"
                st.rerun()

        st.markdown(
            """
            <div class="vn-hero" id="home">
                <div class="vn-hero-inner">
                    <div class="vn-hero-kicker">A living archive for fragile knowledge</div>
                    <blockquote class="vn-hero-quote">
                        “A manuscript is not merely a text; it is a witness that has survived time.”
                    </blockquote>
                    <div class="vn-hero-rule"></div>
                    <div class="vn-hero-attribution">PRESERVE THE EVIDENCE. REVEAL THE STORY.</div>
                </div>
            </div>

            <section class="vn-landing-section" id="about">
                <h2 class="vn-section-heading">About Vellum Node</h2>
                <p class="vn-section-intro">
                    Vellum Node gives scholars and conservators a careful digital workspace for restoring,
                    reading, and documenting historic manuscripts without losing sight of the original artifact.
                </p>
            </section>
            """,
            unsafe_allow_html=True,
        )

        features = [
            ("01", "Preserve", "Store original manuscript images and collection details in one protected scholarly record."),
            ("02", "Process", "Use adaptive image enhancement and multilingual OCR to make faint writing easier to study."),
            ("03", "Review", "Compare uncertain readings against reference corpora and record each editorial decision."),
        ]
        feature_columns = st.columns(3)
        for column, (number, title, copy) in zip(feature_columns, features):
            with column:
                st.markdown(
                    f"""
                    <div class="vn-feature-card">
                        <div class="vn-feature-num">{number}</div>
                        <div class="vn-feature-title">{title}</div>
                        <div class="vn-feature-copy">{copy}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown(
            """
            <section class="vn-landing-section" id="features">
                <h2 class="vn-section-heading">Built for rigorous restoration</h2>
                <p class="vn-section-intro">
                    Every stage connects image processing with provenance, making it easier to return to the
                    source and explain how a reading was reached.
                </p>
            </section>
            <section class="vn-landing-section" id="documentation">
                <h2 class="vn-section-heading">Documentation</h2>
                <p class="vn-section-intro">
                    Start by creating a scholar workspace, upload an original image, process it, then review
                    uncertain readings. Your restoration history stays connected to the manuscript record.
                </p>
            </section>
            <section class="vn-landing-section" id="contact">
                <div class="vn-contact-panel">
                    <h2 class="vn-section-heading">Begin with the original</h2>
                    <p class="vn-section-intro" style="margin: 0 auto;">
                        Sign in to access your manuscript workspace and start a documented restoration.
                    </p>
                </div>
            </section>
            """,
            unsafe_allow_html=True,
        )

        _, sign_in, sign_up, _ = st.columns([1, 1.2, 1.2, 1])
        with sign_in:
            if st.button("Sign In to Workspace", type="primary", use_container_width=True, key="home_signin"):
                st.session_state["auth_view"] = "login"
                st.rerun()
        with sign_up:
            if st.button("Create New Account", use_container_width=True, key="home_to_signup"):
                st.session_state["auth_view"] = "signup"
                st.rerun()

    # 2. DEDICATED PUBLIC NAVBAR PAGES
    elif auth_view in {"about", "features", "documentation", "contact"}:
        render_public_page(auth_view)

    # 3. LOGIN PAGE
    elif auth_view == "login":
        home_col, _ = st.columns([0.22, 0.78])
        with home_col:
            if st.button("← Home", use_container_width=False, key="login_top_home"):
                st.session_state["auth_view"] = "landing"
                st.rerun()

        _, col_center, _ = st.columns([1, 1.4, 1])
        with col_center:
            st.markdown('<div class="vn-auth-wrapper">', unsafe_allow_html=True)
            st.markdown(
                """
                <div class="vn-auth-card">
                    <div class="vn-auth-heading">Sign In</div>
                    <div class="vn-auth-copy">Access your private manuscript workspace.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            email = st.text_input("Email", key="login_email", placeholder="Enter your email")
            password = st.text_input("Password", type="password", key="login_password", placeholder="Enter your password")

            if st.button("Enter Vellum Node", type="primary", use_container_width=True):
                user = authenticate(email, password)
                if user:
                    st.session_state["user_id"] = int(user["id"])
                    st.session_state["active_workspace_tab"] = "Dashboard"
                    st.rerun()
                else:
                    st.error("Invalid email or password.")

            st.write("")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("← Back to Home", use_container_width=True, key="login_to_home"):
                    st.session_state["auth_view"] = "landing"
                    st.rerun()
            with col_b:
                if st.button("Create Account", use_container_width=True, key="login_to_signup"):
                    st.session_state["auth_view"] = "signup"
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    # 4. SIGN UP PAGE
    elif auth_view == "signup":
        home_col, _ = st.columns([0.22, 0.78])
        with home_col:
            if st.button("← Home", use_container_width=False, key="signup_top_home"):
                st.session_state["auth_view"] = "landing"
                st.rerun()

        _, col_center, _ = st.columns([1, 1.4, 1])
        with col_center:
            st.markdown('<div class="vn-auth-wrapper">', unsafe_allow_html=True)
            st.markdown(
                """
                <div class="vn-auth-card">
                    <div class="vn-auth-heading">Create Account</div>
                    <div class="vn-auth-copy">Set up your local scholar workspace.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            signup_name = st.text_input("Display name", key="signup_name", placeholder="e.g. Scholar")
            signup_email = st.text_input("Email address", key="signup_email", placeholder="name@domain.com")
            signup_password = st.text_input("Password", type="password", key="signup_password", placeholder="At least 6 characters")

            if st.button("Complete Sign Up", type="primary", use_container_width=True):
                success, message, new_id = register_user(signup_email, signup_name, signup_password)
                if success and new_id:
                    st.session_state["user_id"] = new_id
                    st.session_state["auth_view"] = "login"
                    st.session_state["active_workspace_tab"] = "Dashboard"
                    st.rerun()
                else:
                    st.error(message)

            st.write("")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("← Back to Home", use_container_width=True, key="signup_to_home"):
                    st.session_state["auth_view"] = "landing"
                    st.rerun()
            with col_b:
                if st.button("Sign In Instead", use_container_width=True, key="signup_to_login"):
                    st.session_state["auth_view"] = "login"
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# ENTRY POINT
# ============================================================

def main() -> None:
    st.set_page_config(
        page_title="Vellum Node",
        page_icon="📜",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_db()
    user_id = st.session_state.get("user_id")

    if user_id:
        user = get_user(int(user_id))
        if user:
            apply_branding(is_authenticated=True)
            render_authenticated_app(user)
            return
        st.session_state.pop("user_id", None)

    apply_branding(
        is_authenticated=False,
        show_top_brand=st.session_state.get("auth_view", "landing") != "landing",
    )
    render_entry()


if __name__ == "__main__":
    main()
