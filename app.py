from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import cv2
import pandas as pd
import pytesseract
import streamlit as st
from pytesseract import Output
from rapidfuzz import fuzz, process

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
ORIGINALS_DIR = DATA_DIR / "originals"
ENHANCED_DIR = DATA_DIR / "enhanced"
CROPS_DIR = DATA_DIR / "crops"
ASSETS_DIR = APP_DIR / "assets"
DB_PATH = DATA_DIR / "app.db"
CORPUS_PATH = APP_DIR / "corpus.txt"
LOGO_PATH = ASSETS_DIR / "vellum_node_logo.png"

for directory in (ORIGINALS_DIR, ENHANCED_DIR, CROPS_DIR, ASSETS_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    """Open a SQLite connection and always close it, including on Windows."""
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
            """
        )
        # These columns make an existing pre-login database usable after upgrade.
        columns = {row["name"] for row in db.execute("PRAGMA table_info(manuscripts)").fetchall()}
        if "owner_id" not in columns:
            db.execute("ALTER TABLE manuscripts ADD COLUMN owner_id INTEGER")


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
    return f"{salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, digest_hex = stored.split("$", 1)
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
        return False, "For this local demo, use a password with at least 6 characters.", None
    try:
        with connect() as db:
            cursor = db.execute(
                "INSERT INTO users (email, display_name, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (email, display_name, hash_password(password), now()),
            )
            inserted_id = cursor.lastrowid
            if inserted_id is None:
                raise RuntimeError("The new user ID was not returned by SQLite")
            return True, "Account created. You can now log in.", int(inserted_id)
    except sqlite3.IntegrityError:
        return False, "An account with this email already exists.", None


def authenticate(email: str, password: str) -> sqlite3.Row | None:
    with connect() as db:
        user = db.execute("SELECT * FROM users WHERE email = ? COLLATE NOCASE", (email.strip().lower(),)).fetchone()
    if user and verify_password(password, user["password_hash"]):
        return user
    return None


def get_user(user_id: int) -> sqlite3.Row | None:
    with connect() as db:
        return db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_corpus() -> list[str]:
    if not CORPUS_PATH.exists():
        return []
    return sorted({line.strip() for line in CORPUS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()})


def user_dir(base: Path, owner_id: int) -> Path:
    directory = base / f"user_{owner_id}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def manuscripts(owner_id: int) -> list[sqlite3.Row]:
    with connect() as db:
        return db.execute("SELECT * FROM manuscripts WHERE owner_id = ? ORDER BY id DESC", (owner_id,)).fetchall()


def get_manuscript(manuscript_id: int, owner_id: int | None = None) -> sqlite3.Row | None:
    query = "SELECT * FROM manuscripts WHERE id = ?"
    params: list[Any] = [manuscript_id]
    if owner_id is not None:
        query += " AND owner_id = ?"
        params.append(owner_id)
    with connect() as db:
        return db.execute(query, params).fetchone()


def save_upload(uploaded_file: Any, title: str, collection: str, script: str, owner_id: int) -> int:
    payload = uploaded_file.getvalue()
    digest = sha256_bytes(payload)[:16]
    safe_suffix = Path(uploaded_file.name).suffix.lower() or ".jpg"
    original_path = user_dir(ORIGINALS_DIR, owner_id) / f"manuscript_{digest}{safe_suffix}"
    if not original_path.exists():
        original_path.write_bytes(payload)
    with connect() as db:
        cursor = db.execute(
            """INSERT INTO manuscripts
            (owner_id, title, collection_name, script_name, original_filename, original_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (owner_id, title, collection, script, uploaded_file.name, str(original_path), now()),
        )
        inserted_id = cursor.lastrowid
        if inserted_id is None:
            raise RuntimeError("The new manuscript ID was not returned by SQLite")
        return int(inserted_id)


def enhance_and_ocr(manuscript_id: int, threshold: float, owner_id: int) -> tuple[int, int]:
    manuscript = get_manuscript(manuscript_id, owner_id)
    if manuscript is None:
        raise ValueError("Manuscript not found for this user")
    original = cv2.imread(manuscript["original_path"])
    if original is None:
        raise ValueError("The original image could not be read")
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(denoised)
    enhanced_path = user_dir(ENHANCED_DIR, owner_id) / f"manuscript_{manuscript_id}_enhanced.png"
    cv2.imwrite(str(enhanced_path), enhanced)
    data = pytesseract.image_to_data(enhanced, output_type=Output.DICT, config="--psm 6")
    with connect() as db:
        db.execute("DELETE FROM suggestions WHERE ocr_word_id IN (SELECT id FROM ocr_words WHERE manuscript_id = ?)", (manuscript_id,))
        db.execute("DELETE FROM ocr_words WHERE manuscript_id = ?", (manuscript_id,))
        db.execute("UPDATE manuscripts SET enhanced_path = ? WHERE id = ? AND owner_id = ?", (str(enhanced_path), manuscript_id, owner_id))
        corpus = load_corpus()
        word_count = 0
        uncertain_count = 0
        for i, raw_text in enumerate(data["text"]):
            text = (raw_text or "").strip()
            try:
                confidence = float(data["conf"][i])
            except (ValueError, TypeError):
                confidence = -1
            if not text or confidence < 0:
                continue
            x, y, w, h = [int(data[key][i]) for key in ("left", "top", "width", "height")]
            status = "uncertain" if confidence < threshold else "normal"
            crop_path = None
            if status == "uncertain" and w > 0 and h > 0:
                pad = 12
                x0, y0 = max(0, x - pad), max(0, y - pad)
                x1, y1 = min(enhanced.shape[1], x + w + pad), min(enhanced.shape[0], y + h + pad)
                crop_path_obj = user_dir(CROPS_DIR, owner_id) / f"manuscript_{manuscript_id}_ocr_{word_count + 1}.png"
                cv2.imwrite(str(crop_path_obj), enhanced[y0:y1, x0:x1])
                crop_path = str(crop_path_obj)
            cursor = db.execute(
                """INSERT INTO ocr_words
                (manuscript_id, word_text, confidence, x, y, width, height, crop_path, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (manuscript_id, text, confidence, x, y, w, h, crop_path, status),
            )
            if status == "uncertain":
                uncertain_count += 1
                matches = process.extract(text, corpus, scorer=fuzz.WRatio, limit=3) if corpus else []
                for rank, (candidate, score, _) in enumerate(matches, start=1):
                    db.execute(
                        """INSERT INTO suggestions
                        (ocr_word_id, suggested_text, match_score, corpus_name, rank)
                        VALUES (?, ?, ?, ?, ?)""",
                        (int(cursor.lastrowid or 0), candidate, float(score), "Local demo corpus", rank),
                    )
            word_count += 1
    return word_count, uncertain_count


def ocr_items(manuscript_id: int, owner_id: int, status: str | None = None) -> list[sqlite3.Row]:
    query = "SELECT ow.* FROM ocr_words ow JOIN manuscripts m ON m.id = ow.manuscript_id WHERE ow.manuscript_id = ? AND m.owner_id = ?"
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
            """SELECT s.* FROM suggestions s
            JOIN ocr_words ow ON ow.id = s.ocr_word_id
            JOIN manuscripts m ON m.id = ow.manuscript_id
            WHERE s.ocr_word_id = ? AND m.owner_id = ? ORDER BY s.rank""",
            (word_id, owner_id),
        ).fetchall()


def record_review(manuscript_id: int, word_id: int, action: str, final_text: str | None, suggestion_id: int | None, reason: str, reviewer: str, owner_id: int) -> None:
    with connect() as db:
        word = db.execute(
            """SELECT ow.word_text FROM ocr_words ow JOIN manuscripts m ON m.id = ow.manuscript_id
            WHERE ow.id = ? AND ow.manuscript_id = ? AND m.owner_id = ?""",
            (word_id, manuscript_id, owner_id),
        ).fetchone()
        if word is None:
            raise ValueError("OCR word not found for this user")
        db.execute(
            """INSERT INTO review_log
            (manuscript_id, ocr_word_id, suggestion_id, action, previous_text, final_text, reviewer, reason, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
            db.execute("UPDATE ocr_words SET status = 'manual_edit' WHERE id = ? AND manuscript_id = ?", (word_id, manuscript_id))


def review_log(manuscript_id: int, owner_id: int) -> pd.DataFrame:
    with connect() as db:
        rows = db.execute(
            """SELECT rl.created_at AS Time, rl.action AS Action, rl.previous_text AS 'Original OCR',
            rl.final_text AS 'Final value', rl.reviewer AS Reviewer, rl.reason AS Reason
            FROM review_log rl JOIN manuscripts m ON m.id = rl.manuscript_id
            WHERE rl.manuscript_id = ? AND m.owner_id = ? ORDER BY rl.id DESC""",
            (manuscript_id, owner_id),
        ).fetchall()
    return pd.DataFrame([dict(row) for row in rows])


def display_image(path: str | None, caption: str) -> None:
    if path and Path(path).exists():
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.info(f"{caption} is not available yet.")


def asset_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def apply_branding() -> None:
    logo_uri = asset_data_uri(LOGO_PATH)
    st.markdown(
        f"""<style>
        .stApp {{ background: linear-gradient(135deg, #071a2a 0%, #0d3046 48%, #162b3c 100%); color: #eef5f8; }}
        [data-testid="stHeader"] {{ background: rgba(0,0,0,0); }}
        .brand-bar {{ display:flex; align-items:center; justify-content:space-between; padding:10px 0 18px; border-bottom:1px solid rgba(211,231,240,.25); margin-bottom:24px; }}
        .brand-left {{ display:flex; align-items:center; gap:12px; }}
        .brand-left img {{ width:54px; height:54px; object-fit:cover; border-radius:50%; border:2px solid #c9a76a; }}
        .brand-name {{ font-family: Georgia, serif; font-size:25px; font-weight:700; letter-spacing:1px; color:#e7cc91; }}
        .brand-tagline {{ color:#b8d2df; font-size:12px; }}
        .hero-card, .info-card {{ background:rgba(234,244,248,.12); border:1px solid rgba(221,237,244,.28); border-radius:20px; padding:30px; box-shadow:0 18px 50px rgba(0,0,0,.23); backdrop-filter: blur(8px); }}
        .hero-title {{ font-family:Georgia,serif; font-size:48px; line-height:1.05; color:#f5e1ad; margin:0 0 12px; }}
        .hero-copy {{ font-size:18px; line-height:1.6; color:#e2eff4; }}
        .section-label {{ color:#d2b374; letter-spacing:2px; text-transform:uppercase; font-size:12px; font-weight:700; }}
        .login-card {{ max-width:560px; margin:28px auto; background:rgba(221,237,245,.94); color:#102536; padding:28px 34px; border-radius:22px; box-shadow:0 20px 70px rgba(0,0,0,.35); }}
        .login-title {{ text-align:center; font-family:Georgia,serif; color:#102536; font-size:32px; font-weight:700; margin-bottom:4px; }}
        .muted {{ color:#c4d8df; }}
        </style>
        <div class="brand-bar"><div class="brand-left">{'<img src="'+logo_uri+'" />' if logo_uri else ''}<div><div class="brand-name">VELLUM NODE</div><div class="brand-tagline">Heritage reconstruction through transparent scholarship</div></div></div><div class="muted">AI-assisted · Scholar-controlled</div></div>""",
        unsafe_allow_html=True,
    )


def public_nav() -> str:
    if "public_page" not in st.session_state:
        st.session_state["public_page"] = "Home"
    cols = st.columns([4, 1, 1, 1, 1])
    with cols[1]:
        if st.button("Home", use_container_width=True):
            st.session_state["public_page"] = "Home"
            st.rerun()
    with cols[2]:
        if st.button("About", use_container_width=True):
            st.session_state["public_page"] = "About"
            st.rerun()
    with cols[3]:
        if st.button("Services", use_container_width=True):
            st.session_state["public_page"] = "Services"
            st.rerun()
    with cols[4]:
        if st.button("Login", type="primary", use_container_width=True):
            st.session_state["auth_mode"] = "Login"
            st.session_state["public_page"] = "Login"
            st.rerun()
    return st.session_state["public_page"]


def render_public_page(page: str) -> None:
    if page == "About":
        st.markdown('<div class="info-card"><div class="section-label">About Vellum Node</div><h1 class="hero-title">Restoring memory with accountable AI.</h1><p class="hero-copy">Vellum Node is a scholar-in-the-loop workspace for faded manuscripts, palm-leaf records, inscriptions, and archival documents. It keeps the original image untouched while making every proposed reading visible, reviewable, and traceable.</p></div>', unsafe_allow_html=True)
    elif page == "Services":
        st.markdown('<div class="info-card"><div class="section-label">Services</div><h1 class="hero-title">From damaged image to reviewed reading.</h1><p class="hero-copy">Upload and preserve an original, enhance a working copy, run OCR, flag uncertain words, compare local corpus suggestions, accept or reject scholarly candidates, and inspect the complete provenance history.</p></div>', unsafe_allow_html=True)
        st.write("")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="info-card"><h3>Preserve</h3><p>Original evidence remains untouched and separate from every generated working copy.</p></div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="info-card"><h3>Suggest</h3><p>OCR and transparent fuzzy matching produce candidates instead of silent guesses.</p></div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="info-card"><h3>Review</h3><p>Experts accept, reject, or manually edit each uncertain reading with a recorded reason.</p></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="hero-card"><div class="section-label">Scholar-in-the-loop restoration</div><h1 class="hero-title">Give damaged records a careful second life.</h1><p class="hero-copy">Vellum Node helps heritage teams enhance faded documents, explore possible readings, and preserve the reasoning behind every edit.</p></div>', unsafe_allow_html=True)
        st.write("")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="info-card"><h2>Why it matters</h2><p>Historical restoration needs both useful automation and respect for uncertainty. Our workflow keeps evidence, suggestions, and expert decisions visibly separate.</p></div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="info-card"><h2>Ready to explore?</h2><p>Create a local demo account to save your manuscripts, review history, and provenance records in your own private workspace.</p></div>', unsafe_allow_html=True)


def render_auth() -> bool:
    st.markdown('<div class="login-card"><div class="login-title">Welcome to Vellum Node</div><p style="text-align:center;color:#40596a">Sign in to your private scholarly workspace.</p></div>', unsafe_allow_html=True)
    mode = st.radio("Account", ["Login", "Register"], horizontal=True, index=0 if st.session_state.get("auth_mode", "Login") == "Login" else 1)
    if mode == "Login":
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", type="primary", use_container_width=True)
        if submitted:
            user = authenticate(email, password)
            if user:
                st.session_state["user_id"] = int(user["id"])
                st.session_state["auth_mode"] = "Login"
                st.success(f"Welcome back, {user['display_name']}.")
                st.rerun()
            else:
                st.error("Email or password is incorrect.")
    else:
        with st.form("register_form"):
            display_name = st.text_input("Your name")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            confirm = st.text_input("Confirm password", type="password")
            submitted = st.form_submit_button("Create account", type="primary", use_container_width=True)
        if submitted:
            if password != confirm:
                st.error("The passwords do not match.")
            else:
                ok, message, _ = register_user(email, display_name, password)
                if ok:
                    st.success(message)
                    st.session_state["auth_mode"] = "Login"
                    st.rerun()
                else:
                    st.error(message)
    return False


def render_dashboard(user: sqlite3.Row, owned: list[sqlite3.Row]) -> None:
    st.header(f"Welcome, {user['display_name']}")
    st.write("This is your private Vellum Node workspace. Other local demo accounts cannot see these manuscripts or review records.")
    reviewed = 0
    with connect() as db:
        reviewed = db.execute("SELECT COUNT(*) AS n FROM review_log rl JOIN manuscripts m ON m.id = rl.manuscript_id WHERE m.owner_id = ?", (user["id"],)).fetchone()["n"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Your manuscripts", len(owned))
    c2.metric("Your review actions", reviewed)
    c3.metric("Account created", user["created_at"][:10])
    st.subheader("Your manuscript history")
    if owned:
        st.dataframe(pd.DataFrame([{ "Title": row["title"], "Collection": row["collection_name"], "Script": row["script_name"], "Added": row["created_at"] } for row in owned]), use_container_width=True, hide_index=True)
    else:
        st.info("You have not uploaded a manuscript yet. Open Upload to begin.")


def render_authenticated_app(user: sqlite3.Row) -> None:
    owner_id = int(user["id"])
    owned = manuscripts(owner_id)
    st.sidebar.image(str(LOGO_PATH), width=100)
    st.sidebar.markdown(f"### {user['display_name']}")
    st.sidebar.caption(user["email"])
    section = st.sidebar.radio("Workspace", ["Dashboard", "Upload", "Process", "Review Suggestions", "Provenance"])
    if st.sidebar.button("Log out", use_container_width=True):
        st.session_state.pop("user_id", None)
        st.session_state["public_page"] = "Home"
        st.rerun()
    st.sidebar.divider()
    st.sidebar.caption("Your data is filtered by your account.")

    if section == "Dashboard":
        render_dashboard(user, owned)
    elif section == "Upload":
        st.header("Upload manuscript")
        st.write("Your original image is preserved in your private account folder and never overwritten.")
        title = st.text_input("Manuscript title", placeholder="Example: Palm-leaf record A")
        collection = st.text_input("Local collection", value="Demo Local Collection")
        script = st.text_input("Indian script used for this demo", value="Selected Indian Script")
        uploaded = st.file_uploader("Upload a JPG, JPEG, or PNG image", type=["jpg", "jpeg", "png"])
        if uploaded:
            st.image(uploaded, caption="Preview — this will be preserved as your original", use_container_width=True)
        if st.button("Save Original", type="primary", disabled=not (title.strip() and uploaded)):
            try:
                manuscript_id = save_upload(uploaded, title.strip(), collection.strip(), script.strip(), owner_id)
                st.success(f"Original preserved in your workspace. Manuscript ID: {manuscript_id}")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not save the manuscript: {exc}")
    elif section == "Process":
        st.header("Enhance image and run OCR")
        if not owned:
            st.warning("Upload a manuscript first.")
        else:
            labels = {f"{row['id']}: {row['title']}": row["id"] for row in owned}
            selected_id = labels[st.selectbox("Choose manuscript", list(labels))]
            threshold = st.slider("Uncertainty threshold", 0, 100, 60)
            if st.button("Enhance and Run OCR", type="primary"):
                try:
                    with st.spinner("Enhancing image and running OCR..."):
                        total, uncertain = enhance_and_ocr(selected_id, threshold, owner_id)
                    st.success(f"Finished. Detected {total} text items; {uncertain} marked uncertain.")
                except Exception as exc:
                    st.error(f"Processing failed: {exc}")
            manuscript = get_manuscript(selected_id, owner_id)
            if manuscript:
                c1, c2 = st.columns(2)
                with c1:
                    display_image(manuscript["original_path"], "Original — preserved and untouched")
                with c2:
                    display_image(manuscript["enhanced_path"], "Enhanced working copy")
                items = ocr_items(selected_id, owner_id)
                if items:
                    st.dataframe(pd.DataFrame([dict(item) for item in items]), use_container_width=True, hide_index=True)
    elif section == "Review Suggestions":
        st.header("Review uncertain readings")
        if not owned:
            st.warning("Upload and process a manuscript first.")
        else:
            labels = {f"{row['id']}: {row['title']}": row["id"] for row in owned}
            selected_id = labels[st.selectbox("Choose manuscript", list(labels))]
            reviewer_default = str(user["display_name"] or "")
            reviewer = st.text_input("Reviewer name", value=reviewer_default)

            uncertain_items = ocr_items(selected_id, owner_id, "uncertain")
            if not uncertain_items:
                st.success("There are no unresolved uncertain words.")
            for item in uncertain_items:
                st.divider()
                left, right = st.columns([1, 2])
                with left:
                    display_image(item["crop_path"], f"Evidence crop for OCR item #{item['id']}")
                with right:
                    st.subheader(f"OCR reading: `{item['word_text']}`")
                    st.write(f"OCR confidence: **{item['confidence']:.1f}%**")
                    choices = suggestions_for(item["id"], owner_id)
                    if choices:
                        choice_labels = [f"{s['suggested_text']} — {s['match_score']:.1f}% match" for s in choices]
                        chosen_index = st.radio("Suggested reconstructions", range(len(choices)), format_func=lambda i: choice_labels[i], key=f"choice_{item['id']}")
                        chosen = choices[chosen_index]
                        chosen_text = str(chosen["suggested_text"] or "")
                        chosen_id = int(chosen["id"]) if chosen["id"] is not None else None
                        reason = st.text_input("Reason (optional)", key=f"reason_{item['id']}")
                        b1, b2 = st.columns(2)
                        with b1:
                            if st.button("Accept suggestion", key=f"accept_{item['id']}"):
                                record_review(selected_id, int(item["id"]), "accept", chosen_text, chosen_id, reason, reviewer, owner_id)
                                st.success("Suggestion accepted and logged.")
                                st.rerun()
                        with b2:
                            if st.button("Reject suggestion", key=f"reject_{item['id']}"):
                                record_review(selected_id, int(item["id"]), "reject", None, chosen_id, reason, reviewer, owner_id)
                                st.success("Suggestion rejected and logged.")
                                st.rerun()
                    else:
                        st.warning("No corpus match was found. Use manual correction.")
                    manual = st.text_input("Manual scholarly correction", key=f"manual_{item['id']}")
                    manual_reason = st.text_input("Reason for manual correction", key=f"manual_reason_{item['id']}")
                    if st.button("Save manual correction", key=f"manual_save_{item['id']}", disabled=not manual.strip()):
                        record_review(selected_id, int(item["id"]), "manual_edit", str(manual).strip(), None, str(manual_reason), reviewer, owner_id)
                        st.success("Manual correction saved and logged.")
                        st.rerun()
    elif section == "Provenance":
        st.header("Your provenance and review history")
        if not owned:
            st.info("You have no manuscript history yet.")
        else:
            labels = {f"{row['id']}: {row['title']}": row["id"] for row in owned}
            selected_id = labels[st.selectbox("Choose manuscript", list(labels))]
            log = review_log(selected_id, owner_id)
            if log.empty:
                st.info("No review actions have been recorded for this manuscript yet.")
            else:
                st.dataframe(log, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Vellum Node", page_icon="📜", layout="wide")
    init_db()
    apply_branding()
    user_id = st.session_state.get("user_id")
    if user_id:
        user = get_user(int(user_id))
        if user:
            render_authenticated_app(user)
            return
        st.session_state.pop("user_id", None)
    page = public_nav()
    if page == "Login":
        render_auth()
    else:
        render_public_page(page)
    st.divider()
    st.caption("Vellum Node · Original evidence preserved · Suggestions reviewed by people")


if __name__ == "__main__":
    main()
