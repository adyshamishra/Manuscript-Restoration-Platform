from pathlib import Path
import sys

import app

DEFAULT_IMAGE = Path(__file__).parent / "sample_input.jpg"
source = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else DEFAULT_IMAGE
if not source.exists():
    raise SystemExit(f"Test image not found: {source}\nPass an image path: python smoke_test.py path/to/image.jpg")


class Uploaded:
    name = source.name

    def getvalue(self):
        return source.read_bytes()


app.init_db()
with app.connect() as db:
    db.execute("DELETE FROM review_log")
    db.execute("DELETE FROM suggestions")
    db.execute("DELETE FROM ocr_words")
    db.execute("DELETE FROM manuscripts")
    db.execute("DELETE FROM users")

ok, message, user_id = app.register_user("smoke@example.com", "Smoke Scholar", "demo123")
assert ok, message
assert user_id is not None
assert app.authenticate("smoke@example.com", "demo123") is not None

manuscript_id = app.save_upload(Uploaded(), "Smoke Test Manuscript", "Demo Local Collection", "Selected Indian Script", user_id)
total, uncertain = app.enhance_and_ocr(manuscript_id, 90, user_id)
assert total >= 0
manuscript = app.get_manuscript(manuscript_id, user_id)
assert manuscript is not None
assert manuscript["enhanced_path"] is not None
assert Path(manuscript["enhanced_path"]).exists()
items = app.ocr_items(manuscript_id, user_id)
assert len(items) == total
if items:
    app.record_review(manuscript_id, items[0]["id"], "manual_edit", "test-reading", None, "Smoke test", "Smoke Scholar", user_id)
    assert not app.review_log(manuscript_id, user_id).empty

print({"image": str(source), "user_id": user_id, "manuscript_id": manuscript_id, "ocr_items": total, "uncertain_items": uncertain, "status": "PASS"})
