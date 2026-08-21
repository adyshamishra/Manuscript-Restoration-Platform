from pathlib import Path
import tempfile

import app

with tempfile.TemporaryDirectory() as temp:
    root = Path(temp)
    app.DB_PATH = root / "auth_test.db"
    app.ORIGINALS_DIR = root / "originals"
    app.ENHANCED_DIR = root / "enhanced"
    app.CROPS_DIR = root / "crops"
    for directory in (app.ORIGINALS_DIR, app.ENHANCED_DIR, app.CROPS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    app.init_db()
    ok_a, _, alice_id = app.register_user("alice@example.com", "Alice", "alice123")
    ok_b, _, bob_id = app.register_user("bob@example.com", "Bob", "bob12345")
    assert ok_a and ok_b and alice_id and bob_id
    assert app.authenticate("alice@example.com", "alice123") is not None
    assert app.authenticate("alice@example.com", "wrong-password") is None

    class Uploaded:
        name = "private.jpg"

        def getvalue(self):
            return b"fake image bytes for ownership test"

    alice_manuscript = app.save_upload(Uploaded(), "Alice Private Manuscript", "Alice Collection", "Demo Script", alice_id)
    assert len(app.manuscripts(alice_id)) == 1
    assert len(app.manuscripts(bob_id)) == 0
    assert app.get_manuscript(alice_manuscript, bob_id) is None
    assert app.get_manuscript(alice_manuscript, alice_id) is not None

print({"users": 2, "ownership_isolation": "PASS", "login_verification": "PASS", "status": "PASS"})
