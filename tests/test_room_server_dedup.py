import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)

# ---------------------------------------------------------------------------
# Stub pymc_core before loading room_server directly via importlib so we
# never trigger repeater.handler_helpers.__init__ (which pulls in all other
# helpers and their deep pymc_core dependencies).
# ---------------------------------------------------------------------------

def _pkg(name: str, **attrs) -> types.ModuleType:
    """Register a stub package (with __path__) in sys.modules."""
    m = types.ModuleType(name)
    m.__path__ = []
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules.setdefault(name, m)
    return sys.modules[name]

_pkg("pymc_core")
_pkg("pymc_core.protocol", CryptoUtils=MagicMock(), PacketBuilder=MagicMock())
_pkg("pymc_core.protocol.constants", PAYLOAD_TYPE_TXT_MSG=0x04)

# Load room_server.py directly — bypasses handler_helpers/__init__.py entirely.
_rs_path = Path(__file__).parent.parent / "repeater" / "handler_helpers" / "room_server.py"
_spec = importlib.util.spec_from_file_location("repeater.handler_helpers.room_server", _rs_path)
_rs_mod = importlib.util.module_from_spec(_spec)
sys.modules["repeater.handler_helpers.room_server"] = _rs_mod
_spec.loader.exec_module(_rs_mod)

RoomServer = _rs_mod.RoomServer
DEDUP_WINDOW_SECS = _rs_mod.DEDUP_WINDOW_SECS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_room_server():
    """Build a RoomServer with fully mocked infrastructure."""
    local_identity = MagicMock()
    local_identity.get_public_key.return_value = b"\x01" * 32

    db = MagicMock()
    acl = MagicMock()
    packet_injector = AsyncMock()

    acl.get_all_clients.return_value = []
    db.get_all_room_clients.return_value = []
    db.insert_room_message.return_value = 1

    with patch.object(_rs_mod, "_global_push_limiter", None):
        server = RoomServer(
            room_hash=0x42,
            room_name="TestRoom",
            local_identity=local_identity,
            sqlite_handler=db,
            packet_injector=packet_injector,
            acl=acl,
            config={},
        )

    server.db = db
    return server, db


# ===========================================================================
# Deduplication of client retransmissions
# ===========================================================================

class TestAddPostDedup:
    """
    Clients retry a message when they don't receive an ack fast enough.
    The MeshCore client sends up to 3 copies of the same message.  text.py
    normalises the decoded text (strips null bytes and U+FFFD replacement chars)
    before calling add_post(), so all retransmissions of the same message arrive
    with identical text regardless of trailing-byte differences.

    add_post() uses (author, normalised_text) + DEDUP_WINDOW_SECS to drop dupes.
    """

    def test_first_post_is_stored(self):
        server, db = _make_room_server()
        pubkey = b"\xaa" * 32
        with patch("time.time", return_value=1000.0):
            result = _run(server.add_post(pubkey, "hello mesh", sender_timestamp=1000))
        assert result is True
        db.insert_room_message.assert_called_once()

    def test_duplicate_within_window_is_dropped(self):
        """Retransmission with same normalised text within the window is dropped."""
        server, db = _make_room_server()
        pubkey = b"\xbb" * 32
        with patch("time.time", return_value=1000.0):
            _run(server.add_post(pubkey, "hello mesh", sender_timestamp=1000))
        db.insert_room_message.reset_mock()
        # Retry 15 s later — same normalised text, still inside the 30 s window
        with patch("time.time", return_value=1015.0):
            result = _run(server.add_post(pubkey, "hello mesh", sender_timestamp=1015))
        assert result is False
        db.insert_room_message.assert_not_called()

    def test_same_text_after_window_is_accepted(self):
        server, db = _make_room_server()
        pubkey = b"\xcc" * 32
        with patch("time.time", return_value=1000.0):
            _run(server.add_post(pubkey, "hello again", sender_timestamp=1000))
        db.insert_room_message.reset_mock()
        # Same text but after the dedup window expires → genuinely new message
        with patch("time.time", return_value=1000.0 + DEDUP_WINDOW_SECS + 1):
            result = _run(server.add_post(pubkey, "hello again", sender_timestamp=1031))
        assert result is True
        db.insert_room_message.assert_called_once()

    def test_different_text_from_same_author_is_accepted(self):
        server, db = _make_room_server()
        pubkey = b"\xdd" * 32
        with patch("time.time", return_value=1000.0):
            _run(server.add_post(pubkey, "message one", sender_timestamp=1000))
        db.insert_room_message.reset_mock()
        with patch("time.time", return_value=1001.0):
            result = _run(server.add_post(pubkey, "message two", sender_timestamp=1001))
        assert result is True
        db.insert_room_message.assert_called_once()

    def test_same_text_different_authors_both_stored(self):
        server, db = _make_room_server()
        pubkey_a = b"\xee" * 32
        pubkey_b = b"\xff" * 32
        with patch("time.time", return_value=1000.0):
            r1 = _run(server.add_post(pubkey_a, "same text", sender_timestamp=1000))
            r2 = _run(server.add_post(pubkey_b, "same text", sender_timestamp=1000))
        assert r1 is True
        assert r2 is True
        assert db.insert_room_message.call_count == 2

    def test_fallback_duplicate_within_window_is_dropped(self):
        """sender_timestamp=0 (web-API path) still uses text-based dedup."""
        server, db = _make_room_server()
        pubkey = b"\xa1" * 32
        with patch("time.time", return_value=1000.0):
            _run(server.add_post(pubkey, "hello mesh", sender_timestamp=0))
        db.insert_room_message.reset_mock()
        with patch("time.time", return_value=1005.0):
            result = _run(server.add_post(pubkey, "hello mesh", sender_timestamp=0))
        assert result is False
        db.insert_room_message.assert_not_called()

    def test_fallback_same_text_after_window_is_accepted(self):
        server, db = _make_room_server()
        pubkey = b"\xa2" * 32
        with patch("time.time", return_value=1000.0):
            _run(server.add_post(pubkey, "hello again", sender_timestamp=0))
        db.insert_room_message.reset_mock()
        with patch("time.time", return_value=1000.0 + DEDUP_WINDOW_SECS + 1):
            result = _run(server.add_post(pubkey, "hello again", sender_timestamp=0))
        assert result is True
        db.insert_room_message.assert_called_once()
