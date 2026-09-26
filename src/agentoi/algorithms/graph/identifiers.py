import base64
import hashlib
import uuid as _uuid

_ADJ = [
    "brisk",
    "calm",
    "clever",
    "crisp",
    "daring",
    "eager",
    "fuzzy",
    "gentle",
    "jolly",
    "lively",
    "mellow",
    "nimble",
    "plucky",
    "quiet",
    "spry",
    "swift",
]
_NOUN = [
    "otter",
    "lynx",
    "falcon",
    "badger",
    "koala",
    "panda",
    "beaver",
    "ram",
    "heron",
    "bison",
    "orca",
    "eagle",
    "python",
    "tiger",
    "yak",
    "ibis",
]

_BASE62_ALPH = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def _uuid_to_base62(uid: _uuid.UUID, pad_to: int = 22) -> str:
    x = uid.int  # 128-bit int
    if x == 0:
        s = _BASE62_ALPH[0]
    else:
        out = []
        while x:
            x, r = divmod(x, 62)
            out.append(_BASE62_ALPH[r])
        s = "".join(reversed(out))
    # left-pad for consistent length (22 for 128-bit)
    if pad_to:
        s = s.rjust(pad_to, _BASE62_ALPH[0])
    return s


def _uuid_to_base32(uid: _uuid.UUID) -> str:
    # RFC 4648 Base32 uses A-Z and 2-7 (alphanumeric). strip '=' padding.
    return base64.b32encode(uid.bytes).decode().rstrip("=")


def name_from_uuid(u: str | _uuid.UUID, style: str = "codename") -> str:
    """generate a stable name from a uuid"""
    uid = u if isinstance(u, _uuid.UUID) else _uuid.UUID(str(u))
    if style == "slug":
        return base64.urlsafe_b64encode(uid.bytes).decode().rstrip("=")
    if style in ("slug62", "slug_alnum"):
        return _uuid_to_base62(uid)
    if style == "slug32":
        return _uuid_to_base32(uid)
    # codename
    h = hashlib.blake2b(uid.bytes, digest_size=4).digest()  # 4 bytes
    adj = _ADJ[h[0] % len(_ADJ)]
    noun = _NOUN[h[1] % len(_NOUN)]
    suffix = h[2:4].hex()
    return f"{adj}-{noun}-{suffix}"
