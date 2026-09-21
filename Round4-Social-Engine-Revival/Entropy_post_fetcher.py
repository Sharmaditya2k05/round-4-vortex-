"""
Fetch the text of a public post from a URL, so the dashboard can score anything.

Team Entropy | Data Vortex, AARUUSH'26

Each platform is handled through the same public, key-less endpoint the Round 3
collector uses, falling back to reading the page itself. Nothing here follows
instructions found in fetched content - the text is only ever returned for
scoring.
"""
from __future__ import annotations

import html
import ipaddress
import json
import re
import socket
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

UA = "Entropy-DataVortex/1.0 (AARUUSH26 student project; contact via GitHub)"
TIMEOUT = 15
MAX_BYTES = 2_000_000
ATOM = {"a": "http://www.w3.org/2005/Atom"}


class FetchError(Exception):
    pass


def _guard(url: str) -> urllib.parse.ParseResult:
    """Refuse anything that is not a public http(s) address."""
    u = urllib.parse.urlparse(url.strip())
    if u.scheme not in ("http", "https") or not u.hostname:
        raise FetchError("That does not look like a http(s) link.")
    try:
        for info in socket.getaddrinfo(u.hostname, None):
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                raise FetchError("That address is not public.")
    except socket.gaierror:
        raise FetchError(f"Could not resolve {u.hostname}.")
    return u


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return r.read(MAX_BYTES)


def _strip(markup: str) -> str:
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", markup or "")
    text = re.sub(r"(?i)<br\s*/?>|</p>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"[ \t\r\f\v]+", " ", html.unescape(text)).strip()


# --------------------------------------------------------------- per platform
def _hackernews(u):
    ident = urllib.parse.parse_qs(u.query).get("id", [None])[0]
    if not ident:
        raise FetchError("No item id in that Hacker News link.")
    item = json.loads(_get(f"https://hn.algolia.com/api/v1/items/{ident}"))
    body = _strip(item.get("text") or "")
    title = item.get("title") or ""
    if not (body or title):
        raise FetchError("That Hacker News item has no text of its own (it may be a bare link).")
    return {"source": "hackernews", "title": title, "text": body,
            "author": item.get("author"), "score": item.get("points")}


def _reddit(u):
    m = re.search(r"/comments/([a-z0-9]+)", u.path, re.I)
    if not m:
        raise FetchError("That does not look like a Reddit post link.")
    # Reddit returns 403 for anonymous JSON, so the Atom view is used instead
    raw = _get(f"https://www.reddit.com/comments/{m.group(1)}.rss")
    root = ET.fromstring(raw)
    entry = root.find("a:entry", ATOM)
    if entry is None:
        raise FetchError("Reddit returned no entry for that link (it may be private or removed).")
    title = (entry.findtext("a:title", "", ATOM) or "").strip()
    body = _strip(entry.findtext("a:content", "", ATOM) or "")
    author = (entry.find("a:author/a:name", ATOM).text
              if entry.find("a:author/a:name", ATOM) is not None else None)
    return {"source": "reddit", "title": title, "text": body, "author": author, "score": None}


def _mastodon(u):
    ident = u.path.rstrip("/").split("/")[-1]
    if not ident.isdigit():
        raise FetchError("Could not find a status id in that Mastodon link.")
    st = json.loads(_get(f"https://{u.hostname}/api/v1/statuses/{ident}"))
    return {"source": "mastodon", "title": "", "text": _strip(st.get("content") or ""),
            "author": (st.get("account") or {}).get("acct"),
            "score": (st.get("favourites_count", 0) or 0) + (st.get("reblogs_count", 0) or 0)}


def _lemmy(u):
    m = re.search(r"/post/(\d+)", u.path)
    if not m:
        raise FetchError("Could not find a post id in that Lemmy link.")
    data = json.loads(_get(f"https://{u.hostname}/api/v3/post?id={m.group(1)}"))
    view = data.get("post_view", {})
    post = view.get("post", {})
    return {"source": "lemmy", "title": post.get("name") or "",
            "text": _strip(post.get("body") or ""),
            "author": (view.get("creator") or {}).get("name"),
            "score": (view.get("counts") or {}).get("score")}


def _generic(u):
    markup = _get(u.geturl()).decode("utf-8", "ignore")
    og = re.search(r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\']([^"\']+)', markup, re.I)
    title = re.search(r"(?is)<title[^>]*>(.*?)</title>", markup)
    paras = re.findall(r"(?is)<p[^>]*>(.*?)</p>", markup)
    body = "\n".join(_strip(p) for p in paras)
    body = body if len(body) > 120 else _strip(og.group(1)) if og else body
    if not body.strip():
        raise FetchError("Could not find readable text on that page.")
    return {"source": u.hostname, "title": _strip(title.group(1)) if title else "",
            "text": body[:6000], "author": None, "score": None}


def fetch_post(url: str) -> dict:
    """Return {source, title, text, author, score} for a public post URL."""
    u = _guard(url)
    host = (u.hostname or "").lower()
    try:
        if "news.ycombinator.com" in host:
            out = _hackernews(u)
        elif "reddit.com" in host:
            out = _reddit(u)
        elif "lemmy" in host or "/post/" in u.path and "programming.dev" in host:
            out = _lemmy(u)
        elif re.search(r"/@[^/]+/\d+$|/statuses/\d+$", u.path):
            out = _mastodon(u)
        else:
            out = _generic(u)
    except FetchError:
        raise
    except urllib.error.HTTPError as e:
        raise FetchError(f"{host} refused the request (HTTP {e.code}). "
                         "Public posts only - private, deleted or login-walled pages cannot be read.")
    except Exception as e:
        raise FetchError(f"Could not read that link: {type(e).__name__}")
    combined = (out["title"] + "\n" + out["text"]).strip()
    if len(combined) < 15:
        raise FetchError("That post has almost no text to score.")
    out["combined"] = combined
    out["url"] = u.geturl()
    return out
