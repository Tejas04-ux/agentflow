import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from app.crawler import UnsafeURL, check_url, crawl, find_urls

PAGES = {
    "/": b"<html><head><title>Home</title></head><body><nav>MENU</nav><p>Welcome to Nimbus.</p>"
         b"<a href='/pricing'>Pricing</a><a href='http://other.invalid/x'>Other</a><script>evil()</script></body></html>",
    "/pricing": b"<html><head><title>Pricing</title></head><body><p>Pro costs $9 per month.</p></body></html>",
}


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/redirect":
            self.send_response(302); self.send_header("Location", "/pricing"); self.end_headers(); return
        body = PAGES.get(self.path)
        self.send_response(200 if body else 404)
        self.send_header("Content-Type", "text/html" if body else "text/plain")
        self.end_headers()
        self.wfile.write(body or b"nope")

    def log_message(self, *a):
        pass


@pytest.fixture(scope="module")
def server():
    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_find_urls():
    assert find_urls("see https://a.com/x, and (http://b.org/y).") == ["https://a.com/x", "http://b.org/y"]


def test_blocks_private_and_bad_schemes():
    for bad in ["http://127.0.0.1/", "http://localhost:8080", "http://169.254.169.254/latest/meta-data", "file:///etc/passwd", "ftp://x.com"]:
        with pytest.raises(UnsafeURL):
            check_url(bad)


def test_crawl_extracts_text_and_follows_same_site_links(server):
    pages = crawl([server + "/"], max_pages=5, max_depth=1, allow_private=True)
    assert [p.title for p in pages] == ["Home", "Pricing"]
    assert "Welcome to Nimbus" in pages[0].text
    assert "MENU" not in pages[0].text and "evil" not in pages[0].text  # nav/script stripped
    assert "$9" in pages[1].text


def test_depth_zero_does_not_follow_links(server):
    assert len(crawl([server + "/"], max_depth=0, allow_private=True)) == 1


def test_redirects_are_followed_and_404_skipped(server):
    pages = crawl([server + "/redirect", server + "/missing"], allow_private=True)
    assert len(pages) == 1 and pages[0].title == "Pricing"


def test_crawl_refuses_private_hosts_by_default(server):
    assert crawl([server + "/"]) == []
