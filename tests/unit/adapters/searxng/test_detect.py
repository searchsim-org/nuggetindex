from nuggetindex.adapters.searxng.detect import CaptchaDetector

CF_CHALLENGE_HTML = '<!DOCTYPE html><html><head><title>Just a moment...</title></head><body><div class="cf-browser-verification"></div></body></html>'
GOOGLE_SORRY_HTML = '<html><body><form action="/sorry/index"><h1>Our systems have detected unusual traffic</h1></form></body></html>'
DDG_ANOMALY_HTML = '<html><body><div class="anomaly-modal__title">Unfortunately, bots use DuckDuckGo too</div></body></html>'
BING_CAPTCHA_HTML = '<html><body><form id="captchaform" action="/fd/ls/CAPT"><input id="captchaAnswer"/></form></body></html>'


def test_detects_cloudflare_challenge():
    d = CaptchaDetector()
    result = d.classify(status_code=503, body=CF_CHALLENGE_HTML, headers={})
    assert result.is_captcha is True
    assert result.category == "cloudflare"


def test_detects_google_sorry():
    d = CaptchaDetector()
    assert (
        d.classify(status_code=200, body=GOOGLE_SORRY_HTML, headers={}).category == "google_sorry"
    )


def test_detects_duckduckgo_anomaly():
    d = CaptchaDetector()
    assert d.classify(status_code=200, body=DDG_ANOMALY_HTML, headers={}).category == "ddg_anomaly"


def test_detects_bing_captcha():
    d = CaptchaDetector()
    assert (
        d.classify(status_code=200, body=BING_CAPTCHA_HTML, headers={}).category == "bing_captcha"
    )


def test_detects_http_429_as_rate_limit():
    d = CaptchaDetector()
    result = d.classify(status_code=429, body="Too Many Requests", headers={})
    assert result.is_captcha is True
    assert result.category == "rate_limit"


def test_detects_silent_empty_searxng_response():
    d = CaptchaDetector()
    body = '{"results": [], "answers": [], "infoboxes": []}'
    result = d.classify(
        status_code=200,
        body=body,
        headers={"content-type": "application/json"},
        searxng_empty=True,
        searxng_engines_failed=True,
    )
    assert result.is_captcha is True
    assert result.category == "searxng_silent_empty"


def test_honest_empty_result_is_not_captcha():
    d = CaptchaDetector()
    body = '{"results": [], "answers": [], "infoboxes": []}'
    result = d.classify(
        status_code=200,
        body=body,
        headers={},
        searxng_empty=True,
        searxng_engines_failed=False,
    )
    assert result.is_captcha is False


def test_clean_response_is_not_captcha():
    d = CaptchaDetector()
    body = '{"results": [{"url": "https://ex.com", "title": "t", "content": "c"}]}'
    result = d.classify(status_code=200, body=body, headers={})
    assert result.is_captcha is False
    assert result.category == "ok"
