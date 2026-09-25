"""Upload small run summaries to the prototype. Never upload model weights or data."""
import json
import os
import time
from urllib.parse import urlparse
from urllib.request import Request, HTTPRedirectHandler, build_opener


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('Results endpoint redirected; refusing to forward credentials')


_last_upload = {}


def publish_report(report, base_url=None, token=None):
    base_url = base_url or os.environ.get('MATMED_API_URL')
    token = token or os.environ.get('MATMED_WRITE_TOKEN')
    if not base_url:
        return {'uploaded': False, 'reason': 'MATMED_API_URL not configured'}
    parsed = urlparse(base_url)
    if parsed.scheme != 'https' and not (parsed.scheme == 'http' and parsed.hostname in ('127.0.0.1', 'localhost')):
        raise ValueError("Connector requires HTTPS (except local testing)")
    if not token:
        raise ValueError("MATMED_WRITE_TOKEN is required")
    if parsed.username or parsed.password:
        raise ValueError('Do not embed credentials in the API URL')
    # KV permits at most one write per second to the same key.
    key = (base_url, report['run_id'])
    delay = 1.1 - (time.monotonic() - _last_upload.get(key, 0))
    if delay > 0:
        time.sleep(delay)
    body = json.dumps(report, allow_nan=False).encode()
    request = Request(base_url.rstrip('/') + '/api/runs/' + report['run_id'], data=body,
                      headers={'Content-Type': 'application/json',
                               'Authorization': 'Bearer ' + token}, method='PUT')
    try:
        with build_opener(NoRedirects()).open(request, timeout=20) as response:
            return json.load(response)
    finally:
        _last_upload[key] = time.monotonic()
