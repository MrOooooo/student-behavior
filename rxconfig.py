import os

import reflex as rx


def _split_origins(value: str) -> list[str]:
    return [origin.strip() for origin in value.split(',') if origin.strip()]


# External access needs the browser to call the public backend URL, not localhost.
API_URL = os.getenv('API_URL', 'http://localhost:8000').rstrip('/')
DEPLOY_URL = os.getenv('DEPLOY_URL', 'http://localhost:3000').rstrip('/')

_DEFAULT_ORIGINS = [
    'http://localhost:3000',
    'http://localhost:3001',
    'http://127.0.0.1:3000',
    'http://127.0.0.1:3001',
    DEPLOY_URL,
]

extra_origins = _split_origins(os.getenv('EXTRA_CORS_ALLOWED_ORIGINS', ''))

config = rx.Config(
    app_name='object_cheating',
    frontend_port=3000,
    backend_port=8000,
    api_url=API_URL,
    deploy_url=DEPLOY_URL,
    backend_host='0.0.0.0',
    cors_allowed_origins=list(dict.fromkeys([*_DEFAULT_ORIGINS, *extra_origins, '*'])),
)
