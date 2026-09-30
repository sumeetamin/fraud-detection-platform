# Verification record

Verified locally on Windows, Python 3.14, 30 September 2026.

- Training command completed and produced the committed evaluation reports.
- `python -m pytest -q`: **5 passed**.
- Streamlit dashboard rendered and its main controls were exercised through AppTest.
- Running dashboard inspected in a browser; screenshot is in `docs/dashboard.png`.
- FastAPI integration tests passed.
- Docker engine was unavailable: container build/run is not verified.
- Hosted deployment has not been performed. GitHub publication is source-code hosting, not application hosting.
- One upstream Starlette/httpx deprecation warning appeared; it did not fail the tests.

The document project additionally passed one clean scanned-image OCR smoke test. The fraud project additionally served 240 local HTTP requests and produced drift alerts under a simulated incident. These are bounded smoke tests, not production reliability claims.

`requirements-lock.txt` records the core environment before optional OCR installation. Use `requirements-ocr.txt` for the optional OCR extras. The base installation and CI do not need those extras.
