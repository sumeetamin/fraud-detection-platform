from pathlib import Path
from streamlit.testing.v1 import AppTest

def test_score_and_incident_render():
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py'),default_timeout=30).run()
    assert not app.exception
    app.button[0].click().run()
    app.toggle[0].set_value(True).run()
    assert not app.exception
