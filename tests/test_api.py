import pytest
from src.api import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home(client):
    res = client.get('/')
    assert res.status_code == 200
    assert res.json['status'] == 'running'

def test_health(client):
    res = client.get('/health')
    assert res.status_code == 200
    assert res.json['status'] == 'healthy'
    assert res.json['model_loaded'] is True

def test_predict(client):
    res = client.post('/predict', json={'features': [5.1, 3.5, 1.4, 0.2]})
    assert res.status_code == 200
    assert 'prediction' in res.json
    assert res.json['prediction'] in [0, 1, 2]
