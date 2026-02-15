import requests

URL = "http://127.0.0.1:5000/predict"
FEATURE_COUNT = 13

def test_valid_input():
    sample = {"features": [0]*FEATURE_COUNT}
    r = requests.post(URL, json=sample)
    assert r.status_code == 200
    data = r.json()
    assert "prediction" in data
    assert "probability" in data

def test_invalid_input():
    sample = {"wrong": [1,2,3]}
    r = requests.post(URL, json=sample)
    assert r.status_code in (400, 422)

def test_multiple_calls():
    sample = {"features": [0]*FEATURE_COUNT}
    for _ in range(3):
        r = requests.post(URL, json=sample)
        assert r.status_code == 200

if __name__ == "__main__":
    test_valid_input()
    test_invalid_input()
    test_multiple_calls()
    print("All smoke tests passed.")
