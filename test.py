import requests

url = "http://localhost:9696/predict/" # for local testing
# url = "http://localhost:8080/2015-03-31/functions/function/invocations" # for AWS Lambda
data = {"url": "https://storage.googleapis.com/kagglesdsdata/datasets/534640/5468571/test/AFRICAN%20CROWNED%20CRANE/1.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20231226%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20231226T050028Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=7a161e5f38cdca1a22757b9d75a41205257cb574919481a903f5ecaf8f2e4c083e9584767b0dc7b7c5e2a64e1b3f27d3aaea3e0cbbec38f0ea372f15635e21b85982bb29e8449e5acc676ec393f9a058e715219f4885e99c6e4feb08cb849bde7181b170b44c8584d246ae5f85833f5f2d9ee2048a5e475d535dad0dfef96ccfa42dff0a0ead93634da47f746cfd4d4e6ca74afde6a6f2f92ca1be88aa579d0e738bb3d8c25eb8fe765e775f12bc0fe84aaace72fd6756947238edec02271ba2c6f9fa3e237bfb66857fa380dd9bd8b74ea46bb41ec03e864467ab3fe5ad541512ec63f3786d672004f9618783df3dbf173b3266b1b9eca2b35b28829970f70b"}
result = requests.post(url, json=data)
print(result.text) # for local testing
# print(result.json()) # for AWS Lambda