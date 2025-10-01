import requests
response = requests.get('https://datasets-server.huggingface.co/first-rows?dataset=Vadera007%2FFormula_1_Dataset&config=default&split=train', timeout=10)
data = response.json()
print(f'✓ Dataset accessible: {len(data.get("rows", []))} rows available')
print(f'✓ First driver: {data["rows"][0]["row"]["Driver"]}')
print(f'✓ First team: {data["rows"][0]["row"]["Team"]}')
