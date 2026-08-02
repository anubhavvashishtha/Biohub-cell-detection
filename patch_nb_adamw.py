import json

with open("C:/projects/Biohub cell detection/notebooks/kaggle_train.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and cell['source'] and 'torch.optim.Adam(' in ''.join(cell['source']):
        source = ''.join(cell['source'])
        
        # Replace Adam with AdamW
        source = source.replace(
            "optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)",
            "optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr, weight_decay=getattr(config.training, 'weight_decay', 1e-4))"
        )

        cell['source'] = source.splitlines(True)

with open("C:/projects/Biohub cell detection/notebooks/kaggle_train.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook successfully patched to use AdamW!")
