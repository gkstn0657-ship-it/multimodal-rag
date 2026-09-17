import pyarrow.parquet as pq
from config import settings

table = pq.read_table(settings.corpus_parquet, columns=['image'])
row = table.column('image')[32148].as_py()
img_bytes = row.get('bytes') if isinstance(row, dict) else bytes(row)
with open('scratch_page666.png', 'wb') as f:
    f.write(img_bytes)
print("saved", len(img_bytes), "bytes")
