import pyarrow.parquet as pq
from config import settings

target_id = "public_pdf/prism/2023년 충남 공공기관(장) 경영평가 _ 결과보고서(정보공개)_666"
table = pq.read_table(settings.corpus_parquet, columns=['id'])
ids = table.column('id').to_pylist()
matches = [i for i, v in enumerate(ids) if v == target_id]
print("exact match rows:", matches)
if not matches:
    # try without page suffix match, sample similar ids
    cand = [v for v in ids if '충남 공공기관' in v][:5]
    print(cand)
