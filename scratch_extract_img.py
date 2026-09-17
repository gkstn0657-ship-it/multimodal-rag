import pyarrow.parquet as pq
from config import settings

target_id = "public_pdf/prism/2023년 충남 공공기관(장) 경영평가 _ 결과보고서(정보공개)_666"
table = pq.read_table(settings.corpus_parquet)
ids = table.column('id').to_pylist() if 'id' in table.column_names else None
print("columns:", table.column_names)
