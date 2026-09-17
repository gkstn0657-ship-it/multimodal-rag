from servers.retrieve import search, rerank
from servers.answer import generate_from_chunks
from servers.vlm_client import VLMClient

q = "요새 국제 협력 관계가 어떻게 되는지 요약해줘"
cands = search(q)
chunks = rerank(q, cands)
out = ["=== top5 ==="]
for c in chunks:
    out.append(f"{c.page_id} route={c.route} score={c.score:.3f}")
    out.append("  " + c.text[:150].replace("\n"," | "))

r = generate_from_chunks(q, chunks, client=VLMClient())
out.append("=== answer ===")
out.append(r.answer)

with open("scratch_diag_q_out.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
