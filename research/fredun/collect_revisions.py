"""Retrieve revised quarterly originals and their independent release."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json

import collect_sources as sc

URLS = {
    "q2_revised_1904.pdf":"https://www.bseindia.com/xml-data/corpfiling/AttachLive/ed278e52-0463-4878-a32b-62455d8ef5d5.pdf",
    "q2_revised_1911.pdf":"https://www.bseindia.com/xml-data/corpfiling/AttachLive/75e1837c-1508-4408-b066-cea4a312765b.pdf",
    "q2_results_exchange.pdf":"https://www.bseindia.com/xml-data/corpfiling/AttachLive/ca52aef5-a2d1-49c2-a58b-a830e4767771.pdf",
    "q4_revised.pdf":"https://www.bseindia.com/xml-data/corpfiling/AttachHis/644e75b2-e04f-49bb-849e-1da38d6f3aae.pdf",
    "q4_revised_live.pdf":"https://www.bseindia.com/xml-data/corpfiling/AttachLive/644e75b2-e04f-49bb-849e-1da38d6f3aae.pdf",
    "dhan_history.html":"https://dhan.co/stocks/fredun-pharmaceuticals-ltd-historical-price/",
}
sc.OUT = Path("fredun_revised_filings")
sc.OUT.mkdir(exist_ok=True)
with ThreadPoolExecutor(max_workers=3) as pool:
    for label, data in zip(URLS, pool.map(lambda pair: sc.cached(*pair), URLS.items())):
        print(label, len(data or b""), (data or b"").startswith(b"%PDF"), flush=True)
(sc.OUT / "source_receipts.json").write_text(json.dumps(sc.RECEIPTS,indent=2,sort_keys=True))
