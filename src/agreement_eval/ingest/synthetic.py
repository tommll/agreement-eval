"""Deterministic synthetic receipts.

Purpose: let the whole pipeline (extract -> judge -> evaluate -> report) run
end-to-end offline, with no API key and no download, so the analysis code can
be exercised and tested. Paired with providers.MockProvider it also makes the
headline effect reproducible: configs whose errors are *correlated* agree with
each other while being wrong together.
"""

from __future__ import annotations

import random
from typing import List, Optional

from ..documents import Document

COMPANIES = [
    ("OJC MARKETING SDN BHD", "NO 2 & 4, JALAN BAYU 4, BANDAR SERI ALAM, 81750 MASAI, JOHOR"),
    ("PERNIAGAAN ZHENG HUI", "NO.59 JALAN PERMAS 9/5, BANDAR BARU PERMAS JAYA, 81750 JOHOR BAHRU"),
    ("GARDENIA BAKERIES (KL) SDN BHD", "LOT 3, JALAN PELABUR 23/1, 40300 SHAH ALAM, SELANGOR"),
    ("SANYU STATIONERY SHOP", "NO. 31G&33G, JALAN SETIA INDAH X, U13/X, 40170 SETIA ALAM"),
    ("99 SPEED MART S/B", "LOT P.T. 2811, JALAN ANGSANA, TAMAN MUTIARA RINI, 81300 SKUDAI"),
    ("KEDAI PAPAN YEW CHUAN", "LOT 276 JALAN BANTING, 43800 DENGKIL, SELANGOR"),
]
ITEMS = [
    ("MINERAL WATER 500ML", 1.50), ("A4 PAPER REAM", 12.90), ("BALLPEN BLUE", 2.30),
    ("TEH TARIK", 3.20), ("NASI LEMAK", 6.50), ("USB CABLE 1M", 18.00),
    ("DETERGENT 2KG", 21.40), ("BREAD LOAF", 4.80),
]


def make_synthetic(n: int = 60, seed: int = 7) -> List[Document]:
    rng = random.Random(seed)
    docs: List[Document] = []
    for i in range(n):
        company, address = COMPANIES[i % len(COMPANIES)]
        day, month, year = rng.randint(1, 28), rng.randint(1, 12), rng.choice([2018, 2019])
        date = f"{day:02d}/{month:02d}/{year}"
        lines = rng.sample(ITEMS, k=rng.randint(1, 4))
        subtotal = round(sum(price * 1 for _, price in lines), 2)
        has_tax = rng.random() < 0.55        # ~45% of receipts have no tax line
        tax = round(subtotal * 0.06, 2) if has_tax else None
        total = round(subtotal + (tax or 0), 2)
        has_phone = rng.random() < 0.6
        phone = f"0{rng.randint(3,9)}-{rng.randint(100,999)} {rng.randint(1000,9999)}" if has_phone else None

        body = [company, address]
        if phone:
            body.append(f"TEL: {phone}")
        body += ["TAX INVOICE", f"DATE: {date} {rng.randint(9,20)}:{rng.randint(10,59)}",
                 f"INVOICE NO: {rng.randint(10000,99999)}", "-" * 24]
        body += [f"{name:<24}{price:>8.2f}" for name, price in lines]
        body += ["-" * 24, f"{'SUBTOTAL':<24}{subtotal:>8.2f}"]
        if tax is not None:
            body.append(f"{'GST 6%':<24}{tax:>8.2f}")
        body += [f"{'TOTAL':<24}{total:>8.2f}", "THANK YOU. PLEASE COME AGAIN"]

        docs.append(
            Document(
                # The seed is part of the identity: a different seed is a
                # different corpus, not a mutation of the same documents.
                doc_id=f"synthetic/s{seed}/{i:04d}",
                dataset="synthetic",
                split="demo",
                gold={
                    "company": company,
                    "date": date,
                    "address": address,
                    "total": f"{total:.2f}",
                    "tax": f"{tax:.2f}" if tax is not None else None,
                    "phone": phone,
                },
                ocr_text="\n".join(body),
                meta={"synthetic": True, "seed": seed},
            )
        )
    return docs
