import subprocess
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pyreadstat

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "CY09_MS_STU_PUF.parquet"
CHUNK = 50_000


def find_sav():
    sav = sorted(RAW.glob("CY09_MS_STU_PUF*.sav"))
    if sav:
        return sav[0]
    archive = RAW / "CY09_MS_STU_PUF.zip"
    with zipfile.ZipFile(archive) as z:
        member = next(n for n in z.namelist() if n.endswith(".sav"))
    subprocess.run(["unzip", "-o", "-j", str(archive), member, "-d", str(RAW)], check=True)
    return RAW / Path(member).name


def main():
    sav = find_sav()
    _, meta = pyreadstat.read_sav(sav, metadataonly=True)
    print(f"{sav.name}: {meta.number_rows:,} rows x {meta.number_columns} columns")
    schema = pa.schema([
        (name, pa.large_string() if vtype == "string" else pa.float64())
        for name, vtype in meta.readstat_variable_types.items()
    ])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    rows = 0
    for offset in range(0, meta.number_rows, CHUNK):
        chunk, _ = pyreadstat.read_sav(sav, row_offset=offset, row_limit=CHUNK)
        table = pa.Table.from_pandas(chunk, schema=schema, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(OUT, table.schema)
        writer.write_table(table)
        rows += len(chunk)
        print(f"  {rows:,} / {meta.number_rows:,} rows", end="\r")
    writer.close()
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
