#!/usr/bin/env python3
"""
MINCE — build the online spectra database
=========================================

Reads
  * MINCE32.csv       published catalog (one row per star)
  * fitstar.csv       list  star -> FITS file  (one row per FITS file)
  * a folder with the original FITS files (searched recursively)

Writes (inside --out, default ./spectra)
  * manifest.json           star -> spectra mapping + per-spectrum metadata
  * data/<id>.ov.f32        tiny overview (min/max envelope, ~24 KB) -> instant first plot
  * data/<id>.f32           full-resolution flux (float32) [+ float64 wavelengths if grid is not regular]
  * data/<id>.hdr.txt       FITS header(s) as plain text (shown on demand)
  * fits/<safe name>.fits   copies of the original FITS, renamed safely (spaces/colons removed)
                            -> upload these to GitHub Releases / Zenodo / the repo

Only stars that are in MINCE32.csv (= published) are included.
Name matching ignores spaces, underscores and case; the Gaia DR3 source_id
is used as a fallback / cross-check.

Usage
  python tools/build_spectra_db.py --catalog MINCE32.csv --fitstar fitstar.csv \
         --fits-dir /path/to/all/fits --out spectra \
         --download-base "https://github.com/StarCat8/mince_rep/releases/download/spectra-v1/"

  --download-base  where the original FITS will be downloadable from.
                   Default "spectra/fits/" (i.e. the FITS are committed in this repo).

Requires: numpy, astropy, pandas
"""
import argparse
import datetime as dt
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

OVERVIEW_BINS = 2048          # min/max envelope bins of the overview file
SPEED_OF_LIGHT = 299792.458


# ─────────────────────────────────────────────────────────────────────────────
# name matching
# ─────────────────────────────────────────────────────────────────────────────
def norm_key(name) -> str:
    """'BD+13  2153', 'BD+13_2153', 'bd+132153'  ->  'bd+132153'"""
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return ""
    return re.sub(r"[\s_]+", "", str(name)).lower()


def gaia_str(v) -> str:
    try:
        if pd.isna(v):
            return ""
        return str(int(v))
    except (TypeError, ValueError):
        return ""


def safe_filename(name: str) -> str:
    """File name safe for URLs and for GitHub Release assets (which mangle spaces/colons)."""
    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, "fits"
    stem = re.sub(r"[^A-Za-z0-9.\-]+", "_", stem).strip("_")
    return f"{stem}.{ext.lower()}"


def spectrum_id(fname: str) -> str:
    base = re.sub(r"[^A-Za-z0-9]+", "_", fname.rsplit(".", 1)[0]).strip("_")[:48]
    h = hashlib.sha1(fname.encode()).hexdigest()[:6]      # guarantees uniqueness
    return f"{base}_{h}"


# ─────────────────────────────────────────────────────────────────────────────
# FITS reading — handles the formats used by MINCE
#   1. 1D image + linear/log WCS  (FIES *_merge.fits, HARPS-N *_s1d_A.fits, IRAF)
#   2. binary table WAVE/FLUX     (ESO Phase-3 ADP.*.fits, e.g. UVES, FEROS)
# ─────────────────────────────────────────────────────────────────────────────
WAVE_COLS = ("WAVE", "WAVELENGTH", "LAMBDA", "LAM", "WAVE_AIR", "WAVE_VAC")
FLUX_COLS = ("FLUX", "FLUX_REDUCED", "FLUX_NORM", "FLUX_EL", "SPEC", "DATA")


def unit_factor(unit: str) -> float:
    """multiply by this to get Angstrom"""
    u = (unit or "").strip().lower()
    if u in ("nm", "nanometer", "nanometers", "nanometre"):
        return 10.0
    if u in ("um", "micron", "microns", "micrometer"):
        return 1e4
    if u in ("m", "meter", "metre"):
        return 1e10
    return 1.0                      # Angstrom / unknown


def wave_from_wcs(h, n):
    crval = h.get("CRVAL1")
    cdelt = h.get("CDELT1", h.get("CD1_1"))
    crpix = h.get("CRPIX1", 1.0)
    if crval is None or cdelt is None:
        raise ValueError("no CRVAL1/CDELT1 in header")
    w = crval + (np.arange(n) + 1.0 - crpix) * cdelt
    ctype = str(h.get("CTYPE1", "")).upper()
    if "LOG" in ctype or h.get("DC-FLAG", 0) == 1:
        w = 10.0 ** w if np.nanmax(w) < 10 else np.exp(w)
    return w * unit_factor(h.get("CUNIT1", ""))


def read_spectrum(path: Path):
    """returns wave [Å] (float64), flux (float32), instrument, header text"""
    with fits.open(path, memmap=False) as hl:
        header_txt = "\n\n".join(
            f"===== HDU {i}: {type(hdu).__name__} =====\n" + hdu.header.tostring(sep="\n").rstrip()
            for i, hdu in enumerate(hl))
        prim = hl[0].header
        instr = str(prim.get("INSTRUME", "") or "").strip()
        wave = flux = None

        # 1) image HDUs with 1D data (or 2D whose first row is the flux)
        for hdu in hl:
            if hdu.data is None or not isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU)):
                continue
            d = np.asarray(hdu.data)
            if d.ndim == 1:
                flux = d
            elif d.ndim >= 2 and d.shape[-1] > 100:
                flux = d.reshape(-1, d.shape[-1])[0]   # IRAF multispec: band 0 = spectrum
            else:
                continue
            wave = wave_from_wcs(hdu.header, flux.size)
            instr = instr or str(hdu.header.get("INSTRUME", "") or "").strip()
            break

        # 2) binary tables
        if flux is None:
            for hdu in hl:
                if not isinstance(hdu, fits.BinTableHDU) or hdu.data is None:
                    continue
                names = {c.upper(): c for c in hdu.columns.names}
                wc = next((names[c] for c in WAVE_COLS if c in names), None)
                fc = next((names[c] for c in FLUX_COLS if c in names), None)
                if not (wc and fc):
                    continue
                wave = np.asarray(hdu.data[wc], dtype=np.float64).ravel()
                flux = np.asarray(hdu.data[fc]).ravel()
                col = hdu.columns[wc]
                wave = wave * unit_factor(col.unit or "")
                instr = instr or str(hdu.header.get("INSTRUME", "") or "").strip()
                break

        if flux is None:
            raise ValueError("no 1D spectrum or WAVE/FLUX table found")

    wave = np.asarray(wave, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float32)
    good = np.isfinite(wave)
    wave, flux = wave[good], flux[good]
    if wave.size > 1 and np.any(np.diff(wave) < 0):
        o = np.argsort(wave, kind="stable")
        wave, flux = wave[o], flux[o]
    return wave, flux, instr, header_txt


INSTR_ALIASES = {"HARPN": "HARPS-N", "HARPS_N": "HARPS-N", "FIES": "FIES", "UVES": "UVES",
                 "ESPADONS": "ESPaDOnS", "SOPHIE": "SOPHIE", "FEROS": "FEROS", "MIKE": "MIKE", "VUES": "VUES"}


def guess_instrument(fname: str, instr: str) -> str:
    if instr:
        return INSTR_ALIASES.get(instr.upper(), instr)
    f = fname.upper()
    if f.startswith("FI"):
        return "FIES"
    if f.startswith("HARPN") or "S1D" in f:
        return "HARPS-N"
    if f.startswith("ADP"):
        return "ESO"
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# binary products
# ─────────────────────────────────────────────────────────────────────────────
def describe_grid(wave):
    """Return ('linear', w0, dw) | ('log', l0, dl) | ('array',)"""
    n = wave.size
    if n < 3:
        return ("array",)
    dw = (wave[-1] - wave[0]) / (n - 1)
    if np.max(np.abs(wave - (wave[0] + dw * np.arange(n)))) < 0.01 * abs(dw):
        return ("linear", float(wave[0]), float(dw))
    lw = np.log10(wave)
    dl = (lw[-1] - lw[0]) / (n - 1)
    if np.max(np.abs(lw - (lw[0] + dl * np.arange(n)))) < 0.01 * abs(dl):
        return ("log", float(lw[0]), float(dl))
    return ("array",)


def overview(wave, flux, nbins=OVERVIEW_BINS):
    """min/max envelope: interleaved [x, ymin, ymax] float32 per bin"""
    n = wave.size
    nbins = min(nbins, max(1, n // 2))
    edges = np.linspace(0, n, nbins + 1).astype(int)
    out = np.empty((nbins, 3), dtype=np.float32)
    for i in range(nbins):
        a, b = edges[i], edges[i + 1]
        seg = flux[a:b]
        fin = seg[np.isfinite(seg)]
        out[i, 0] = wave[(a + b - 1) // 2]
        out[i, 1] = fin.min() if fin.size else np.nan
        out[i, 2] = fin.max() if fin.size else np.nan
    return out


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default="MINCE32.csv")
    ap.add_argument("--fitstar", default="fitstar.csv")
    ap.add_argument("--fits-dir", required=True, help="folder containing the original FITS (searched recursively)")
    ap.add_argument("--out", default="spectra")
    ap.add_argument("--download-base", default="spectra/fits/",
                    help="URL prefix for FITS downloads (folder in repo, GitHub release, Zenodo...)")
    ap.add_argument("--no-copy", action="store_true", help="do not copy FITS into <out>/fits/")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    if not args.no_copy:
        (out / "fits").mkdir(parents=True, exist_ok=True)

    cat = pd.read_csv(args.catalog)
    fs = pd.read_csv(args.fitstar)

    # ── index of the published catalog ───────────────────────────────────────
    cat_by_key, cat_by_gaia = {}, {}
    for i, r in cat.iterrows():
        canon = norm_key(r.get("main_id_2")) or norm_key(r.get("target_id"))
        for col in ("main_id_2", "target_id"):
            k = norm_key(r.get(col))
            if k:
                cat_by_key.setdefault(k, canon)
        g = gaia_str(r.get("source_id_GAIA"))
        if g:
            cat_by_gaia.setdefault(g, canon)

    # ── index of the FITS files on disk ──────────────────────────────────────
    on_disk = {}
    for p in Path(args.fits_dir).rglob("*"):
        if p.is_file() and p.suffix.lower() in (".fits", ".fit", ".fts", ".gz"):
            on_disk.setdefault(p.name, p)

    stars, files = {}, {}
    report = {"matched_by_name": 0, "matched_by_gaia_only": 0, "gaia_conflict": [],
              "not_published": set(), "missing_on_disk": [], "read_errors": []}

    for _, r in fs.iterrows():
        fname = str(r["FITS"]).strip()
        k_name = next((cat_by_key[k] for k in (norm_key(r.get("main_id")), norm_key(r.get("NOME")))
                       if k in cat_by_key), None)
        k_gaia = cat_by_gaia.get(gaia_str(r.get("source_id")))
        if k_name and k_gaia and k_name != k_gaia:
            report["gaia_conflict"].append((r.get("main_id"), fname))
        key = k_name or k_gaia
        if key is None:
            report["not_published"].add(str(r.get("main_id")))
            continue
        report["matched_by_name" if k_name else "matched_by_gaia_only"] += 1

        src = on_disk.get(fname)
        if src is None:
            report["missing_on_disk"].append(fname)
            continue

        sid = spectrum_id(fname)
        if sid not in files:
            try:
                wave, flux, instr, hdr = read_spectrum(src)
            except Exception as e:                       # noqa: BLE001
                report["read_errors"].append((fname, str(e)))
                continue

            grid = describe_grid(wave)
            with open(out / "data" / f"{sid}.f32", "wb") as fh:
                fh.write(flux.astype("<f4").tobytes())
                if grid[0] == "array":
                    fh.write(wave.astype("<f8").tobytes())
            overview(wave, flux).astype("<f4").tofile(out / "data" / f"{sid}.ov.f32")
            (out / "data" / f"{sid}.hdr.txt").write_text(hdr, encoding="utf-8")

            safe = safe_filename(fname)
            if not args.no_copy:
                shutil.copy2(src, out / "fits" / safe)
            fin = flux[np.isfinite(flux)]
            files[sid] = {
                "fits": fname,
                "download": args.download_base + safe,
                "size": src.stat().st_size,
                "instrument": guess_instrument(fname, instr),
                "facility": None if pd.isna(r.get("facility")) else str(r.get("facility")),
                "n": int(flux.size),
                "wmin": round(float(wave[0]), 3),
                "wmax": round(float(wave[-1]), 3),
                "grid": grid[0],
                "g0": grid[1] if len(grid) > 1 else None,
                "dg": grid[2] if len(grid) > 1 else None,
                "median": float(np.median(fin)) if fin.size else 1.0,
                "bytes": int(flux.size * 4 + (wave.size * 8 if grid[0] == "array" else 0)),
            }
            print(f"  ✓ {fname:55s} {files[sid]['instrument']:10s} {flux.size:7d} px  "
                  f"{wave[0]:8.1f}–{wave[-1]:8.1f} Å  grid={grid[0]}")

        st = stars.setdefault(key, {"name": str(r.get("main_id")).strip(), "files": [],
                                    "rv": None if pd.isna(r.get("radial_velocity")) else float(r.get("radial_velocity"))})
        if sid not in st["files"]:
            st["files"].append(sid)

    manifest = {
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "key_rule": "lowercase, spaces and underscores removed",
        "stars": stars,
        "files": files,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")

    # ── report ───────────────────────────────────────────────────────────────
    published_keys = {norm_key(r.get("main_id_2")) or norm_key(r.get("target_id")) for _, r in cat.iterrows()}
    print("\n──────── summary ────────")
    print(f"published stars in catalog          : {len(published_keys)}")
    print(f"published stars with ≥1 spectrum    : {len(stars)}")
    print(f"spectra written                     : {len(files)}")
    print(f"fitstar rows matched by name        : {report['matched_by_name']}")
    print(f"fitstar rows matched only by Gaia id: {report['matched_by_gaia_only']}")
    print(f"fitstar stars not (yet) published   : {len(report['not_published'])}  (skipped)")
    if report["gaia_conflict"]:
        print("⚠ name/Gaia id disagree for:", report["gaia_conflict"])
    if report["missing_on_disk"]:
        print(f"⚠ {len(report['missing_on_disk'])} FITS listed but not found in --fits-dir:")
        for f in report["missing_on_disk"]:
            print("    ", f)
    if report["read_errors"]:
        print("⚠ unreadable FITS:")
        for f, e in report["read_errors"]:
            print("    ", f, "→", e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
