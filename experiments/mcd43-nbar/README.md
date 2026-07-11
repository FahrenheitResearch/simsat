# MCD43 regional NBAR availability probe

This is the isolated Stage-D0 discovery gate for a possible radiometric land
surface boundary. It does not change SimSat production code, assets, defaults,
CLI/Python/Studio controls, or rendering.

The probe uses only Python's standard library and official NASA endpoints. It
separates three states that must not be conflated:

1. the exact requested nominal date;
2. the latest nominal date currently published; and
3. an explicitly labeled prior-year same-calendar-day control.

A control is never substituted for the target. Discovery is read-only;
downloading is a separate command.

## Captured result: A2026191

The committed capture was queried at `2026-07-11T19:41:33Z` for
`2026-07-10` (`A2026191`) and WGS84 bounding box
`-71.5,42.5,-65.5,48.0`.

- Exact MCD43A4 V061 target granules: **0**.
- Exact MCD43A2 V061 target granules: **0**.
- Latest nominal date in both collections: `A2026182` (`2026-07-01`),
  nine nominal days behind the target.
- Regional tiles evidenced by the current and control inventories:
  `h12v04` and `h13v04`.
- Prior-year `A2025191` has both products and both tiles, but is retained only
  as an availability/latency control.
- A one-byte ranged `GET` to each of the four control HDF URLs returned
  `HTTP 401`; the metadata are anonymously searchable, but the source HDF
  objects require Earthdata authorization. A `HEAD` request can follow to a
  signed-looking URL and is therefore not accepted as proof of data access.

The official MCD43A4 catalog says that each daily product uses a 16-day
Terra+Aqua window and is weighted to the ninth day identified in the filename.
The 2025 same-day control was produced 14--15 calendar days after its nominal
date. The exact 2026 absence one day after the nominal date is therefore a
measured latency state, not evidence that the collection lacks the region.

No HDF was downloaded and no float/QA crop was produced. That is intentional:
the exact target is not published and unauthenticated data access is blocked.

## Run

Offline checks:

```powershell
python experiments/mcd43-nbar/mcd43_probe.py self-check
```

Strict live plan. It writes the plan before returning exit code 2 when the
target is unavailable:

```powershell
python experiments/mcd43-nbar/mcd43_probe.py plan `
  --target-date 2026-07-10 `
  --control-year 2025 `
  --probe-control-access `
  --output mcd43-a2026191-plan.json
```

For polling automation that handles the status inside JSON, add
`--allow-target-unavailable`. The target still remains explicitly
`"status": "unavailable"`; the flag changes only the process exit code.

The plan writer is no-clobber. Use a new filename for a later observation.

## Explicit download

The default source is the exact target. If that block is unavailable or
partial, the command refuses to download anything:

```powershell
python experiments/mcd43-nbar/mcd43_probe.py download mcd43-plan.json `
  --output-dir C:\path\to\mcd43
```

Downloading a prior-year control requires the conspicuous source selection:

With `EARTHDATA_TOKEN` populated through the user's credential workflow:

```powershell
python experiments/mcd43-nbar/mcd43_probe.py download mcd43-plan.json `
  --source control `
  --output-dir C:\path\to\mcd43-control `
  --token-env EARTHDATA_TOKEN
```

Do not put a token in a plan, command argument, repository file, or log. The
script reads only the named environment variable and records its name, never
its value.

For every selected HDF the downloader:

- permits only the official LP DAAC HTTPS host and `.hdf` paths from the plan;
- refuses existing targets, provenance files, and stale partial files;
- streams to a temporary file;
- verifies the NASA-published byte count and checksum from native CMR
  metadata (current granules may publish MD5; the 2025 controls publish
  SHA-256);
- computes SHA-256 independently for every download; and
- renames the file into place only after validation, then writes no-clobber
  provenance.

Product and tile filters are available as repeatable `--product` and `--tile`
arguments. Omitting them selects the minimum complete regional set named by
the chosen plan block.

## Float/QA crop gate after publication and authorization

Once exact `A2026191` A4+A2 granules for both tiles exist and can be
authenticated, the D0 crop must:

1. download exactly four HDF files: A4 and A2 for `h12v04` and `h13v04`;
2. preserve NASA checksums plus locally computed SHA-256;
3. read the A4 NBAR red/green/blue candidates from MODIS bands 1/4/3 using
   each HDF dataset's documented scale, fill, and valid-range attributes;
4. combine the corresponding A4 mandatory flags with the band-specific A2
   inversion quality, initially retaining only the highest-quality full
   inversions;
5. reproject only the requested geographic crop while retaining float linear
   reflectance and an explicit coverage/quality mask; and
6. create display PNGs only as derived previews, never as the scientific
   intermediate.

Missing or low-quality pixels must remain masked. The D0 experiment may not
paint gaps or fall back silently to the 2025 control or BMNG.

## Captured evidence

`fixtures/mcd43-a2026191-maine-probe-20260711.json` records collection and
granule concept IDs, revisions, exact native-metadata URLs, granule URLs,
byte counts, checksums, temporal/production metadata, access probe responses,
and the latest-day calculation. It contains no credentials or signed query
URLs.

## Official NASA sources

- [MCD43A4 V061 catalog](https://www.earthdata.nasa.gov/data/catalog/lpcloud-mcd43a4-061)
- [MCD43A2 V061 catalog](https://www.earthdata.nasa.gov/data/catalog/lpcloud-mcd43a2-061)
- [NASA CMR Search API](https://cmr.earthdata.nasa.gov/search/site/docs/search/api.html)
- [NASA LP DAAC cloud data-access examples](https://git.earthdata.nasa.gov/projects/LPDUR/repos/lpdaac_cloud_data_access/browse)
- [NASA Earthdata Login](https://urs.earthdata.nasa.gov/)
