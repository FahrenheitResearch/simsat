#!/usr/bin/env python3
"""Strict NASA CMR availability and download planner for a regional MCD43 D0.

The default action is read-only discovery.  Downloads are a separate explicit
subcommand, never overwrite files, verify the NASA-published checksum from the
granule's native CMR metadata, and compute SHA-256 before committing a file.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "simsat.mcd43-regional-probe.v1"
CMR_ROOT = "https://cmr.earthdata.nasa.gov"
CMR_COLLECTIONS = f"{CMR_ROOT}/search/collections.umm_json"
CMR_GRANULES = f"{CMR_ROOT}/search/granules.umm_json"
CMR_CONCEPT = f"{CMR_ROOT}/search/concepts"
USER_AGENT = "SimSat-MCD43-D0/1.0 (+https://github.com/FahrenheitResearch/simsat)"
PAGE_SIZE = 2000
ALLOWED_DOWNLOAD_HOSTS = {"data.lpdaac.earthdatacloud.nasa.gov"}
DEFAULT_BBOX = (-71.5, 42.5, -65.5, 48.0)
DEFAULT_TARGET = dt.date(2026, 7, 10)

PRODUCTS = (
    {
        "short_name": "MCD43A4",
        "version": "061",
        "doi": "10.5067/MODIS/MCD43A4.061",
        "catalog_url":
            "https://www.earthdata.nasa.gov/data/catalog/lpcloud-mcd43a4-061",
    },
    {
        "short_name": "MCD43A2",
        "version": "061",
        "doi": "10.5067/MODIS/MCD43A2.061",
        "catalog_url":
            "https://www.earthdata.nasa.gov/data/catalog/lpcloud-mcd43a2-061",
    },
)

GRANULE_RE = re.compile(
    r"^(MCD43A[24])\.A(\d{4})(\d{3})\.(h\d{2}v\d{2})\.061\.(\d{13})$"
)


class ProbeError(RuntimeError):
    """Expected, user-actionable probe failure."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def generator_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def date_to_tag(value: dt.date) -> str:
    return f"A{value.year:04d}{value.timetuple().tm_yday:03d}"


def tag_to_date(tag: str) -> dt.date:
    if not re.fullmatch(r"A\d{7}", tag):
        raise ProbeError(f"invalid nominal date tag: {tag!r}")
    year = int(tag[1:5])
    doy = int(tag[5:8])
    value = dt.date(year, 1, 1) + dt.timedelta(days=doy - 1)
    if value.year != year:
        raise ProbeError(f"day-of-year is outside {year}: {tag}")
    return value


def parse_granule_name(name: str) -> dict[str, str]:
    match = GRANULE_RE.fullmatch(name)
    if not match:
        raise ProbeError(f"unexpected MCD43 V061 granule name: {name}")
    product, year, doy, tile, production = match.groups()
    tag = f"A{year}{doy}"
    tag_to_date(tag)
    return {
        "product": product,
        "nominal_tag": tag,
        "nominal_date": tag_to_date(tag).isoformat(),
        "tile": tile,
        "production_tag": production,
    }


def bbox_text(bbox: tuple[float, float, float, float]) -> str:
    return ",".join(f"{part:g}" for part in bbox)


def sanitized_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def _request(
    url: str,
    *,
    accept: str | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 45.0,
) -> urllib.response.addinfourl:
    merged = {"User-Agent": USER_AGENT}
    if accept:
        merged["Accept"] = accept
    if headers:
        merged.update(headers)
    request = urllib.request.Request(url, headers=merged)
    return urllib.request.urlopen(request, timeout=timeout)


def _cmr_url(endpoint: str, params: dict[str, Any]) -> str:
    return f"{endpoint}?{urllib.parse.urlencode(params, doseq=True)}"


def cmr_json(endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    url = _cmr_url(endpoint, params)
    with _request(
        url,
        accept="application/vnd.nasa.cmr.umm_results+json",
    ) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ProbeError(f"CMR returned non-object JSON for {sanitized_url(url)}")
    return payload


def query_collection(product: dict[str, str]) -> dict[str, Any]:
    payload = cmr_json(
        CMR_COLLECTIONS,
        {
            "short_name": product["short_name"],
            "version": product["version"],
            "page_size": 100,
        },
    )
    items = payload.get("items", [])
    if len(items) != 1:
        raise ProbeError(
            f"expected one {product['short_name']} V{product['version']} collection; "
            f"CMR returned {len(items)}"
        )
    item = items[0]
    meta = item["meta"]
    umm = item["umm"]
    if umm.get("ShortName") != product["short_name"]:
        raise ProbeError("CMR collection short name did not round-trip")
    return {
        **product,
        "concept_id": meta["concept-id"],
        "provider_id": meta["provider-id"],
        "revision_id": meta["revision-id"],
        "revision_date": meta["revision-date"],
        "entry_title": umm.get("EntryTitle"),
    }


def query_granules(
    collection_id: str,
    pattern: str,
    bbox: tuple[float, float, float, float],
) -> list[dict[str, Any]]:
    page_num = 1
    all_items: list[dict[str, Any]] = []
    expected_hits: int | None = None
    while True:
        payload = cmr_json(
            CMR_GRANULES,
            {
                "collection_concept_id": collection_id,
                "granule_ur[]": pattern,
                "options[granule_ur][pattern]": "true",
                "bounding_box": bbox_text(bbox),
                "page_size": PAGE_SIZE,
                "page_num": page_num,
            },
        )
        hits = int(payload.get("hits", 0))
        if expected_hits is None:
            expected_hits = hits
        elif hits != expected_hits:
            raise ProbeError("CMR hit count changed while paging; rerun the probe")
        items = payload.get("items", [])
        if not isinstance(items, list):
            raise ProbeError("CMR granule response has no item list")
        all_items.extend(items)
        if len(all_items) >= hits:
            break
        if not items:
            raise ProbeError("CMR paging ended before the advertised hit count")
        page_num += 1
    if len(all_items) != (expected_hits or 0):
        raise ProbeError(
            f"CMR pagination mismatch: got {len(all_items)}, expected {expected_hits}"
        )
    return all_items


def native_echo10_detail(concept_id: str, revision_id: int, name: str) -> dict[str, Any]:
    url = f"{CMR_CONCEPT}/{urllib.parse.quote(concept_id)}/{revision_id}"
    with _request(url, accept="application/echo10+xml") as response:
        payload = response.read()
    return parse_native_echo10(payload, url, name)


def parse_native_echo10(payload: bytes | str, url: str, name: str) -> dict[str, Any]:
    root = ET.fromstring(payload)

    if root.findtext("./GranuleUR") != name:
        raise ProbeError(f"native CMR metadata did not round-trip {name}")
    wanted = f"{name}.hdf"
    match: ET.Element | None = None
    for candidate in root.findall("./DataGranule/AdditionalFile"):
        if candidate.findtext("./Name") == wanted:
            match = candidate
            break
    if match is None:
        raise ProbeError(f"native CMR metadata has no HDF checksum entry for {name}")
    algorithm = (match.findtext("./Checksum/Algorithm") or "").upper().replace("-", "")
    checksum = (match.findtext("./Checksum/Value") or "").lower()
    checksum_lengths = {"MD5": 32, "SHA256": 64}
    if algorithm not in checksum_lengths or not re.fullmatch(
        rf"[0-9a-f]{{{checksum_lengths.get(algorithm, 0)}}}", checksum
    ):
        raise ProbeError(f"{name} has no supported NASA-published checksum")
    size_text = match.findtext("./SizeInBytes")
    if not size_text or int(size_text) <= 0:
        raise ProbeError(f"{name} has no valid NASA-published byte size")
    return {
        "native_metadata_url": url,
        "production_datetime": root.findtext("./DataGranule/ProductionDateTime"),
        "local_version_id": root.findtext("./DataGranule/LocalVersionId"),
        "hdf_name": wanted,
        "hdf_size_bytes": int(size_text),
        "hdf_checksum": {"algorithm": algorithm, "value": checksum},
    }


def first_url(related: Iterable[dict[str, Any]], predicate: Any) -> str | None:
    for entry in related:
        url = entry.get("URL")
        if isinstance(url, str) and predicate(entry, url):
            return url
    return None


def enrich_granule(item: dict[str, Any]) -> dict[str, Any]:
    meta = item["meta"]
    umm = item["umm"]
    name = umm["GranuleUR"]
    parsed = parse_granule_name(name)
    related = umm.get("RelatedUrls", [])
    http_url = first_url(
        related,
        lambda entry, url: entry.get("Type") == "GET DATA" and url.endswith(".hdf"),
    )
    s3_url = first_url(
        related,
        lambda entry, url: entry.get("Type") == "GET DATA VIA DIRECT ACCESS"
        and url.startswith("s3://")
        and url.endswith(".hdf"),
    )
    opendap_url = first_url(
        related,
        lambda entry, url: entry.get("Type") == "USE SERVICE API"
        and entry.get("Subtype") == "OPENDAP DATA",
    )
    if not http_url:
        raise ProbeError(f"CMR did not publish an HTTPS HDF URL for {name}")
    native = native_echo10_detail(meta["concept-id"], meta["revision-id"], name)
    production_text = native["production_datetime"]
    if not production_text:
        raise ProbeError(f"native CMR metadata has no production time for {name}")
    production_date = dt.datetime.fromisoformat(
        production_text.replace("Z", "+00:00")
    ).date()
    nominal_date = dt.date.fromisoformat(parsed["nominal_date"])
    temporal = umm.get("TemporalExtent", {}).get("RangeDateTime", {})
    return {
        **parsed,
        "granule_ur": name,
        "concept_id": meta["concept-id"],
        "revision_id": meta["revision-id"],
        "cmr_revision_date": meta.get("revision-date"),
        "temporal_begin": temporal.get("BeginningDateTime"),
        "temporal_end": temporal.get("EndingDateTime"),
        "https_url": http_url,
        "s3_url": s3_url,
        "opendap_url": opendap_url,
        "production_lag_days_from_nominal": (production_date - nominal_date).days,
        **native,
    }


def exact_granules(
    collection: dict[str, Any],
    nominal_date: dt.date,
    bbox: tuple[float, float, float, float],
) -> list[dict[str, Any]]:
    pattern = f"{collection['short_name']}.{date_to_tag(nominal_date)}.*"
    return [
        enrich_granule(item)
        for item in query_granules(collection["concept_id"], pattern, bbox)
    ]


def latest_at_or_before(
    collection: dict[str, Any],
    nominal_date: dt.date,
    bbox: tuple[float, float, float, float],
) -> dict[str, Any] | None:
    pattern = f"{collection['short_name']}.A{nominal_date.year:04d}*"
    candidates = query_granules(collection["concept_id"], pattern, bbox)
    dated: list[tuple[dt.date, dict[str, Any]]] = []
    for item in candidates:
        parsed = parse_granule_name(item["umm"]["GranuleUR"])
        value = dt.date.fromisoformat(parsed["nominal_date"])
        if value <= nominal_date:
            dated.append((value, item))
    if not dated:
        return None
    latest_date = max(value for value, _ in dated)
    items = [enrich_granule(item) for value, item in dated if value == latest_date]
    return {
        "nominal_date": latest_date.isoformat(),
        "nominal_tag": date_to_tag(latest_date),
        "age_days_at_target": (nominal_date - latest_date).days,
        "granules": sorted(items, key=lambda entry: entry["tile"]),
    }


def access_probe(url: str) -> dict[str, Any]:
    """Probe a protected object with one ranged GET, not a misleading HEAD."""
    headers = {"Range": "bytes=0-0"}
    result: dict[str, Any] = {
        "method": "GET",
        "range": "bytes=0-0",
        "requested_url": url,
    }
    try:
        with _request(url, headers=headers) as response:
            sample = response.read(1)
            content_type = response.headers.get_content_type()
            status = response.status
            result.update(
                {
                    "http_status": status,
                    "final_url_without_query": sanitized_url(response.geturl()),
                    "content_type": content_type,
                    "sample_bytes_read": len(sample),
                    "accessible_without_credentials": status in (200, 206)
                    and content_type not in {"text/html", "application/json"}
                    and len(sample) == 1,
                }
            )
    except urllib.error.HTTPError as exc:
        result.update(
            {
                "http_status": exc.code,
                "final_url_without_query": sanitized_url(exc.geturl()),
                "content_type": exc.headers.get_content_type(),
                "sample_bytes_read": 0,
                "accessible_without_credentials": False,
                "error": f"HTTP {exc.code}: {exc.reason}",
            }
        )
    except urllib.error.URLError as exc:
        result.update(
            {
                "http_status": None,
                "final_url_without_query": None,
                "sample_bytes_read": 0,
                "accessible_without_credentials": False,
                "error": f"URL error: {exc.reason}",
            }
        )
    return result


def product_block(
    collection: dict[str, Any], granules: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "collection": collection,
        "granule_count": len(granules),
        "tiles": sorted({entry["tile"] for entry in granules}),
        "granules": sorted(granules, key=lambda entry: entry["tile"]),
    }


def availability_status(
    products: dict[str, Any], expected_tiles: set[str] | None = None
) -> str:
    nonempty = [block["granule_count"] > 0 for block in products.values()]
    if not any(nonempty):
        return "unavailable"
    tile_sets = [tuple(block["tiles"]) for block in products.values()]
    complete = all(nonempty) and len(set(tile_sets)) == 1
    if expected_tiles is not None:
        complete = complete and set(tile_sets[0]) == expected_tiles
    if complete:
        return "available"
    return "partial"


def build_plan(
    target: dt.date,
    bbox: tuple[float, float, float, float],
    control_year: int | None,
    probe_control_access: bool,
) -> dict[str, Any]:
    collections = {
        product["short_name"]: query_collection(product) for product in PRODUCTS
    }
    exact_products: dict[str, Any] = {}
    latest: dict[str, Any] = {}
    for name, collection in collections.items():
        exact_products[name] = product_block(
            collection, exact_granules(collection, target, bbox)
        )
        latest[name] = latest_at_or_before(collection, target, bbox)

    exact_block = {
        "role": "target",
        "nominal_date": target.isoformat(),
        "nominal_tag": date_to_tag(target),
        "products": exact_products,
    }

    control_block: dict[str, Any] | None = None
    if control_year is not None:
        try:
            control_date = target.replace(year=control_year)
        except ValueError as exc:
            raise ProbeError(
                "a February 29 target needs an explicitly supported control date"
            ) from exc
        control_products: dict[str, Any] = {}
        for name, collection in collections.items():
            granules = exact_granules(collection, control_date, bbox)
            if probe_control_access:
                for granule in granules:
                    granule["unauthenticated_access_probe"] = access_probe(
                        granule["https_url"]
                    )
            control_products[name] = product_block(collection, granules)
        control_block = {
            "role": "prior_year_same_month_day_control_only",
            "must_not_be_substituted_for_target": True,
            "nominal_date": control_date.isoformat(),
            "nominal_tag": date_to_tag(control_date),
            "products": control_products,
        }

    available_tile_evidence: set[str] = set()
    for block in exact_products.values():
        available_tile_evidence.update(block["tiles"])
    if control_block:
        for block in control_block["products"].values():
            available_tile_evidence.update(block["tiles"])
    for recent in latest.values():
        if recent:
            available_tile_evidence.update(item["tile"] for item in recent["granules"])

    exact_block["status"] = availability_status(
        exact_products, available_tile_evidence
    )
    if control_block:
        control_block["status"] = availability_status(
            control_block["products"], available_tile_evidence
        )

    return {
        "schema": SCHEMA,
        "generator": {
            "path": "experiments/mcd43-nbar/mcd43_probe.py",
            "version": "1.0",
            "sha256": generator_sha256(),
        },
        "queried_at_utc": utc_now(),
        "official_sources_only": True,
        "region": {
            "name": "Maine, adjacent Canada, and Gulf of Maine D0 probe",
            "bbox_wgs84_west_south_east_north": list(bbox),
            "tile_evidence_from_available_granules": sorted(available_tile_evidence),
        },
        "official_endpoints": {
            "cmr_search_api":
                "https://cmr.earthdata.nasa.gov/search/site/docs/search/api.html",
            "cmr_collections": CMR_COLLECTIONS,
            "cmr_granules": CMR_GRANULES,
            "cmr_native_concepts": CMR_CONCEPT,
            "earthdata_login": "https://urs.earthdata.nasa.gov/",
            "lp_daac_cloud_access_examples":
                "https://git.earthdata.nasa.gov/projects/LPDUR/repos/"
                "lpdaac_cloud_data_access/browse",
        },
        "exact_target": exact_block,
        "latest_at_or_before_target": latest,
        "prior_year_control": control_block,
        "decision": {
            "exact_target_ready_for_crop": exact_block["status"] == "available",
            "control_is_not_a_target_substitute": True,
            "download_is_a_separate_explicit_command": True,
        },
    }


def write_json_no_clobber(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    except FileExistsError as exc:
        raise ProbeError(f"refusing to overwrite existing file: {path}") from exc


def plan_summary(plan: dict[str, Any]) -> str:
    exact = plan["exact_target"]
    lines = [
        f"exact target {exact['nominal_tag']}: {exact['status']}",
    ]
    for name, latest in sorted(plan["latest_at_or_before_target"].items()):
        if latest:
            lines.append(
                f"{name} latest: {latest['nominal_tag']} "
                f"({latest['age_days_at_target']} nominal days behind target), "
                f"tiles={','.join(item['tile'] for item in latest['granules'])}"
            )
        else:
            lines.append(f"{name} latest: none in target year")
    control = plan.get("prior_year_control")
    if control:
        lines.append(
            f"control {control['nominal_tag']}: {control['status']} "
            "(control only; never substituted)"
        )
        probes = [
            entry.get("unauthenticated_access_probe", {})
            for block in control["products"].values()
            for entry in block["granules"]
            if "unauthenticated_access_probe" in entry
        ]
        if probes:
            accessible = sum(
                bool(probe.get("accessible_without_credentials")) for probe in probes
            )
            statuses = sorted({str(probe.get("http_status")) for probe in probes})
            lines.append(
                f"control unauthenticated ranged GET: {accessible}/{len(probes)} "
                f"accessible; HTTP statuses={','.join(statuses)}"
            )
    return "\n".join(lines)


def load_plan(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        plan = json.load(handle)
    if plan.get("schema") != SCHEMA:
        raise ProbeError(f"unsupported or missing plan schema in {path}")
    return plan


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def selected_granules(
    plan: dict[str, Any],
    source: str,
    products: set[str],
    tiles: set[str],
) -> list[dict[str, Any]]:
    if source == "target":
        block = plan["exact_target"]
    else:
        block = plan.get("prior_year_control")
        if not block:
            raise ProbeError("the plan contains no prior-year control")
    if block["status"] != "available":
        raise ProbeError(f"selected {source} block is {block['status']}; nothing to download")
    selected: list[dict[str, Any]] = []
    for name, product_block_value in block["products"].items():
        if products and name not in products:
            continue
        for granule in product_block_value["granules"]:
            if tiles and granule["tile"] not in tiles:
                continue
            selected.append(granule)
    if not selected:
        raise ProbeError("product/tile filters selected no granules")
    return sorted(selected, key=lambda item: (item["product"], item["tile"]))


def validate_download_url(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_DOWNLOAD_HOSTS:
        raise ProbeError(f"refusing non-LP-DAAC download URL: {sanitized_url(url)}")
    if parsed.query or parsed.fragment or not parsed.path.endswith(".hdf"):
        raise ProbeError(f"refusing unexpected HDF URL shape: {sanitized_url(url)}")


def validate_hdf_identity(granule: dict[str, Any]) -> None:
    name = granule.get("hdf_name")
    granule_ur = granule.get("granule_ur")
    url = granule.get("https_url")
    if not isinstance(name, str) or not isinstance(granule_ur, str):
        raise ProbeError("plan granule has no HDF name or granule UR")
    parse_granule_name(granule_ur)
    if name != f"{granule_ur}.hdf" or Path(name).name != name or "\\" in name:
        raise ProbeError(f"refusing unexpected HDF filename: {name!r}")
    if not isinstance(url, str):
        raise ProbeError(f"plan granule has no HTTPS URL for {name}")
    url_name = Path(urllib.parse.unquote(urllib.parse.urlsplit(url).path)).name
    if url_name != name:
        raise ProbeError(f"HDF filename does not match its source URL: {name}")


def download_one(
    granule: dict[str, Any],
    output_dir: Path,
    token_env: str | None,
    plan_path: Path,
) -> dict[str, Any]:
    url = granule["https_url"]
    validate_download_url(url)
    validate_hdf_identity(granule)
    expected_name = granule["hdf_name"]
    published_algorithm = granule["hdf_checksum"]["algorithm"]
    published_checksum = granule["hdf_checksum"]["value"]
    expected_size = int(granule["hdf_size_bytes"])
    destination = output_dir / expected_name
    provenance_path = output_dir / f"{expected_name}.provenance.json"
    if destination.exists() or provenance_path.exists():
        raise ProbeError(f"refusing to overwrite {destination} or its provenance")
    output_dir.mkdir(parents=True, exist_ok=True)

    headers: dict[str, str] = {}
    auth_method = "unauthenticated"
    if token_env:
        token = os.environ.get(token_env)
        if not token:
            raise ProbeError(f"environment variable {token_env!r} is empty or missing")
        headers["Authorization"] = f"Bearer {token}"
        auth_method = f"bearer token from environment variable {token_env}"

    temporary = output_dir / f".{expected_name}.part-{os.getpid()}"
    if temporary.exists():
        raise ProbeError(f"refusing to overwrite stale partial file: {temporary}")
    sha256_digest = hashlib.sha256()
    published_digest = hashlib.new(published_algorithm.lower())
    byte_count = 0
    try:
        with _request(url, headers=headers, timeout=120.0) as response:
            content_type = response.headers.get_content_type()
            if response.status != 200 or content_type in {
                "text/html",
                "application/json",
            }:
                raise ProbeError(
                    f"download returned HTTP {response.status} as {content_type}; "
                    "Earthdata authentication may be required"
                )
            with temporary.open("xb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    sha256_digest.update(chunk)
                    published_digest.update(chunk)
                    byte_count += len(chunk)
        actual_sha256 = sha256_digest.hexdigest()
        actual_published_checksum = published_digest.hexdigest()
        if byte_count != expected_size:
            raise ProbeError(
                f"size mismatch for {expected_name}: {byte_count} != {expected_size}"
            )
        if actual_published_checksum != published_checksum:
            raise ProbeError(
                f"{published_algorithm} mismatch for {expected_name}: "
                f"{actual_published_checksum} != {published_checksum}"
            )
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise ProbeError(
                f"refusing to overwrite file created during download: {destination}"
            ) from exc
        except OSError as exc:
            raise ProbeError(
                f"filesystem cannot perform an atomic no-clobber commit for {destination}: {exc}"
            ) from exc
        temporary.unlink()
    except urllib.error.HTTPError as exc:
        if temporary.exists():
            temporary.unlink()
        raise ProbeError(
            f"download failed with HTTP {exc.code} ({exc.reason}); "
            "Earthdata authentication may be required"
        ) from exc
    except urllib.error.URLError as exc:
        if temporary.exists():
            temporary.unlink()
        raise ProbeError(f"download failed: {exc.reason}") from exc
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise

    provenance = {
        "schema": "simsat.mcd43-download-provenance.v1",
        "downloaded_at_utc": utc_now(),
        "source_plan": str(plan_path.resolve()),
        "source_plan_sha256": sha256_file(plan_path),
        "source_url": url,
        "authentication": auth_method,
        "granule_ur": granule["granule_ur"],
        "concept_id": granule["concept_id"],
        "bytes": byte_count,
        "sha256": actual_sha256,
        "nasa_published_checksum": {
            "algorithm": published_algorithm,
            "value": published_checksum,
        },
    }
    write_json_no_clobber(provenance_path, provenance)
    return provenance


def self_check() -> None:
    checks: list[tuple[str, bool]] = []
    checks.append(("target-date-tag", date_to_tag(DEFAULT_TARGET) == "A2026191"))
    checks.append(("leap-date-tag", date_to_tag(dt.date(2024, 2, 29)) == "A2024060"))
    parsed = parse_granule_name("MCD43A4.A2025191.h13v04.061.2025205235824")
    checks.append(("granule-product", parsed["product"] == "MCD43A4"))
    checks.append(("granule-date", parsed["nominal_date"] == "2025-07-10"))
    checks.append(("granule-tile", parsed["tile"] == "h13v04"))
    native_sample = """\
<Granule>
  <GranuleUR>MCD43A4.A2025191.h13v04.061.2025205235824</GranuleUR>
  <DataGranule>
    <ProductionDateTime>2025-07-25T00:01:16Z</ProductionDateTime>
    <LocalVersionId>6.1.34</LocalVersionId>
    <AdditionalFile>
      <Name>MCD43A4.A2025191.h13v04.061.2025205235824.hdf</Name>
      <SizeInBytes>36328315</SizeInBytes>
      <Checksum>
        <Value>a8093d6e3dcdf6410c5618a6e1e7f24a0248fe0049ae293b4abfa2515d59bd8f</Value>
        <Algorithm>SHA-256</Algorithm>
      </Checksum>
    </AdditionalFile>
  </DataGranule>
</Granule>
"""
    native = parse_native_echo10(
        native_sample,
        "https://example.nasa.gov/native",
        "MCD43A4.A2025191.h13v04.061.2025205235824",
    )
    checks.append(("native-size", native["hdf_size_bytes"] == 36_328_315))
    checks.append(("native-sha256", native["hdf_checksum"]["algorithm"] == "SHA256"))
    checks.append(
        (
            "signed-url-sanitization",
            sanitized_url("https://example.nasa.gov/file.hdf?token=secret")
            == "https://example.nasa.gov/file.hdf",
        )
    )
    try:
        validate_download_url(
            "https://data.lpdaac.earthdatacloud.nasa.gov/a/file.hdf"
        )
        allowed_ok = True
    except ProbeError:
        allowed_ok = False
    checks.append(("download-host-allowlist", allowed_ok))
    try:
        validate_download_url("https://example.com/file.hdf")
        rejected_foreign = False
    except ProbeError:
        rejected_foreign = True
    checks.append(("foreign-host-rejected", rejected_foreign))
    try:
        validate_hdf_identity(
            {
                "granule_ur": "MCD43A4.A2025191.h13v04.061.2025205235824",
                "hdf_name": "../escape.hdf",
                "https_url":
                    "https://data.lpdaac.earthdatacloud.nasa.gov/a/escape.hdf",
            }
        )
        rejected_traversal = False
    except ProbeError:
        rejected_traversal = True
    checks.append(("path-traversal-rejected", rejected_traversal))

    with tempfile.TemporaryDirectory() as raw_dir:
        path = Path(raw_dir) / "plan.json"
        write_json_no_clobber(path, {"schema": SCHEMA})
        try:
            write_json_no_clobber(path, {"schema": SCHEMA})
            refused = False
        except ProbeError:
            refused = True
        checks.append(("plan-no-clobber", refused))

    failed = [name for name, passed in checks if not passed]
    for name, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    if failed:
        raise ProbeError(f"self-check failures: {', '.join(failed)}")
    print(f"PASS {len(checks)}/{len(checks)} checks")


def parse_bbox(value: str) -> tuple[float, float, float, float]:
    try:
        parts = tuple(float(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("bbox must contain four numbers") from exc
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must be west,south,east,north")
    west, south, east, north = parts
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise argparse.ArgumentTypeError("bbox bounds or ordering are invalid")
    return west, south, east, north


def parse_date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must be YYYY-MM-DD") from exc


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="query CMR and create a no-clobber plan")
    plan.add_argument("--target-date", type=parse_date, default=DEFAULT_TARGET)
    plan.add_argument(
        "--bbox",
        type=parse_bbox,
        default=DEFAULT_BBOX,
        metavar="WEST,SOUTH,EAST,NORTH",
    )
    plan.add_argument(
        "--control-year",
        type=int,
        help="list the same month/day in a prior year as a labeled control",
    )
    plan.add_argument(
        "--probe-control-access",
        action="store_true",
        help="use a one-byte ranged GET to test unauthenticated control access",
    )
    plan.add_argument("--output", type=Path, help="write JSON with no overwrite")
    plan.add_argument(
        "--allow-target-unavailable",
        action="store_true",
        help="return zero while preserving unavailable status in the plan",
    )

    download = commands.add_parser(
        "download", help="explicitly download and verify files named in a plan"
    )
    download.add_argument("plan", type=Path)
    download.add_argument("--output-dir", type=Path, required=True)
    download.add_argument(
        "--source",
        choices=("target", "control"),
        default="target",
        help="control is an explicit seasonal control, never a fallback",
    )
    download.add_argument(
        "--product", action="append", choices=("MCD43A4", "MCD43A2"), default=[]
    )
    download.add_argument("--tile", action="append", default=[])
    download.add_argument(
        "--token-env",
        help="name of an environment variable holding an Earthdata bearer token",
    )

    commands.add_parser("self-check", help="run dependency-free offline checks")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "self-check":
        self_check()
        return 0
    if args.command == "plan":
        if args.probe_control_access and args.control_year is None:
            raise ProbeError("--probe-control-access requires --control-year")
        plan = build_plan(
            args.target_date,
            args.bbox,
            args.control_year,
            args.probe_control_access,
        )
        if args.output:
            write_json_no_clobber(args.output, plan)
            print(f"wrote {args.output}")
        print(plan_summary(plan))
        if plan["exact_target"]["status"] != "available":
            print(
                "TARGET NOT READY: exact nominal-date MCD43A4+A2 granules are not "
                "available; no control was substituted.",
                file=sys.stderr,
            )
            return 0 if args.allow_target_unavailable else 2
        return 0
    if args.command == "download":
        plan = load_plan(args.plan)
        granules = selected_granules(
            plan, args.source, set(args.product), set(args.tile)
        )
        print(
            f"downloading {len(granules)} explicitly selected {args.source} granules"
        )
        for granule in granules:
            provenance = download_one(
                granule, args.output_dir, args.token_env, args.plan
            )
            print(
                f"verified {granule['hdf_name']} {provenance['bytes']} bytes "
                f"sha256={provenance['sha256']}"
            )
        return 0
    raise ProbeError(f"unknown command: {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProbeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
