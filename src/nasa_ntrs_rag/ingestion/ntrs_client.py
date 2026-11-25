"""Client for downloading open-access technical reports and metadata from NASA NTRS."""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx

from nasa_ntrs_rag.config import Settings, get_settings
from nasa_ntrs_rag.schemas import NTRSDocumentMetadata

logger = logging.getLogger(__name__)

DEFAULT_SAFETY_TOPICS: list[str] = [
    "aviation safety",
    "aircraft icing",
    "flight control",
    "cockpit human factors",
    "runway safety",
    "turbine engine",
    "aerodynamic stall",
    "avionics software",
    "thermal protection",
    "structural fatigue",
    "air traffic management",
    "unmanned aircraft",
    "rocket propulsion",
    "space shuttle safety",
    "wind shear turbulence",
    "composite structures",
    "life support spaceflight",
    "hypersonic aerodynamics",
    "rotorcraft helicopter",
    "reliability hazard analysis",
]


class NTRSClient:
    """HTTP client for the NASA Technical Reports Server (NTRS) Open API."""

    def __init__(self, settings: Settings | None = None, max_connections: int = 20) -> None:
        self.settings = settings or get_settings()
        self.base_url = self.settings.ntrs_api_base.rstrip("/")
        self._client = httpx.Client(
            timeout=self.settings.ntrs_request_timeout,
            follow_redirects=True,
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=min(10, max_connections),
            ),
            headers={"User-Agent": "nasa-ntrs-rag/0.1 (portfolio research project)"},
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> NTRSClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def search_citations(
        self,
        query: str,
        size: int = 20,
        page_from: int = 0,
        only_fulltext_pdf: bool = True,
    ) -> list[dict[str, Any]]:
        """Query the NTRS search endpoint and return raw citation dicts."""
        params: dict[str, Any] = {
            "q": query,
            "page[size]": size,
            "page[from]": page_from,
        }
        if only_fulltext_pdf:
            params["downloadsAvailable"] = "true"

        resp = self._client.get(f"{self.base_url}/search", params=params)
        resp.raise_for_status()
        payload = resp.json()
        return list(payload.get("results", []))

    def _resolve_pdf_url(self, item: dict[str, Any]) -> str | None:
        """Extract the best full-text PDF download URL from a citation record."""
        downloads = item.get("downloads") or []
        for dl in downloads:
            links = dl.get("links") or {}
            pdf_link = links.get("pdf") or links.get("fulltext") or links.get("original")
            if pdf_link and str(pdf_link).lower().endswith(".pdf"):
                if str(pdf_link).startswith("http"):
                    return str(pdf_link)
                return f"https://ntrs.nasa.gov{pdf_link}"

            name = str(dl.get("name", ""))
            if name.lower().endswith(".pdf") and item.get("id"):
                return f"{self.base_url}/{item['id']}/downloads/{name}"

        doc_id = item.get("id")
        if doc_id and item.get("downloadsAvailable"):
            return f"{self.base_url}/{doc_id}/downloads/{doc_id}.pdf"
        return None

    def normalize_metadata(
        self, item: dict[str, Any], topic_query: str = ""
    ) -> NTRSDocumentMetadata:
        """Normalize raw NTRS JSON into a clean NTRSDocumentMetadata schema."""
        doc_id = str(item["id"])
        pub_year: int | None = None
        publications = item.get("publications") or []
        if publications and publications[0].get("publicationDate"):
            try:
                pub_year = int(str(publications[0]["publicationDate"])[:4])
            except ValueError:
                pub_year = None
        if pub_year is None and item.get("distributionDate"):
            try:
                pub_year = int(str(item["distributionDate"])[:4])
            except ValueError:
                pub_year = None

        authors: list[str] = []
        for aff in item.get("authorAffiliations") or []:
            meta = aff.get("meta") or {}
            author_name = (meta.get("author") or {}).get("name")
            if author_name:
                authors.append(str(author_name))

        center_obj = item.get("center") or {}
        center = center_obj.get("name") or center_obj.get("code")

        return NTRSDocumentMetadata(
            doc_id=doc_id,
            title=str(item.get("title") or f"NASA NTRS {doc_id}").strip(),
            abstract=str(item.get("abstract") or "").strip(),
            publication_year=pub_year,
            center=str(center) if center else None,
            sti_type=str(item.get("stiType")) if item.get("stiType") else None,
            subject_categories=[str(s) for s in (item.get("subjectCategories") or [])],
            keywords=[str(k) for k in (item.get("keywords") or [])],
            authors=authors,
            topic_query=topic_query,
            pdf_url=self._resolve_pdf_url(item),
            ntrs_url=f"https://ntrs.nasa.gov/citations/{doc_id}",
        )

    def download_pdf(
        self,
        pdf_url: str,
        dest_path: Path,
        max_pdf_bytes: int | None = None,
    ) -> bool:
        """Stream a PDF to disk, verifying the payload starts with %PDF and obeys size caps."""
        if dest_path.exists() and dest_path.stat().st_size > 1024:
            return True

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dest_path.with_suffix(".pdf.part")
        for attempt in range(2):
            try:
                with self._client.stream("GET", pdf_url) as resp:
                    if resp.status_code in (429, 502, 503, 504) and attempt == 0:
                        time.sleep(1.5)
                        continue
                    if resp.status_code != 200:
                        logger.warning("Skipping %s (HTTP %d)", pdf_url, resp.status_code)
                        return False
                    if max_pdf_bytes is not None:
                        content_length = resp.headers.get("content-length")
                        if (
                            content_length
                            and content_length.isdigit()
                            and int(content_length) > max_pdf_bytes
                        ):
                            logger.info(
                                "Skipping %s (%.1f MB exceeds max_pdf_mb cap)",
                                pdf_url,
                                int(content_length) / (1024 * 1024),
                            )
                            return False
                    first_chunk = True
                    written = 0
                    with tmp_path.open("wb") as fh:
                        for chunk in resp.iter_bytes(chunk_size=65536):
                            if not chunk:
                                continue
                            if first_chunk:
                                first_chunk = False
                                if not chunk.lstrip().startswith(b"%PDF"):
                                    logger.warning("Response at %s is not a valid PDF", pdf_url)
                                    return False
                            written += len(chunk)
                            if max_pdf_bytes is not None and written > max_pdf_bytes:
                                logger.info(
                                    "Aborting %s (exceeded max_pdf_mb cap while streaming)",
                                    pdf_url,
                                )
                                return False
                            fh.write(chunk)
                    if written <= 1024:
                        return False
                tmp_path.replace(dest_path)
                return True
            except httpx.HTTPError as exc:
                if attempt == 0:
                    time.sleep(1.5)
                    continue
                logger.warning("Download failed for %s: %s", pdf_url, exc)
                return False
            finally:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
        return False

    def _current_pdf_bytes(self) -> int:
        if not self.settings.raw_pdf_dir.exists():
            return 0
        return sum(p.stat().st_size for p in self.settings.raw_pdf_dir.glob("*.pdf") if p.is_file())

    def build_corpus(
        self,
        topics: list[str] | None = None,
        per_topic: int = 10,
        max_total_docs: int = 50,
        sleep_seconds: float = 0.25,
        max_total_gb: float | None = None,
        max_pdf_mb: float | None = 80.0,
        workers: int = 6,
    ) -> list[NTRSDocumentMetadata]:
        """Search across aerospace topics in balanced rounds, download PDFs, and update manifest.jsonl."""
        self.settings.ensure_directories()
        topics = topics or DEFAULT_SAFETY_TOPICS

        existing = load_manifest(self.settings.manifest_path)
        seen_ids: set[str] = {doc.doc_id for doc in existing}
        collected: list[NTRSDocumentMetadata] = list(existing)

        max_total_bytes = int(max_total_gb * (1024**3)) if max_total_gb is not None else None
        max_pdf_bytes = int(max_pdf_mb * (1024**2)) if max_pdf_mb is not None else None
        total_bytes = self._current_pdf_bytes()

        topic_added: dict[str, int] = {t: 0 for t in topics}
        topic_offset: dict[str, int] = {t: 0 for t in topics}
        topic_exhausted: dict[str, bool] = {t: False for t in topics}
        topic_queue: dict[str, list[dict[str, Any]]] = {t: [] for t in topics}

        # Round-robin across all topics so size/doc caps always produce a balanced corpus
        round_quota = min(per_topic, max(10, workers * 3))
        page_size = 100

        def _cap_reached() -> bool:
            if len(collected) >= max_total_docs:
                return True
            if max_total_bytes is not None and total_bytes >= max_total_bytes:
                return True
            return False

        def _fetch_one(
            item: dict[str, Any], topic: str
        ) -> tuple[NTRSDocumentMetadata, dict[str, Any], int] | None:
            meta = self.normalize_metadata(item, topic_query=topic)
            if not meta.pdf_url:
                return None
            pdf_path = self.settings.raw_pdf_dir / f"{meta.doc_id}.pdf"
            if not self.download_pdf(meta.pdf_url, pdf_path, max_pdf_bytes=max_pdf_bytes):
                return None
            meta.pdf_path = str(pdf_path)
            size_bytes = pdf_path.stat().st_size if pdf_path.exists() else 0
            return meta, item, size_bytes

        while not _cap_reached():
            made_progress = False
            for topic in topics:
                if _cap_reached():
                    break
                if topic_added[topic] >= per_topic:
                    continue
                if topic_exhausted[topic] and not topic_queue[topic]:
                    continue

                remaining_for_topic = per_topic - topic_added[topic]
                remaining_total = max_total_docs - len(collected)
                target_this_round = min(round_quota, remaining_for_topic, remaining_total)
                if target_this_round <= 0:
                    continue

                # Refill persistent per-topic candidate queue via paginated search
                queue = topic_queue[topic]
                needed_candidates = int(target_this_round * 1.5) + 4
                while len(queue) < needed_candidates and not topic_exhausted[topic]:
                    try:
                        raw_items = self.search_citations(
                            query=topic,
                            size=page_size,
                            page_from=topic_offset[topic],
                        )
                    except httpx.HTTPError as exc:
                        logger.warning(
                            "Search failed for topic '%s' at offset %d: %s",
                            topic,
                            topic_offset[topic],
                            exc,
                        )
                        topic_exhausted[topic] = True
                        break

                    if not raw_items:
                        topic_exhausted[topic] = True
                        break

                    topic_offset[topic] += len(raw_items)
                    if len(raw_items) < page_size:
                        topic_exhausted[topic] = True

                    for item in raw_items:
                        doc_id = str(item.get("id", ""))
                        if not doc_id or doc_id in seen_ids:
                            continue
                        if not self._resolve_pdf_url(item):
                            continue
                        seen_ids.add(doc_id)
                        queue.append(item)

                if not queue:
                    continue

                logger.info(
                    "Topic '%s': downloading up to %d (queued: %d | corpus: %d docs, %.2f GB)",
                    topic,
                    target_this_round,
                    len(queue),
                    len(collected),
                    total_bytes / (1024**3),
                )

                added_round = 0
                if workers <= 1:
                    while queue and not _cap_reached() and added_round < target_this_round:
                        item = queue.pop(0)
                        result = _fetch_one(item, topic)
                        if result is None:
                            continue
                        meta, raw_item, size_bytes = result
                        raw_json_path = self.settings.raw_metadata_dir / f"{meta.doc_id}.json"
                        raw_json_path.write_text(json.dumps(raw_item, indent=2), encoding="utf-8")
                        collected.append(meta)
                        total_bytes += size_bytes
                        topic_added[topic] += 1
                        added_round += 1
                        made_progress = True
                        if sleep_seconds > 0:
                            time.sleep(sleep_seconds)
                else:
                    # Submit in controlled waves, leaving unused candidates in queue for future rounds
                    wave_size = max(workers * 2, 8)
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        while queue and not _cap_reached() and added_round < target_this_round:
                            batch_limit = min(wave_size, target_this_round - added_round + workers)
                            wave = queue[:batch_limit]
                            del queue[:batch_limit]
                            futures = [pool.submit(_fetch_one, item, topic) for item in wave]
                            for fut in as_completed(futures):
                                result = fut.result()
                                if result is None:
                                    continue
                                meta, raw_item, size_bytes = result
                                raw_json_path = self.settings.raw_metadata_dir / f"{meta.doc_id}.json"
                                raw_json_path.write_text(json.dumps(raw_item, indent=2), encoding="utf-8")
                                collected.append(meta)
                                total_bytes += size_bytes
                                topic_added[topic] += 1
                                added_round += 1
                                made_progress = True
                            save_manifest(self.settings.manifest_path, collected)

                if added_round > 0:
                    save_manifest(self.settings.manifest_path, collected)
                    logger.info(
                        "Progress: %d docs | %.2f GB%s",
                        len(collected),
                        total_bytes / (1024**3),
                        f" / {max_total_gb:.2f} GB" if max_total_gb is not None else "",
                    )

            if not made_progress:
                break

        save_manifest(self.settings.manifest_path, collected)
        return collected


def load_manifest(manifest_path: Path) -> list[NTRSDocumentMetadata]:
    """Load NTRS document records from manifest.jsonl if it exists."""
    if not manifest_path.exists():
        return []
    docs: list[NTRSDocumentMetadata] = []
    with manifest_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                docs.append(NTRSDocumentMetadata.model_validate_json(line))
    return docs


def save_manifest(manifest_path: Path, docs: list[NTRSDocumentMetadata]) -> None:
    """Persist NTRS document metadata to manifest.jsonl."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(doc.model_dump_json() + "\n")
