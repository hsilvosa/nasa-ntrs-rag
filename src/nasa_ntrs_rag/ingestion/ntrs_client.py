"""Client for downloading open-access technical reports and metadata from NASA NTRS."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx

from nasa_ntrs_rag.config import Settings, get_settings
from nasa_ntrs_rag.schemas import NTRSDocumentMetadata

logger = logging.getLogger(__name__)

DEFAULT_SAFETY_TOPICS: list[str] = [
    "aviation safety accident investigation",
    "aircraft icing aerodynamics loss of control",
    "flight control system failure fault tolerance",
    "cockpit human factors pilot workload",
    "runway excursion landing safety",
    "turbine engine failure rotorcraft propulsion",
    "stall recovery upset prevention",
    "avionics software verification safety",
]


class NTRSClient:
    """HTTP client for the NASA Technical Reports Server (NTRS) Open API."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.base_url = self.settings.ntrs_api_base.rstrip("/")
        self._client = httpx.Client(
            timeout=self.settings.ntrs_request_timeout,
            follow_redirects=True,
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

    def download_pdf(self, pdf_url: str, dest_path: Path) -> bool:
        """Stream a PDF to disk, verifying the payload starts with %PDF."""
        if dest_path.exists() and dest_path.stat().st_size > 1024:
            return True

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dest_path.with_suffix(".pdf.part")
        try:
            with self._client.stream("GET", pdf_url) as resp:
                if resp.status_code != 200:
                    logger.warning("Skipping %s (HTTP %d)", pdf_url, resp.status_code)
                    return False
                first_chunk = True
                with tmp_path.open("wb") as fh:
                    for chunk in resp.iter_bytes(chunk_size=16384):
                        if not chunk:
                            continue
                        if first_chunk:
                            first_chunk = False
                            if not chunk.lstrip().startswith(b"%PDF"):
                                logger.warning("Response at %s is not a valid PDF", pdf_url)
                                return False
                        fh.write(chunk)
            tmp_path.replace(dest_path)
            return True
        except httpx.HTTPError as exc:
            logger.warning("Download failed for %s: %s", pdf_url, exc)
            return False
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def build_corpus(
        self,
        topics: list[str] | None = None,
        per_topic: int = 10,
        max_total_docs: int = 50,
        sleep_seconds: float = 0.25,
    ) -> list[NTRSDocumentMetadata]:
        """Search across aerospace safety topics, download PDFs, and update manifest.jsonl."""
        self.settings.ensure_directories()
        topics = topics or DEFAULT_SAFETY_TOPICS

        existing = load_manifest(self.settings.manifest_path)
        seen_ids: set[str] = {doc.doc_id for doc in existing}
        collected: list[NTRSDocumentMetadata] = list(existing)

        for topic in topics:
            if len(collected) >= max_total_docs:
                break
            logger.info("Searching NASA NTRS for topic: %s", topic)
            raw_items = self.search_citations(query=topic, size=per_topic * 2)
            added_for_topic = 0

            for item in raw_items:
                if len(collected) >= max_total_docs or added_for_topic >= per_topic:
                    break
                doc_id = str(item.get("id", ""))
                if not doc_id or doc_id in seen_ids:
                    continue

                meta = self.normalize_metadata(item, topic_query=topic)
                if not meta.pdf_url:
                    continue

                pdf_path = self.settings.raw_pdf_dir / f"{doc_id}.pdf"
                if not self.download_pdf(meta.pdf_url, pdf_path):
                    continue

                meta.pdf_path = str(pdf_path)
                raw_json_path = self.settings.raw_metadata_dir / f"{doc_id}.json"
                raw_json_path.write_text(json.dumps(item, indent=2), encoding="utf-8")

                seen_ids.add(doc_id)
                collected.append(meta)
                added_for_topic += 1
                save_manifest(self.settings.manifest_path, collected)
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)

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
