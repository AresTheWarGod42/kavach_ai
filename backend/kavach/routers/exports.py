from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from kavach.config import settings
from kavach.runtime import REQUEST_COUNT
from kavach.security import require_jwt


router = APIRouter(prefix="/exports", tags=["exports"], dependencies=[Depends(require_jwt)])


@router.get("/dataset1-test")
def export_dataset1_test() -> FileResponse:
    REQUEST_COUNT.labels(route="/exports/dataset1-test", method="GET").inc()
    if not settings.dataset1_export_path.exists():
        raise HTTPException(status_code=404, detail="Dataset 1 test export not found")
    return FileResponse(settings.dataset1_export_path, filename=settings.dataset1_export_filename, media_type="text/csv")


@router.get("/dataset2-test")
def export_dataset2_test() -> FileResponse:
    REQUEST_COUNT.labels(route="/exports/dataset2-test", method="GET").inc()
    if not settings.dataset2_export_path.exists():
        raise HTTPException(status_code=404, detail="Dataset 2 test export not found")
    return FileResponse(settings.dataset2_export_path, filename=settings.dataset2_export_filename, media_type="text/csv")

