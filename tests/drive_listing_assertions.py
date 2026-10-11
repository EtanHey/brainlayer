from collections.abc import Iterable


def snapshot_non_trashed_drive_ids(service, *, folder_id: str) -> set[str]:
    """Read every visible object ID from a fake Drive folder listing."""
    visible: set[str] = set()
    page_token = None
    while True:
        result = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed = false",
                spaces="drive",
                fields="nextPageToken,files(id,name)",
                pageSize=1000,
                pageToken=page_token,
                supportsAllDrives=True,
            )
            .execute()
        )
        visible.update(item["id"] for item in result.get("files", []))
        page_token = result.get("nextPageToken")
        if not page_token:
            return visible


def assert_non_trashed_drive_files_only_grow_or_are_trashed(
    before: set[str],
    after: set[str],
    *,
    trashed_ids: Iterable[str] = (),
) -> None:
    expected_removed = set(trashed_ids)
    actual_removed = before - after
    assert actual_removed == expected_removed
    assert expected_removed <= before
