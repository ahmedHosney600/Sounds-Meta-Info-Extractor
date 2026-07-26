"""extractors/drive_links.py — Google Drive file ID resolution and URL construction."""

from __future__ import annotations


def build_drive_id_map(
    folder_id: str,
    drive_service,
    recursive: bool = True,
) -> dict[str, str]:
    """
    Query the Google Drive API and return a mapping of filename → file_id.

    Also stores relative paths (e.g. ``SubFolder/file.wav → file_id``)
    so files in subdirectories are resolved correctly.

    Parameters
    ----------
    folder_id     : Drive folder ID (from the URL).
    drive_service : Authenticated ``googleapiclient`` Drive v3 service object.
    recursive     : Whether to descend into subfolders.

    Returns
    -------
    dict mapping bare filename strings (and relative paths) to Drive file IDs.
    """
    id_map: dict[str, str] = {}

    def _list_folder(fid: str, prefix: str = "") -> None:
        page_token = None
        while True:
            resp = drive_service.files().list(
                q=f"'{fid}' in parents and trashed=false",
                fields="nextPageToken, files(id, name, mimeType)",
                pageToken=page_token,
                pageSize=1000,
            ).execute()

            for item in resp.get("files", []):
                name  = item["name"]
                ftype = item["mimeType"]
                rel   = f"{prefix}/{name}" if prefix else name

                if ftype == "application/vnd.google-apps.folder":
                    if recursive:
                        _list_folder(item["id"], rel)
                else:
                    id_map[name] = item["id"]   # bare filename
                    id_map[rel]  = item["id"]   # relative path (handles duplicates across folders)

            page_token = resp.get("nextPageToken")
            if not page_token:
                break

    _list_folder(folder_id)
    return id_map


def get_drive_links(filename: str, id_map: dict | None) -> dict:
    """
    Construct Google Drive URLs for a file given the id_map built above.

    URLs returned
    -------------
    drive_file_id      : The raw Drive file ID
    drive_preview_url  : In-browser preview (works for audio, images, PDFs)
    drive_download_url : Direct download link
    drive_stream_url   : Stream/embed-friendly direct URL
    drive_embed_url    : Embeddable iframe-friendly URL
    """
    fid = (id_map or {}).get(filename)
    if not fid:
        return {
            "drive_file_id":      None,
            "drive_preview_url":  None,
            "drive_download_url": None,
            "drive_stream_url":   None,
            "drive_embed_url":    None,
        }
    return {
        "drive_file_id":      fid,
        "drive_preview_url":  f"https://drive.google.com/file/d/{fid}/preview",
        "drive_download_url": f"https://drive.google.com/uc?id={fid}&export=download",
        "drive_stream_url":   f"https://drive.google.com/uc?id={fid}",
        "drive_embed_url":    f"https://drive.google.com/file/d/{fid}/preview?usp=drivesdk",
    }
