# Legal audit — Condensearr

**MIT License.** Copyright (c) 2025 snapetech. See [LICENSE](../LICENSE) in the repo root.

This document is a lightweight legal audit of the project: why MIT is used, dependency compatibility, and sharp corners to be aware of. It is not legal advice.

---

## Is MIT right for this project?

**Yes.** MIT is a permissive license that allows anyone to use, modify, distribute, and sublicense the code (including commercially) as long as they keep the copyright and license notice. It fits a small, reusable CLI tool meant for broad use and automation (including in commercial/Arr pipelines). There is no copyleft: downstream code can stay proprietary if they wish.

**What MIT does not give:** an explicit patent grant. Licenses like Apache 2.0 include a patent license from contributors. For this project (no patent claims, typical video/audio processing), the practical risk is low. If you ever need a clear patent license, consider dual-licensing or switching to Apache 2.0 after checking dependency compatibility.

---

## Dependencies and license compatibility

| Dependency | How used | License | Compatible with MIT? |
|------------|----------|---------|----------------------|
| **Python 3 stdlib** | argparse, json, subprocess, pathlib, etc. | PSF License | Yes. Permissive, no copyleft. |
| **numpy** | Required (array math, signal processing). | BSD-3-Clause | Yes. BSD-3 is permissive; no conflict. |
| **pytesseract** | Optional (OCR wrapper). | Apache 2.0 (typical for wrapper) | Yes. Apache 2.0 is compatible with MIT; preserve their notices if you distribute. |
| **opencv-python** | Optional (auto scorebug ROI). | Apache 2.0 / BSD (OpenCV) | Yes. Permissive. |
| **Tesseract OCR** | Optional (system binary called by pytesseract). | Apache 2.0 | Yes. Not linked; user installs separately. |
| **ffmpeg / ffprobe** | External binaries invoked via subprocess. | LGPL/GPL | Yes for this project. We do not link to FFmpeg; we only launch it as a separate process. That is “mere aggregation” — our script is not a derivative work of FFmpeg, so GPL does not apply to our code. |

**Conclusion:** All current dependencies are compatible with distributing Condensearr under MIT. No copyleft (GPL/LGPL) code is incorporated into or linked by this repo.

---

## Sharp corners and caveats

1. **No patent grant**  
   MIT does not expressly grant patent rights. If you or contributors ever assert patents over the code, consider adding a patent license (e.g. via Apache 2.0 or a separate patent grant).

2. **Do not add GPL/LGPL code**  
   If you copy or link GPL/LGPL code into this repo, the combined work may need to be licensed under GPL, which would conflict with keeping the project “MIT only.” Keeping FFmpeg as an external process (no in-process linking) is the right approach.

3. **Attribution when distributing**  
   MIT requires preserving the copyright and license notice. If you ever ship a bundle (e.g. a PyInstaller binary or Docker image) that includes numpy/opencv/pytesseract, include their license notices too (see NOTICE in the repo root and their respective LICENSE files). For source-only distribution and “pip install” instructions, our LICENSE plus NOTICE is sufficient.

4. **Trademark**  
   “Condensearr” is not claimed as a trademark in the LICENSE. If you want to reserve it, add a short TRADEMARKS or NOTICE section (e.g. “Condensearr is a trademark of …”). Not required for MIT compliance.

5. **Content processed**  
   The script only processes video files the user provides. We do not distribute or claim rights over any third-party content. No additional license needed for “input data.”

---

## Summary

| Question | Answer |
|----------|--------|
| Is MIT appropriate? | Yes. |
| Any license conflict with dependencies? | No. |
| FFmpeg (GPL/LGPL) a problem? | No; used as external process only. |
| Sharp corners? | No patent grant in MIT; don’t add GPL code; keep attribution (and NOTICE) when bundling. |

No changes to the current MIT licensing are required for normal use and distribution of the project.
