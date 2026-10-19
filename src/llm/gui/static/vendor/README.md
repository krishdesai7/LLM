# Vendored browser libraries

Committed copies so the page works offline and cannot change under us.
No build step: they are loaded with plain `<script>` tags.

| file | package | version | source | SHA-256 (base64) |
| --- | --- | --- | --- | --- |
| `marked.umd.js` | [marked](https://github.com/markedjs/marked) (MIT) | 18.1.1 | `https://cdn.jsdelivr.net/npm/marked@18.1.1/lib/marked.umd.js` | `uCuTNEBti90WtiNsCN4ctlsqhZhzypUD2cwLKSyjwSo=` |
| `purify.min.js` | [DOMPurify](https://github.com/cure53/DOMPurify) (Apache-2.0 / MPL-2.0) | 3.4.17 | `https://cdn.jsdelivr.net/npm/dompurify@3.4.17/dist/purify.min.js` | `zTeJj5eCrfVIIMQTGRtWkYA/DxN8eZ1u3P04UjtMLDE=` |

Both hashes match the ones jsDelivr publishes for the npm package contents
(`https://data.jsdelivr.com/v1/packages/npm/<name>@<version>?structure=flat`).

To update: download the new version, check its hash against that API, and
update this table.
