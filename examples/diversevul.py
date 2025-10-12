sample = {
  "func": "static int wrap_nettle_hash_fast(gnutls_digest_algorithm_t algo,\n\t\t\t\t const void *text, size_t text_size,\n\t\t\t\t void *digest)\n{\n\tstruct nettle_hash_ctx ctx;\n\tint ret;\n\n\tret = _ctx_init(algo, &ctx);\n\tif (ret < 0)\n\t\treturn gnutls_assert_val(ret);\n\n\tctx.update(&ctx, text_size, text);\n\tctx.digest(&ctx, ctx.length, digest);\n\n\treturn 0;\n}",
  "target": 1,
  "cwe": [
    "CWE-476",
  ],
  "project": "gnutls",
  "commit_id": "3db352734472d851318944db13be73da61300568",
  "hash": 114949019574032720573340450057665032671,
  "size": 16,
  "message": "wrap_nettle_hash_fast: avoid calling _update with zero-length input\n\nAs Nettle's hash update functions internally call memcpy, providing\nzero-length input may cause undefined behavior.\n\nSigned-off-by: Daiki Ueno <ueno@gnu.org>",
}
