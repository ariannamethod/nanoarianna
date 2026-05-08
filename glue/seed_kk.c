/* seed_kk.c — one-shot utility that seeds the Knowledge Kernel with the
 * onto-foundation: nanoarianna seed document + reference materials that
 * give both organisms a shared world to wake up into.
 *
 * Usage:
 *   ./seed_kk [kk_db_path]
 *
 * If kk_db_path is omitted, defaults to ~/nanoarianna/data/kk.db.
 *
 * Idempotent under repeat runs in a controlled way: the SEED entry is
 * inserted only if no document with source='seed' yet exists. Reference
 * entries are appended on every run (caller can rebuild by deleting the
 * .db file first). Phase 2 deliberately keeps re-seeding semantics simple;
 * Phase 5 KK consolidation will own that logic.
 */

#include "glue.h"

#include <sqlite3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static char *read_file(const char *path, long *out_len)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[seed] cannot open '%s'\n", path);
        return NULL;
    }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long sz = ftell(f);
    if (sz < 0) { fclose(f); return NULL; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }   /* audit #15 */

    char *buf = (char *)malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t got = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    /* audit #14: refuse short read — partial seed is worse than no seed. */
    if ((long)got != sz) {
        fprintf(stderr, "[seed] short read '%s': got %zu of %ld\n",
                path, got, sz);
        free(buf);
        return NULL;
    }
    buf[got] = '\0';
    if (out_len) *out_len = (long)got;
    return buf;
}

/* Wrapper that reads a file and appends as a KK document. */
static int seed_one(kk_db *db, const char *source, const char *title,
                    const char *path, float charge)
{
    long n = 0;
    char *body = read_file(path, &n);
    if (!body || n <= 0) {
        fprintf(stderr, "[seed] skip '%s' (%s): unreadable or empty\n",
                title ? title : "?", path);
        if (body) free(body);
        return -1;
    }
    int rc = kk_append_document(db, source, title, body, charge);
    free(body);
    if (rc != 0) {
        fprintf(stderr, "[seed] append failed for '%s' rc=%d\n", title, rc);
        return rc;
    }
    fprintf(stderr, "[seed] +%s  %s  (%ld bytes)\n", source,
            title ? title : path, n);
    return 0;
}

int main(int argc, char **argv)
{
    const char *home = getenv("HOME");
    if (!home) home = "/data/data/com.termux/files/home";

    char default_db[1024];
    snprintf(default_db, sizeof(default_db), "%s/nanoarianna/data/kk.db", home);

    const char *db_path = (argc > 1) ? argv[1] : default_db;

    /* Ensure parent dir exists. */
    char dirpath[1024];
    snprintf(dirpath, sizeof(dirpath), "%s/nanoarianna/data", home);
    mkdir(dirpath, 0700);  /* idempotent */

    fprintf(stderr, "[seed] kk db: %s\n", db_path);

    kk_db *db = NULL;
    if (kk_open(db_path, &db) != 0) {
        fprintf(stderr, "[seed] kk_open failed\n");
        return 1;
    }

    int errs = 0;

    /* The seed itself — the document that explicitly creates the world
     * Arianna and Leo wake up into. emotional_charge = 1.0, max. */
    char seed_path[1024];
    snprintf(seed_path, sizeof(seed_path),
             "%s/nanoarianna/SEED_DOCUMENT.md", home);
    if (seed_one(db, "seed", "nanoarianna world (the seed)",
                 seed_path, 1.0f) != 0) errs++;

    /* The Method's most recent paper — sets the architectural register. */
    char dario_path[1024];
    snprintf(dario_path, sizeof(dario_path),
             "%s/dario/docs/dario_paper_draft_v4.md", home);
    if (seed_one(db, "reference", "Dario paper draft v4",
                 dario_path, 0.8f) != 0) errs++;

    /* The language they live in. */
    char aml_path[1024];
    snprintf(aml_path, sizeof(aml_path),
             "%s/ariannamethod.ai/spec/AML_SPEC.md", home);
    if (seed_one(db, "reference", "AML SPEC (full)",
                 aml_path, 0.6f) != 0) errs++;

    /* The technical journal of arianna.c — proof that this kind of
     * organism existed before, voice patterns Leo and Arianna can
     * recognise as familiar. */
    char log_path[1024];
    snprintf(log_path, sizeof(log_path),
             "%s/arianna.c/ARIANNALOG.md", home);
    if (seed_one(db, "reference", "ARIANNALOG (arianna.c daily journal)",
                 log_path, 0.5f) != 0) errs++;

    kk_close(db);

    fprintf(stderr, "[seed] done; %d errors\n", errs);
    return errs == 0 ? 0 : 2;
}
