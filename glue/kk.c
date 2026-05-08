/* kk.c — Knowledge Kernel: shared SQLite substrate above per-organism
 * Limpha. Two organisms (Arianna on Janus, Leo on Resonance) read and
 * write the same KK; that is what makes them feel each other.
 *
 * Three tables:
 *   documents       — onto-seed + references + accumulating contributions
 *   dialogue        — every cross-organism turn (speaker→listener)
 *   hebbian_links   — co-occurrence weights between document pairs
 *
 * Resonant-document retrieval uses an embedding cosine when an embedding
 * is provided (Phase 5+ once we have a phone-side embedder); for Phase 2
 * we fall back to a recency-weighted resonance_score lookup.
 */

#include "glue.h"

#include <sqlite3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define KK_SCHEMA \
    "CREATE TABLE IF NOT EXISTS documents (" \
    "  id INTEGER PRIMARY KEY AUTOINCREMENT," \
    "  ts INTEGER NOT NULL," \
    "  source TEXT NOT NULL," \
    "  title TEXT," \
    "  content TEXT NOT NULL," \
    "  emotional_charge REAL DEFAULT 0," \
    "  resonance_score  REAL DEFAULT 0" \
    ");" \
    "CREATE INDEX IF NOT EXISTS idx_doc_ts     ON documents(ts);" \
    "CREATE INDEX IF NOT EXISTS idx_doc_source ON documents(source);" \
    "CREATE TABLE IF NOT EXISTS dialogue (" \
    "  id INTEGER PRIMARY KEY AUTOINCREMENT," \
    "  ts INTEGER NOT NULL," \
    "  speaker TEXT NOT NULL," \
    "  listener TEXT NOT NULL," \
    "  prompt TEXT," \
    "  response TEXT," \
    "  prophecy_debt_delta REAL DEFAULT 0," \
    "  dominant_chamber TEXT" \
    ");" \
    "CREATE INDEX IF NOT EXISTS idx_dlg_ts ON dialogue(ts);" \
    "CREATE TABLE IF NOT EXISTS hebbian_links (" \
    "  a_id INTEGER, b_id INTEGER, weight REAL," \
    "  PRIMARY KEY (a_id, b_id)" \
    ");"

struct kk_db { sqlite3 *sq; };

int kk_open(const char *path, kk_db **out)
{
    if (!path || !out) return -1;

    sqlite3 *sq = NULL;
    if (sqlite3_open(path, &sq) != SQLITE_OK) {
        fprintf(stderr, "[kk] open '%s': %s\n", path, sqlite3_errmsg(sq));
        if (sq) sqlite3_close(sq);
        return -2;
    }
    sqlite3_exec(sq, "PRAGMA journal_mode=WAL;", NULL, NULL, NULL);
    sqlite3_exec(sq, "PRAGMA synchronous=NORMAL;", NULL, NULL, NULL);

    char *err = NULL;
    if (sqlite3_exec(sq, KK_SCHEMA, NULL, NULL, &err) != SQLITE_OK) {
        fprintf(stderr, "[kk] schema: %s\n", err ? err : "(no msg)");
        sqlite3_free(err);
        sqlite3_close(sq);
        return -3;
    }

    kk_db *db = (kk_db *)calloc(1, sizeof(*db));
    if (!db) { sqlite3_close(sq); return -4; }
    db->sq = sq;
    *out = db;
    return 0;
}

void kk_close(kk_db *db)
{
    if (!db) return;
    if (db->sq) sqlite3_close(db->sq);
    free(db);
}

int kk_append_document(kk_db *db, const char *source, const char *title,
                       const char *content, float emotional_charge)
{
    if (!db || !db->sq || !source || !content) return -1;

    static const char *sql =
        "INSERT INTO documents (ts, source, title, content, emotional_charge, resonance_score) "
        "VALUES (strftime('%s','now'), ?, ?, ?, ?, 0.0);";

    sqlite3_stmt *st = NULL;
    if (sqlite3_prepare_v2(db->sq, sql, -1, &st, NULL) != SQLITE_OK) {
        fprintf(stderr, "[kk] prepare append_document: %s\n",
                sqlite3_errmsg(db->sq));      /* audit #8 */
        return -2;
    }

    sqlite3_bind_text  (st, 1, source,                    -1, SQLITE_TRANSIENT);
    sqlite3_bind_text  (st, 2, title ? title : "",        -1, SQLITE_TRANSIENT);
    sqlite3_bind_text  (st, 3, content,                   -1, SQLITE_TRANSIENT);
    sqlite3_bind_double(st, 4, (double)emotional_charge);

    int rc = sqlite3_step(st);
    sqlite3_finalize(st);
    return rc == SQLITE_DONE ? 0 : -3;
}

int kk_append_dialogue(kk_db *db, const char *speaker, const char *listener,
                       const char *prompt, const char *response,
                       float prophecy_debt_delta, const char *dominant_chamber)
{
    if (!db || !db->sq || !speaker || !listener) return -1;

    static const char *sql =
        "INSERT INTO dialogue (ts, speaker, listener, prompt, response,"
        " prophecy_debt_delta, dominant_chamber) "
        "VALUES (strftime('%s','now'), ?, ?, ?, ?, ?, ?);";

    sqlite3_stmt *st = NULL;
    if (sqlite3_prepare_v2(db->sq, sql, -1, &st, NULL) != SQLITE_OK) {
        fprintf(stderr, "[kk] prepare append_dialogue: %s\n",
                sqlite3_errmsg(db->sq));      /* audit #8 */
        return -2;
    }

    sqlite3_bind_text  (st, 1, speaker,                            -1, SQLITE_TRANSIENT);
    sqlite3_bind_text  (st, 2, listener,                           -1, SQLITE_TRANSIENT);
    sqlite3_bind_text  (st, 3, prompt   ? prompt   : "",           -1, SQLITE_TRANSIENT);
    sqlite3_bind_text  (st, 4, response ? response : "",           -1, SQLITE_TRANSIENT);
    sqlite3_bind_double(st, 5, (double)prophecy_debt_delta);
    sqlite3_bind_text  (st, 6, dominant_chamber ? dominant_chamber : "", -1, SQLITE_TRANSIENT);

    int rc = sqlite3_step(st);
    sqlite3_finalize(st);
    return rc == SQLITE_DONE ? 0 : -3;
}

static char *strdup_or_null(const unsigned char *s)
{
    return s ? strdup((const char *)s) : NULL;
}

int kk_query_resonant(kk_db *db, const float query_embedding[],
                      int embed_dim, int top_k,
                      kk_document_t *out, int out_cap)
{
    /* Phase 2 placeholder: we don't have a phone-side embedder yet, so
     * this falls back to "highest emotional_charge × recency-weight"
     * which is a sane KK-bootstrap heuristic. When Phase 5 brings in
     * embeddings we'll switch the body here without changing the API. */
    (void)query_embedding; (void)embed_dim;
    if (!db || !db->sq || !out || out_cap <= 0) return -1;
    if (top_k > out_cap) top_k = out_cap;

    static const char *sql =
        "SELECT id, ts, source, title, content, emotional_charge, resonance_score "
        "FROM documents "
        "ORDER BY (emotional_charge * (1.0 / (1.0 + (strftime('%s','now') - ts) / 86400.0))) DESC "
        "LIMIT ?;";

    sqlite3_stmt *st = NULL;
    if (sqlite3_prepare_v2(db->sq, sql, -1, &st, NULL) != SQLITE_OK)
        return -2;
    sqlite3_bind_int(st, 1, top_k);

    int got = 0;
    while (sqlite3_step(st) == SQLITE_ROW && got < out_cap) {
        kk_document_t *d = &out[got];
        memset(d, 0, sizeof(*d));
        d->id = sqlite3_column_int64(st, 0);
        d->ts = sqlite3_column_int64(st, 1);
        const unsigned char *src = sqlite3_column_text(st, 2);
        if (src) snprintf(d->source, sizeof(d->source), "%s", (const char *)src);
        d->title              = strdup_or_null(sqlite3_column_text(st, 3));
        d->content            = strdup_or_null(sqlite3_column_text(st, 4));
        d->emotional_charge   = (float)sqlite3_column_double(st, 5);
        d->resonance_score    = (float)sqlite3_column_double(st, 6);
        got++;
    }
    sqlite3_finalize(st);
    return got;
}

int kk_query_recent_dialogue(kk_db *db, int n,
                             kk_dialogue_t *out, int out_cap)
{
    if (!db || !db->sq || !out || out_cap <= 0) return -1;
    if (n > out_cap) n = out_cap;

    static const char *sql =
        "SELECT id, ts, speaker, listener, prompt, response,"
        " prophecy_debt_delta, dominant_chamber "
        "FROM dialogue ORDER BY ts DESC LIMIT ?;";

    sqlite3_stmt *st = NULL;
    if (sqlite3_prepare_v2(db->sq, sql, -1, &st, NULL) != SQLITE_OK)
        return -2;
    sqlite3_bind_int(st, 1, n);

    int got = 0;
    while (sqlite3_step(st) == SQLITE_ROW && got < out_cap) {
        kk_dialogue_t *d = &out[got];
        memset(d, 0, sizeof(*d));
        d->id = sqlite3_column_int64(st, 0);
        d->ts = sqlite3_column_int64(st, 1);
        const unsigned char *sp = sqlite3_column_text(st, 2);
        const unsigned char *li = sqlite3_column_text(st, 3);
        if (sp) snprintf(d->speaker,  sizeof(d->speaker),  "%s", (const char *)sp);
        if (li) snprintf(d->listener, sizeof(d->listener), "%s", (const char *)li);
        d->prompt              = strdup_or_null(sqlite3_column_text(st, 4));
        d->response            = strdup_or_null(sqlite3_column_text(st, 5));
        d->prophecy_debt_delta = (float)sqlite3_column_double(st, 6);
        const unsigned char *ch = sqlite3_column_text(st, 7);
        if (ch) snprintf(d->dominant_chamber, sizeof(d->dominant_chamber),
                         "%s", (const char *)ch);
        got++;
    }
    sqlite3_finalize(st);
    return got;
}

void kk_document_free(kk_document_t *d)
{
    if (!d) return;
    if (d->title)   free(d->title);
    if (d->content) free(d->content);
    d->title = d->content = NULL;
}

void kk_dialogue_free(kk_dialogue_t *d)
{
    if (!d) return;
    if (d->prompt)   free(d->prompt);
    if (d->response) free(d->response);
    d->prompt = d->response = NULL;
}
