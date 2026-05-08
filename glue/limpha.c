/* limpha.c — per-organism persistent SQLite memory.
 *
 * One Limpha database per organism. Schema mirrors arianna.c/limpha/
 * episodes.py simplified into native C. Each turn writes one episode
 * with the 7-feature inner-state vector inline (separate REAL columns —
 * no BLOB, so we can also do plain SQL filters when useful).
 *
 *   trauma, arousal, valence, coherence, prophecy_debt, entropy, dissonance
 *
 * Cosine retrieval over that 7-vector is computed in C after a windowed
 * SELECT — we don't need a vector extension at this scale.
 *
 * Per Oleg's refined Python ban (2026-05-06): inference path = C only.
 * Limpha straddles between turns, but is called from inside the AML
 * BLOOD COMPILE block during organism boot — therefore C+SQLite,
 * not Python+aiosqlite.
 */

#include "glue.h"

#include <sqlite3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define LIMPHA_SCHEMA \
    "CREATE TABLE IF NOT EXISTS episodes (" \
    "  id            INTEGER PRIMARY KEY AUTOINCREMENT," \
    "  ts            INTEGER NOT NULL," \
    "  organism      TEXT    NOT NULL," \
    "  prompt        TEXT," \
    "  response      TEXT," \
    "  trauma        REAL    DEFAULT 0," \
    "  arousal       REAL    DEFAULT 0," \
    "  valence       REAL    DEFAULT 0," \
    "  coherence     REAL    DEFAULT 0," \
    "  prophecy_debt REAL    DEFAULT 0," \
    "  entropy       REAL    DEFAULT 0," \
    "  dissonance    REAL    DEFAULT 0," \
    "  temperature   REAL    DEFAULT 1.0," \
    "  quality       REAL    DEFAULT 0" \
    ");" \
    "CREATE INDEX IF NOT EXISTS idx_ep_ts ON episodes(ts);" \
    "CREATE INDEX IF NOT EXISTS idx_ep_org_ts ON episodes(organism, ts);"

struct limpha_db { sqlite3 *sq; };

int limpha_open(const char *path, limpha_db **out)
{
    if (!path || !out) return -1;

    sqlite3 *sq = NULL;
    if (sqlite3_open(path, &sq) != SQLITE_OK) {
        fprintf(stderr, "[limpha] open '%s': %s\n", path, sqlite3_errmsg(sq));
        if (sq) sqlite3_close(sq);
        return -2;
    }

    /* WAL + sane sync — phone is one-process-at-a-time so contention is nil,
     * WAL just gives us crash-safety when the schedule kills us mid-write. */
    sqlite3_exec(sq, "PRAGMA journal_mode=WAL;", NULL, NULL, NULL);
    sqlite3_exec(sq, "PRAGMA synchronous=NORMAL;", NULL, NULL, NULL);

    char *err = NULL;
    if (sqlite3_exec(sq, LIMPHA_SCHEMA, NULL, NULL, &err) != SQLITE_OK) {
        fprintf(stderr, "[limpha] schema: %s\n", err ? err : "(no msg)");
        sqlite3_free(err);
        sqlite3_close(sq);
        return -3;
    }

    limpha_db *db = (limpha_db *)calloc(1, sizeof(*db));
    if (!db) { sqlite3_close(sq); return -4; }
    db->sq = sq;
    *out = db;
    return 0;
}

void limpha_close(limpha_db *db)
{
    if (!db) return;
    if (db->sq) sqlite3_close(db->sq);
    free(db);
}

int limpha_append(limpha_db *db, const char *organism,
                  const char *prompt, const char *response,
                  const float state[LIMPHA_STATE_DIM],
                  float temperature, float quality)
{
    if (!db || !db->sq || !organism) return -1;

    static const char *sql =
        "INSERT INTO episodes "
        "(ts, organism, prompt, response, trauma, arousal, valence, coherence,"
        " prophecy_debt, entropy, dissonance, temperature, quality) "
        "VALUES (strftime('%s','now'), ?,?,?,?,?,?,?,?,?,?,?,?);";

    sqlite3_stmt *st = NULL;
    if (sqlite3_prepare_v2(db->sq, sql, -1, &st, NULL) != SQLITE_OK) {
        fprintf(stderr, "[limpha] prepare append: %s\n", sqlite3_errmsg(db->sq));
        return -2;
    }

    int i = 1;
    sqlite3_bind_text(st, i++, organism, -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(st, i++, prompt   ? prompt   : "", -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(st, i++, response ? response : "", -1, SQLITE_TRANSIENT);
    /* 7-vector */
    sqlite3_bind_double(st, i++, state ? state[0] : 0.0);
    sqlite3_bind_double(st, i++, state ? state[1] : 0.0);
    sqlite3_bind_double(st, i++, state ? state[2] : 0.0);
    sqlite3_bind_double(st, i++, state ? state[3] : 0.0);
    sqlite3_bind_double(st, i++, state ? state[4] : 0.0);
    sqlite3_bind_double(st, i++, state ? state[5] : 0.0);
    sqlite3_bind_double(st, i++, state ? state[6] : 0.0);
    sqlite3_bind_double(st, i++, (double)temperature);
    sqlite3_bind_double(st, i++, (double)quality);

    int rc = sqlite3_step(st);
    sqlite3_finalize(st);
    if (rc != SQLITE_DONE) {
        fprintf(stderr, "[limpha] step append: %s\n", sqlite3_errmsg(db->sq));
        return -3;
    }
    return 0;
}

/* Helper: hydrate one episode from a stmt cursor. dst's prompt/response are
 * malloc'd; caller frees via limpha_episode_free. */
static void hydrate_episode(sqlite3_stmt *st, limpha_episode_t *dst)
{
    memset(dst, 0, sizeof(*dst));
    dst->id = sqlite3_column_int64(st, 0);
    dst->ts = sqlite3_column_int64(st, 1);
    const unsigned char *org = sqlite3_column_text(st, 2);
    if (org) snprintf(dst->organism, sizeof(dst->organism), "%s", (const char *)org);
    const unsigned char *p = sqlite3_column_text(st, 3);
    const unsigned char *r = sqlite3_column_text(st, 4);
    dst->prompt   = p ? strdup((const char *)p) : NULL;
    dst->response = r ? strdup((const char *)r) : NULL;
    dst->state[0] = (float)sqlite3_column_double(st,  5);
    dst->state[1] = (float)sqlite3_column_double(st,  6);
    dst->state[2] = (float)sqlite3_column_double(st,  7);
    dst->state[3] = (float)sqlite3_column_double(st,  8);
    dst->state[4] = (float)sqlite3_column_double(st,  9);
    dst->state[5] = (float)sqlite3_column_double(st, 10);
    dst->state[6] = (float)sqlite3_column_double(st, 11);
    dst->temperature = (float)sqlite3_column_double(st, 12);
    dst->quality     = (float)sqlite3_column_double(st, 13);
}

int limpha_query_recent(limpha_db *db, const char *organism,
                        int n, limpha_episode_t *out, int out_cap)
{
    if (!db || !db->sq || !out || out_cap <= 0) return -1;
    if (n > out_cap) n = out_cap;

    static const char *sql =
        "SELECT id, ts, organism, prompt, response, "
        "trauma, arousal, valence, coherence, prophecy_debt, entropy, dissonance,"
        " temperature, quality "
        "FROM episodes ";

    char full[512];
    if (organism && organism[0])
        snprintf(full, sizeof(full),
                 "%s WHERE organism = ? ORDER BY ts DESC LIMIT ?;", sql);
    else
        snprintf(full, sizeof(full), "%s ORDER BY ts DESC LIMIT ?;", sql);

    sqlite3_stmt *st = NULL;
    if (sqlite3_prepare_v2(db->sq, full, -1, &st, NULL) != SQLITE_OK)
        return -2;

    int idx = 1;
    if (organism && organism[0])
        sqlite3_bind_text(st, idx++, organism, -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(st, idx++, n);

    int got = 0;
    while (sqlite3_step(st) == SQLITE_ROW && got < out_cap) {
        hydrate_episode(st, &out[got]);
        got++;
    }
    sqlite3_finalize(st);
    return got;
}

/* Cosine similarity over 7-dim state vector. Pulls a window of recent rows
 * (last 200 by default), scores them in C, sorts by descending similarity,
 * returns top_k. */
int limpha_query_similar(limpha_db *db, const float state[LIMPHA_STATE_DIM],
                         int top_k, limpha_episode_t *out, int out_cap)
{
    if (!db || !db->sq || !state || !out || out_cap <= 0) return -1;
    if (top_k > out_cap) top_k = out_cap;

    /* Pull window. 200 rows is plenty for an organism that talks 8x/day. */
    static const int WINDOW = 200;
    limpha_episode_t *buf = (limpha_episode_t *)calloc(WINDOW, sizeof(*buf));
    if (!buf) return -2;

    int got = limpha_query_recent(db, NULL, WINDOW, buf, WINDOW);
    if (got <= 0) { free(buf); return got; }

    /* Compute cosine = dot / (|a| * |b|) */
    double q_norm = 0.0;
    for (int i = 0; i < LIMPHA_STATE_DIM; i++) q_norm += (double)state[i] * state[i];
    q_norm = sqrt(q_norm);
    if (q_norm < 1e-9) q_norm = 1e-9;

    /* score each row, store score in quality field temporarily for sorting */
    float *scores = (float *)calloc(got, sizeof(float));
    int *idx = (int *)calloc(got, sizeof(int));
    if (!scores || !idx) { free(buf); free(scores); free(idx); return -3; }

    for (int i = 0; i < got; i++) {
        idx[i] = i;
        double dot = 0.0, n = 0.0;
        for (int k = 0; k < LIMPHA_STATE_DIM; k++) {
            dot += (double)state[k] * buf[i].state[k];
            n   += (double)buf[i].state[k] * buf[i].state[k];
        }
        n = sqrt(n);
        if (n < 1e-9) n = 1e-9;
        scores[i] = (float)(dot / (q_norm * n));
    }

    /* simple selection sort — O(got*top_k); got is small, fine */
    int picked = 0;
    int chosen[200] = {0};
    while (picked < top_k && picked < got) {
        int best = -1;
        float bs = -2.0f;
        for (int i = 0; i < got; i++) {
            if (chosen[i]) continue;
            if (scores[i] > bs) { bs = scores[i]; best = i; }
        }
        if (best < 0) break;
        chosen[best] = 1;
        out[picked] = buf[best];                /* shallow move — we own the malloc'd p/r */
        memset(&buf[best], 0, sizeof(buf[best])); /* avoid double free */
        picked++;
    }

    /* free everything still in buf */
    for (int i = 0; i < got; i++) limpha_episode_free(&buf[i]);
    free(buf);
    free(scores);
    free(idx);
    return picked;
}

void limpha_episode_free(limpha_episode_t *e)
{
    if (!e) return;
    if (e->prompt)   free(e->prompt);
    if (e->response) free(e->response);
    e->prompt = e->response = NULL;
}
