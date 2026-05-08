/* glue.h — shared types and prototypes for nanoarianna inference glue.
 *
 * The glue layer sits between an AML organism (yent.aml or resonance.aml)
 * and three external substrates: persona files, per-organism Limpha SQLite
 * memory, and the shared Knowledge Kernel (KK) SQLite.
 *
 *   organism boot    : am_init() -> persona_load() -> am_exec("LOAD …")
 *   pre-forward      : limpha_query_recent() + kk_query_resonant() -> ctx
 *   forward          : (yent_forward / resonance_forward — unchanged upstream)
 *   post-forward     : limpha_append() + kk_append_dialogue() (if dialogue turn)
 *
 * Everything here is C linkage so it compiles cleanly inside an AML
 * BLOOD COMPILE block via amlc.
 */

#ifndef NANOARIANNA_GLUE_H
#define NANOARIANNA_GLUE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ──────────────────────────────────────────────────────────────────────
 *  persona_loader.c
 * ────────────────────────────────────────────────────────────────────── */

/* Load and execute a persona .aml file via libaml's am_exec_file().
 *
 * Resolution order:
 *   1. argument `path` if non-NULL and non-empty
 *   2. environment $PERSONA_AML
 *   3. no-op (returns 0 silently — organism boots with its hardcoded defaults)
 *
 * Returns 0 on success or no-op, non-zero on parse/exec failure. Errors are
 * logged to stderr; caller decides whether to abort.
 */
int persona_load(const char *path);


/* ──────────────────────────────────────────────────────────────────────
 *  limpha.c (Phase 2)
 *
 *  Per-organism SQLite memory. Schema: episodes(ts, organism, prompt,
 *  response, 7-feature inner state, temperature, quality).
 * ────────────────────────────────────────────────────────────────────── */

#define LIMPHA_STATE_DIM 7   /* trauma, arousal, valence, coherence,
                                prophecy_debt, entropy, dissonance */

typedef struct {
    int64_t  id;
    int64_t  ts;
    char     organism[16];
    char    *prompt;       /* malloc'd */
    char    *response;     /* malloc'd */
    float    state[LIMPHA_STATE_DIM];
    float    temperature;
    float    quality;
} limpha_episode_t;

/* Forward-declare opaque handle so the header doesn't pull in sqlite3.h.
 * Implementation casts to sqlite3*. */
typedef struct limpha_db limpha_db;

int  limpha_open(const char *path, limpha_db **out);
void limpha_close(limpha_db *db);

int  limpha_append(limpha_db *db, const char *organism,
                   const char *prompt, const char *response,
                   const float state[LIMPHA_STATE_DIM],
                   float temperature, float quality);

/* Cosine-similarity over 7-feature state. Caller frees each
 * episode.prompt/response with limpha_episode_free(). */
int  limpha_query_similar(limpha_db *db, const float state[LIMPHA_STATE_DIM],
                          int top_k, limpha_episode_t *out, int out_cap);

int  limpha_query_recent(limpha_db *db, const char *organism,
                         int n, limpha_episode_t *out, int out_cap);

void limpha_episode_free(limpha_episode_t *e);


/* ──────────────────────────────────────────────────────────────────────
 *  kk.c (Phase 2)
 *
 *  Shared Knowledge Kernel — documents + cross-organism dialogue +
 *  Hebbian co-occurrence between docs.
 * ────────────────────────────────────────────────────────────────────── */

typedef struct kk_db kk_db;

typedef struct {
    int64_t id;
    int64_t ts;
    char    source[32];        /* 'seed', 'arianna', 'leo', 'oleg', 'mesh', 'reference' */
    char   *title;             /* malloc'd, may be NULL */
    char   *content;           /* malloc'd */
    float   emotional_charge;
    float   resonance_score;
} kk_document_t;

typedef struct {
    int64_t id;
    int64_t ts;
    char    speaker[16];
    char    listener[16];
    char   *prompt;
    char   *response;
    float   prophecy_debt_delta;
    char    dominant_chamber[16];
} kk_dialogue_t;

int  kk_open(const char *path, kk_db **out);
void kk_close(kk_db *db);

int  kk_append_document(kk_db *db, const char *source, const char *title,
                        const char *content, float emotional_charge);

int  kk_append_dialogue(kk_db *db, const char *speaker, const char *listener,
                        const char *prompt, const char *response,
                        float prophecy_debt_delta, const char *dominant_chamber);

int  kk_query_resonant(kk_db *db, const float query_embedding[],
                       int embed_dim, int top_k,
                       kk_document_t *out, int out_cap);

int  kk_query_recent_dialogue(kk_db *db, int n,
                              kk_dialogue_t *out, int out_cap);

void kk_document_free(kk_document_t *d);
void kk_dialogue_free(kk_dialogue_t *d);


/* ──────────────────────────────────────────────────────────────────────
 *  Build constants
 * ────────────────────────────────────────────────────────────────────── */

#define NANOARIANNA_VERSION  "0.1.0-dev"

#ifdef __cplusplus
}
#endif

#endif  /* NANOARIANNA_GLUE_H */
