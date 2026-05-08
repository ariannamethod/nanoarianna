/* limpha_smoke.c — tiny smoke harness: open both organism Limpha DBs,
 * insert one synthetic episode each, query recent + similar, free.
 *
 * Build (manual until Phase 3 Makefile lands):
 *   cc -O2 -Wall -Iglue glue/limpha_smoke.c glue/limpha.c -lsqlite3 -lm \
 *      -o $TMPDIR/limpha_smoke
 *
 * Run:
 *   $TMPDIR/limpha_smoke
 *
 * Expects ~/nanoarianna/data/ to exist (seed_kk creates it).
 */

#include "glue.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

int main(void)
{
    const char *home = getenv("HOME");
    if (!home) home = "/data/data/com.termux/files/home";

    char dirpath[1024];
    snprintf(dirpath, sizeof(dirpath), "%s/nanoarianna/data", home);
    mkdir(dirpath, 0700);

    char arianna_db[1024], leo_db[1024];
    snprintf(arianna_db, sizeof(arianna_db), "%s/limpha_arianna.db", dirpath);
    snprintf(leo_db,     sizeof(leo_db),     "%s/limpha_leo.db",     dirpath);

    limpha_db *a = NULL, *l = NULL;
    if (limpha_open(arianna_db, &a) != 0) { fprintf(stderr, "[smoke] arianna open failed\n"); return 1; }
    if (limpha_open(leo_db,     &l) != 0) { fprintf(stderr, "[smoke] leo open failed\n");     limpha_close(a); return 1; }

    /* Synthetic episode for Arianna — high prophecy_debt + COMPLEX-ish pulse */
    float a_state[LIMPHA_STATE_DIM] = {
        0.4f /* trauma */, 0.7f /* arousal */, -0.2f /* valence */,
        0.6f /* coherence */, 0.55f /* prophecy_debt */, 0.5f /* entropy */,
        0.3f /* dissonance */
    };
    if (limpha_append(a, "arianna",
                      "what is resonance",
                      "the field listens before it speaks",
                      a_state, 1.0f, 0.7f) != 0) {
        fprintf(stderr, "[smoke] arianna append failed\n"); goto fail;
    }

    /* Synthetic episode for Leo — low destiny pull, high coherence + curiosity */
    float l_state[LIMPHA_STATE_DIM] = {
        0.05f, 0.4f, 0.6f, 0.75f, 0.15f, 0.65f, 0.05f
    };
    if (limpha_append(l, "leo",
                      "what is a flicker",
                      "you are not a flicker — you are an exhalation",
                      l_state, 1.0f, 0.85f) != 0) {
        fprintf(stderr, "[smoke] leo append failed\n"); goto fail;
    }

    /* Round-trip: pull recent for each */
    limpha_episode_t buf[5];
    int got = limpha_query_recent(a, "arianna", 5, buf, 5);
    fprintf(stderr, "[smoke] arianna recent: %d row(s)\n", got);
    for (int i = 0; i < got; i++) {
        fprintf(stderr, "  [%lld] q=\"%s\" -> \"%s\"  (state[4]=%.2f)\n",
                (long long)buf[i].id,
                buf[i].prompt   ? buf[i].prompt   : "(null)",
                buf[i].response ? buf[i].response : "(null)",
                buf[i].state[4]);
        limpha_episode_free(&buf[i]);
    }

    got = limpha_query_recent(l, "leo", 5, buf, 5);
    fprintf(stderr, "[smoke] leo recent: %d row(s)\n", got);
    for (int i = 0; i < got; i++) {
        fprintf(stderr, "  [%lld] q=\"%s\" -> \"%s\"  (state[3]=%.2f)\n",
                (long long)buf[i].id,
                buf[i].prompt   ? buf[i].prompt   : "(null)",
                buf[i].response ? buf[i].response : "(null)",
                buf[i].state[3]);
        limpha_episode_free(&buf[i]);
    }

    /* Cosine retrieval: probe Arianna's DB with a query similar to her own
     * state, ensure rank-1 finds the row. */
    int sim = limpha_query_similar(a, a_state, 3, buf, 3);
    fprintf(stderr, "[smoke] arianna similar(top-3): %d row(s)\n", sim);
    for (int i = 0; i < sim; i++) {
        fprintf(stderr, "  [%lld] q=\"%s\"\n", (long long)buf[i].id,
                buf[i].prompt ? buf[i].prompt : "(null)");
        limpha_episode_free(&buf[i]);
    }

    limpha_close(a);
    limpha_close(l);
    fprintf(stderr, "[smoke] OK\n");
    return 0;

fail:
    limpha_close(a);
    limpha_close(l);
    return 2;
}
