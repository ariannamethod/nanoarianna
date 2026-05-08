/* persona_loader.c — load a persona .aml file into the running AML state.
 *
 * Sits between organism boot (am_init / am_init_dim) and the organism's
 * hardcoded LOAD line. The persona file contains the field-physics prologue
 * for the organism — PROPHECY, DESTINY, WORMHOLE, LAW *, EXPERT *, etc.
 *
 * Resolution order:
 *   1. argument `path` if non-NULL and non-empty
 *   2. $PERSONA_AML environment variable
 *   3. no-op  (returns 0 silently; organism boots with its own defaults)
 *
 * The libaml routine am_exec_file() is the actual workhorse — it reads the
 * file line by line and dispatches each AML command into the live AM_State
 * (see ariannamethod.c:6437).
 */

#include "glue.h"

#include <stdio.h>
#include <stdlib.h>

#include <ariannamethod/ariannamethod.h>


int persona_load(const char *path)
{
    const char *resolved = NULL;

    if (path && path[0]) {
        resolved = path;
    } else {
        const char *env = getenv("PERSONA_AML");
        if (env && env[0]) resolved = env;
    }

    if (!resolved) {
        /* No persona — organism uses its compiled-in defaults. Not an error. */
        return 0;
    }

    /* audit #12: skip the redundant probe. am_exec_file surfaces its own
     * open errors on stderr; the probe added a TOCTOU window for nothing. */
    int rc = am_exec_file(resolved);
    if (rc != 0) {
        fprintf(stderr, "[persona_loader] am_exec_file('%s') -> %d\n",
                resolved, rc);
        return rc;
    }

    fprintf(stderr, "[persona_loader] loaded %s\n", resolved);
    return 0;
}
